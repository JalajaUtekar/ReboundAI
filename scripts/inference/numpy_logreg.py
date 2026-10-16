"""Pure-NumPy inference path for ScholarTrack's PRODUCTION model (Step 5a).

This is the model that actually ships. Per `models/benchmark_report.md`
(Step 3's final result) and `scripts/training/explainability.py::
train_production_model`, the deployed model is a scikit-learn
`LogisticRegression(C=0.1, class_weight="balanced", penalty="l2",
max_iter=1000)` fit on `StandardScaler`-standardized features, with
decision threshold 0.90.

`sklearn.linear_model.LogisticRegression.predict_proba` for a fitted
binary model is exactly:

    sigmoid(X_scaled @ coef_.T + intercept_)

and `StandardScaler.transform` is exactly `(X - mean_) / scale_`. Both are
simple enough that re-implementing them by hand in plain NumPy reproduces
sklearn's output to within floating-point rounding (not an approximation),
which is what the parity test in `tests/inference/test_numpy_logreg.py`
checks for with `atol=1e-8`.

This mirrors `Lecture2Revision/lstm_model.py`'s `NumPyLSTMModel` pattern:
train with a framework, export weights, reimplement the forward pass in
plain NumPy -- except here there is no approximation involved at all,
since logistic regression inference *is* just a dot product and a sigmoid.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


class NumPyLogisticRegression:
    """Plain-NumPy forward pass for a fitted sklearn LogisticRegression +
    StandardScaler pair. Retains no sklearn object -- only plain NumPy
    arrays (and a float threshold / list of feature names).
    """

    def __init__(
        self,
        coef: np.ndarray,
        intercept: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        threshold: float,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.coef = np.asarray(coef, dtype=float).reshape(-1)  # shape (n_features,)
        self.intercept = float(np.asarray(intercept, dtype=float).reshape(-1)[0])
        self.mean = np.asarray(mean, dtype=float).reshape(-1)
        self.scale = np.asarray(scale, dtype=float).reshape(-1)
        self.threshold = float(threshold)
        self.feature_names = list(feature_names) if feature_names is not None else None

        n = self.coef.shape[0]
        if self.mean.shape[0] != n or self.scale.shape[0] != n:
            raise ValueError(
                f"Shape mismatch: coef has {n} features but mean/scale have "
                f"{self.mean.shape[0]}/{self.scale.shape[0]}."
            )
        if self.feature_names is not None and len(self.feature_names) != n:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but coef has {n}."
            )

    # -- construction -----------------------------------------------------

    @classmethod
    def from_sklearn_model(
        cls,
        model,
        scaler,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> "NumPyLogisticRegression":
        """Extract `model.coef_`/`model.intercept_` and
        `scaler.mean_`/`scaler.scale_` into plain NumPy arrays. The
        returned instance holds no reference to `model` or `scaler`.
        """
        coef = np.array(model.coef_, dtype=float).reshape(-1).copy()
        intercept = np.array(model.intercept_, dtype=float).reshape(-1).copy()
        mean = np.array(scaler.mean_, dtype=float).reshape(-1).copy()
        scale = np.array(scaler.scale_, dtype=float).reshape(-1).copy()
        return cls(
            coef=coef,
            intercept=intercept,
            mean=mean,
            scale=scale,
            threshold=threshold,
            feature_names=feature_names,
        )

    # -- inference ----------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(at_risk=True) for each row of `X`, shape (n_samples,).

        `X` must be the RAW (unstandardized) feature matrix, shape
        (n_samples, n_features), in the same feature order used to build
        `coef`/`mean`/`scale`.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = (X - self.mean) / self.scale
        z = X_scaled @ self.coef + self.intercept
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Boolean at-risk predictions at `threshold` (defaults to the
        stored decision threshold, 0.90 for the production model).
        """
        t = self.threshold if threshold is None else float(threshold)
        return self.predict_proba(X) >= t

    # -- persistence ----------------------------------------------------

    def save(self, path) -> None:
        """Persists via `np.savez` -- plain arrays only, no pickled sklearn
        objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        feature_names = np.array(self.feature_names, dtype="U64") if self.feature_names else np.array([], dtype="U64")
        np.savez(
            path,
            coef=self.coef,
            intercept=np.array([self.intercept]),
            mean=self.mean,
            scale=self.scale,
            threshold=np.array([self.threshold]),
            feature_names=feature_names,
        )

    @classmethod
    def load(cls, path) -> "NumPyLogisticRegression":
        path = Path(path)
        # Guard against the caller omitting the .npz suffix: np.load would
        # otherwise silently resolve to a different file via the "allow
        # pickled fallback" path on some numpy versions.
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        with np.load(path, allow_pickle=False) as data:
            feature_names = list(data["feature_names"]) if data["feature_names"].size else None
            return cls(
                coef=data["coef"],
                intercept=data["intercept"],
                mean=data["mean"],
                scale=data["scale"],
                threshold=float(data["threshold"][0]),
                feature_names=feature_names,
            )
