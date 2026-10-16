"""Pure-NumPy/plain-Python inference path for ScholarTrack's Gradient
Boosting baseline (Step 5a) -- documented, NOT the production model (see
`models/benchmark_report.md`: Logistic Regression wins the held-out test
and is what actually ships; `numpy_logreg.py` is the served path).

sklearn's `HistGradientBoostingClassifier` exposes its learned trees via
the private `model._predictors` attribute: a list (one per boosting
iteration) of lists (one per "tree per iteration" -- 1 for binary
classification) of `sklearn.ensemble._hist_gradient_boosting.predictor.
TreePredictor` objects. Each `TreePredictor.nodes` is a structured NumPy
array with fields including `feature_idx`, `num_threshold`, `left`,
`right`, `is_leaf`, and `value` -- a flat binary-tree encoding we can
traverse directly, no re-implementation of sklearn's histogram-binning
training logic needed, only its (much simpler) *inference*-time split
rule: `go left if x[feature_idx] <= num_threshold else right`, accumulate
each tree's leaf `value`, add `model._baseline_prediction`, and take
`sigmoid` of the sum to get P(at_risk=True).

This was verified directly against a real fitted
`HistGradientBoostingClassifier` in a scratch exploration (see this
module's test file) to be an EXACT reproduction (max abs diff 0.0 in that
check) -- `TreePredictor.nodes["num_threshold"]` stores the real
(unbinned) feature threshold, so no bin-edge lookup table is needed at
inference time. This is the full ensemble traversal, not a simplified
single-tree stand-in; `.nodes` is a private/internal sklearn attribute
though, so this implementation is version-pinned in spirit -- it was
validated against the scikit-learn version this repo pins, and a future
sklearn upgrade that changes the `TreePredictor.nodes` field layout would
require re-verifying this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


class NumPyGBDTModel:
    """Plain-NumPy traversal of a fitted sklearn `HistGradientBoostingClassifier`.

    Retains no sklearn object -- only plain NumPy arrays describing every
    tree's nodes, plus the ensemble baseline/threshold/feature names.
    """

    # Field order matches TreePredictor.nodes' structured dtype; stored as a
    # plain float64 array of shape (n_nodes, 6):
    # [feature_idx, num_threshold, left, right, is_leaf, value]
    _FIELDS = ("feature_idx", "num_threshold", "left", "right", "is_leaf", "value")

    def __init__(
        self,
        trees: List[np.ndarray],
        baseline: float,
        threshold: float,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.trees = [np.asarray(t, dtype=float) for t in trees]
        self.baseline = float(baseline)
        self.threshold = float(threshold)
        self.feature_names = list(feature_names) if feature_names is not None else None

    # -- construction -----------------------------------------------------

    @classmethod
    def from_sklearn_model(
        cls,
        model,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> "NumPyGBDTModel":
        """Extract every boosting iteration's tree(s) from
        `model._predictors` into plain NumPy arrays."""
        trees: List[np.ndarray] = []
        for iter_predictors in model._predictors:
            for tree_predictor in iter_predictors:
                nodes = tree_predictor.nodes
                arr = np.stack(
                    [
                        nodes["feature_idx"].astype(float),
                        nodes["num_threshold"].astype(float),
                        nodes["left"].astype(float),
                        nodes["right"].astype(float),
                        nodes["is_leaf"].astype(float),
                        nodes["value"].astype(float),
                    ],
                    axis=1,
                )
                trees.append(arr)

        baseline = float(np.asarray(model._baseline_prediction).reshape(-1)[0])
        return cls(trees=trees, baseline=baseline, threshold=threshold, feature_names=feature_names)

    # -- inference ----------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _traverse_one_tree(self, tree: np.ndarray, x: np.ndarray) -> float:
        i = 0
        while True:
            feature_idx, num_threshold, left, right, is_leaf, value = tree[i]
            if is_leaf:
                return value
            i = int(left) if x[int(feature_idx)] <= num_threshold else int(right)

    def _raw_score_one_row(self, x: np.ndarray) -> float:
        s = self.baseline
        for tree in self.trees:
            s += self._traverse_one_tree(tree, x)
        return s

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(at_risk=True) for each row of `X`, shape (n_samples,).

        Plain-Python row-by-row tree traversal (not vectorized across
        rows) -- this mirrors sklearn's own per-sample tree-descent
        structure and keeps the implementation a direct, readable
        translation of `TreePredictor.nodes`, trading vectorized speed
        (not needed here -- GBDT is a documented, non-served path) for
        clarity.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = np.array([self._raw_score_one_row(row) for row in X])
        return self._sigmoid(raw)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        t = self.threshold if threshold is None else float(threshold)
        return self.predict_proba(X) >= t

    # -- persistence ----------------------------------------------------

    def save(self, path) -> None:
        """Persists via `np.savez` -- one array per tree, plus baseline/
        threshold/feature_names/tree count. No pickled sklearn objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        feature_names = np.array(self.feature_names, dtype="U64") if self.feature_names else np.array([], dtype="U64")
        arrays = {f"tree_{i}": t for i, t in enumerate(self.trees)}
        np.savez(
            path,
            n_trees=np.array([len(self.trees)]),
            baseline=np.array([self.baseline]),
            threshold=np.array([self.threshold]),
            feature_names=feature_names,
            **arrays,
        )

    @classmethod
    def load(cls, path) -> "NumPyGBDTModel":
        path = Path(path)
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        with np.load(path, allow_pickle=False) as data:
            n_trees = int(data["n_trees"][0])
            trees = [data[f"tree_{i}"] for i in range(n_trees)]
            feature_names = list(data["feature_names"]) if data["feature_names"].size else None
            return cls(
                trees=trees,
                baseline=float(data["baseline"][0]),
                threshold=float(data["threshold"][0]),
                feature_names=feature_names,
            )
