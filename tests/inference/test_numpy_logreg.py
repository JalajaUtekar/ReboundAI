"""Tests for scripts/inference/numpy_logreg.py -- the PRIORITY inference
path, since Logistic Regression is the model that actually ships (per
models/benchmark_report.md and scripts/training/explainability.py::
train_production_model).

The most important test here is `test_parity_with_real_production_model`:
it trains the REAL production model on the REAL data/training_rows.json
and asserts the hand-rolled NumPy forward pass matches sklearn's
`predict_proba` to atol=1e-8 -- this should be essentially exact, since
it's the same math (standardize, dot product, sigmoid), just hand-rolled
instead of called through sklearn.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from scripts.inference.numpy_logreg import NumPyLogisticRegression
from scripts.training.explainability import FEATURE_NAMES, train_production_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "training_rows.json"
EXPORTED_WEIGHTS_PATH = REPO_ROOT / "scripts" / "inference" / "numpy_logreg_exported.npz"


def _features_matrix(rows, feature_names):
    return np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)


@pytest.fixture(scope="module")
def production_fit():
    """Fits the real production model once (per module) on the real data."""
    if not DATA_PATH.exists():
        pytest.skip(f"{DATA_PATH} not present")
    rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    result = train_production_model(rows)
    X = _features_matrix(rows, FEATURE_NAMES)
    return result, X


class TestParity:
    def test_parity_with_real_production_model(self, production_fit):
        """The critical test: numpy_model.predict_proba(X) must match
        sklearn_model.predict_proba(X_scaled)[:, 1] almost exactly."""
        result, X = production_fit
        sk_model = result["model"]
        scaler = result["scaler"]
        threshold = result["threshold"]

        numpy_model = NumPyLogisticRegression.from_sklearn_model(
            sk_model, scaler, feature_names=FEATURE_NAMES, threshold=threshold
        )

        numpy_proba = numpy_model.predict_proba(X)
        X_scaled = scaler.transform(X)
        sklearn_proba = sk_model.predict_proba(X_scaled)[:, 1]

        max_abs_diff = float(np.max(np.abs(numpy_proba - sklearn_proba)))
        print(f"\n[parity] max abs diff (numpy vs sklearn predict_proba): {max_abs_diff:.3e}")

        assert np.allclose(numpy_proba, sklearn_proba, atol=1e-8)

    def test_parity_predictions_match_at_threshold(self, production_fit):
        """Boolean predict() at the production threshold must also agree."""
        result, X = production_fit
        sk_model = result["model"]
        scaler = result["scaler"]
        threshold = result["threshold"]

        numpy_model = NumPyLogisticRegression.from_sklearn_model(
            sk_model, scaler, feature_names=FEATURE_NAMES, threshold=threshold
        )

        numpy_preds = numpy_model.predict(X)
        X_scaled = scaler.transform(X)
        sklearn_preds = sk_model.predict_proba(X_scaled)[:, 1] >= threshold

        assert np.array_equal(numpy_preds, sklearn_preds)

    def test_numpy_model_retains_no_sklearn_object(self, production_fit):
        result, X = production_fit
        numpy_model = NumPyLogisticRegression.from_sklearn_model(
            result["model"], result["scaler"], feature_names=FEATURE_NAMES, threshold=result["threshold"]
        )
        # Only plain numpy arrays / python scalars should be stored.
        assert isinstance(numpy_model.coef, np.ndarray)
        assert isinstance(numpy_model.mean, np.ndarray)
        assert isinstance(numpy_model.scale, np.ndarray)
        assert isinstance(numpy_model.intercept, float)
        for attr in vars(numpy_model).values():
            assert not hasattr(attr, "predict_proba")  # no sklearn estimator snuck in


class TestSaveLoadRoundTrip:
    def test_save_and_load_round_trip(self, production_fit, tmp_path):
        result, X = production_fit
        numpy_model = NumPyLogisticRegression.from_sklearn_model(
            result["model"], result["scaler"], feature_names=FEATURE_NAMES, threshold=result["threshold"]
        )
        path = tmp_path / "model.npz"
        numpy_model.save(path)
        assert path.exists()

        loaded = NumPyLogisticRegression.load(path)
        assert np.allclose(loaded.coef, numpy_model.coef)
        assert np.allclose(loaded.mean, numpy_model.mean)
        assert np.allclose(loaded.scale, numpy_model.scale)
        assert loaded.intercept == pytest.approx(numpy_model.intercept)
        assert loaded.threshold == pytest.approx(numpy_model.threshold)
        assert loaded.feature_names == numpy_model.feature_names

        # Loaded model must predict identically to the original.
        assert np.allclose(loaded.predict_proba(X), numpy_model.predict_proba(X), atol=1e-12)

    def test_load_without_npz_suffix(self, production_fit, tmp_path):
        result, X = production_fit
        numpy_model = NumPyLogisticRegression.from_sklearn_model(
            result["model"], result["scaler"], feature_names=FEATURE_NAMES, threshold=result["threshold"]
        )
        path = tmp_path / "model"  # no .npz suffix
        numpy_model.save(path)
        loaded = NumPyLogisticRegression.load(path)  # also no suffix
        assert np.allclose(loaded.coef, numpy_model.coef)

    def test_exported_weights_file_loads_and_is_sane(self):
        """The committed scripts/inference/numpy_logreg_exported.npz artifact
        should load and produce probabilities in [0, 1]."""
        if not EXPORTED_WEIGHTS_PATH.exists():
            pytest.skip(f"{EXPORTED_WEIGHTS_PATH} not present")
        model = NumPyLogisticRegression.load(EXPORTED_WEIGHTS_PATH)
        assert model.coef.shape == (10,)
        assert model.feature_names == FEATURE_NAMES
        dummy = np.tile(model.mean, (5, 1))  # rows at the training mean
        proba = model.predict_proba(dummy)
        assert proba.shape == (5,)
        assert np.all((proba >= 0.0) & (proba <= 1.0))


class TestSanity:
    def test_shape_and_probability_range(self, production_fit):
        result, X = production_fit
        numpy_model = NumPyLogisticRegression.from_sklearn_model(
            result["model"], result["scaler"], feature_names=FEATURE_NAMES, threshold=result["threshold"]
        )
        proba = numpy_model.predict_proba(X)
        assert proba.shape == (X.shape[0],)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

        preds = numpy_model.predict(X)
        assert preds.dtype == bool
        assert preds.shape == (X.shape[0],)

    def test_single_row_input(self, production_fit):
        result, X = production_fit
        numpy_model = NumPyLogisticRegression.from_sklearn_model(
            result["model"], result["scaler"], feature_names=FEATURE_NAMES, threshold=result["threshold"]
        )
        single = X[0]
        proba = numpy_model.predict_proba(single)
        assert proba.shape == (1,)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            NumPyLogisticRegression(
                coef=np.zeros(3),
                intercept=np.array([0.0]),
                mean=np.zeros(4),  # mismatched
                scale=np.ones(3),
                threshold=0.5,
            )


class TestLatency:
    """Nice-to-have latency sanity check (not a hard requirement) per the
    original stepstbd.md draft's '<10ms' serving aspiration for the
    LogReg path -- the one that actually matters for real serving."""

    def test_latency_sanity_check(self, production_fit):
        result, X = production_fit
        numpy_model = NumPyLogisticRegression.from_sklearn_model(
            result["model"], result["scaler"], feature_names=FEATURE_NAMES, threshold=result["threshold"]
        )
        row = X[0:1]  # single row, shape (1, 10)

        n_calls = 1000
        start = time.perf_counter()
        for _ in range(n_calls):
            numpy_model.predict_proba(row)
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / n_calls) * 1000.0
        print(f"\n[latency] {n_calls} calls to predict_proba on 1 row: "
              f"{per_call_ms:.4f} ms/call average ({elapsed:.4f}s total)")

        # Not asserted as a hard gate (aspiration, not a requirement) --
        # just reported. Sanity bound well above <10ms so this doesn't
        # become flaky on a slow CI box.
        assert per_call_ms < 50.0
