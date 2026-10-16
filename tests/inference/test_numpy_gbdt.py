"""Tests for scripts/inference/numpy_gbdt.py.

NOT the served model (Logistic Regression is -- see
models/benchmark_report.md); this is the documented, lower-priority
Gradient Boosting NumPy serving path.

What IS covered: the FULL ensemble traversal of a real, fitted
`sklearn.ensemble.HistGradientBoostingClassifier` (every boosting
iteration's tree, via the model's private `_predictors` attribute, not a
simplified single-tree stand-in). This was chosen over the "simpler
single-tree demonstration" fallback the task allowed, because a scratch
exploration showed `TreePredictor.nodes["num_threshold"]` stores the real
(unbinned) feature threshold, making a direct, exact-parity traversal
straightforward to implement and verify.

What is NOT covered: no attempt is made to handle missing values beyond
reading `missing_go_to_left` for completeness in the node layout (this
dataset and these tests have no missing/NaN features, so that code path
is untested), and no attempt is made to handle categorical splits
(`is_categorical`/`binned_left_cat_bitsets`) since
`HistGradientBoostingClassifier` here is trained on purely numeric
features and never produces one. This implementation is also pinned in
spirit to the installed scikit-learn version's private `TreePredictor`
layout -- a future sklearn upgrade that changes that layout would need
re-verification (see the module docstring).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:  # pragma: no cover
    HistGradientBoostingClassifier = None

from scripts.inference.numpy_gbdt import NumPyGBDTModel
from scripts.training.baselines import DEFAULT_FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "training_rows.json"


def _features_matrix(rows, feature_names):
    return np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)


def _labels_array(rows):
    return np.array([bool(row["at_risk"]) for row in rows], dtype=bool)


@pytest.fixture(scope="module")
def fitted_gbdt():
    if not DATA_PATH.exists():
        pytest.skip(f"{DATA_PATH} not present")
    rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    X = _features_matrix(rows, DEFAULT_FEATURE_NAMES)
    y = _labels_array(rows)

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_depth=3,
        min_samples_leaf=5,
        learning_rate=0.1,
        max_iter=50,
        random_state=42,
    )
    model.fit(X, y)
    return model, X


class TestFullEnsembleParity:
    def test_parity_on_real_data(self, fitted_gbdt):
        model, X = fitted_gbdt
        numpy_model = NumPyGBDTModel.from_sklearn_model(model, feature_names=DEFAULT_FEATURE_NAMES, threshold=0.5)

        numpy_proba = numpy_model.predict_proba(X)
        sklearn_proba = model.predict_proba(X)[:, 1]

        max_abs_diff = float(np.max(np.abs(numpy_proba - sklearn_proba)))
        print(f"\n[gbdt parity] max abs diff: {max_abs_diff:.3e}")

        assert np.allclose(numpy_proba, sklearn_proba, atol=1e-10)

    def test_predictions_match_at_threshold(self, fitted_gbdt):
        model, X = fitted_gbdt
        threshold = 0.5
        numpy_model = NumPyGBDTModel.from_sklearn_model(model, feature_names=DEFAULT_FEATURE_NAMES, threshold=threshold)
        numpy_preds = numpy_model.predict(X)
        sklearn_preds = model.predict_proba(X)[:, 1] >= threshold
        assert np.array_equal(numpy_preds, sklearn_preds)

    def test_tree_count_matches_boosting_iterations(self, fitted_gbdt):
        model, X = fitted_gbdt
        numpy_model = NumPyGBDTModel.from_sklearn_model(model, feature_names=DEFAULT_FEATURE_NAMES)
        expected_trees = sum(len(iter_predictors) for iter_predictors in model._predictors)
        assert len(numpy_model.trees) == expected_trees


class TestSaveLoadRoundTrip:
    def test_round_trip(self, fitted_gbdt, tmp_path):
        model, X = fitted_gbdt
        numpy_model = NumPyGBDTModel.from_sklearn_model(model, feature_names=DEFAULT_FEATURE_NAMES, threshold=0.5)
        path = tmp_path / "gbdt.npz"
        numpy_model.save(path)
        loaded = NumPyGBDTModel.load(path)

        assert len(loaded.trees) == len(numpy_model.trees)
        assert loaded.baseline == pytest.approx(numpy_model.baseline)
        assert loaded.feature_names == numpy_model.feature_names
        assert np.allclose(loaded.predict_proba(X), numpy_model.predict_proba(X), atol=1e-12)


class TestSanity:
    def test_shape_and_probability_range(self, fitted_gbdt):
        model, X = fitted_gbdt
        numpy_model = NumPyGBDTModel.from_sklearn_model(model, feature_names=DEFAULT_FEATURE_NAMES, threshold=0.5)
        proba = numpy_model.predict_proba(X)
        assert proba.shape == (X.shape[0],)
        assert np.all((proba >= 0.0) & (proba <= 1.0))
