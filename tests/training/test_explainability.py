"""
Tests for scripts/training/explainability.py (spec section 5, point 9).

These lean on the real data/training_rows.json and the real winning
Logistic Regression config throughout (rather than synthetic fixtures),
since explainability is specifically about real model behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.training import explainability

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "training_rows.json"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="data/training_rows.json not generated yet"
)


@pytest.fixture(scope="module")
def rows():
    return json.loads(DATA_PATH.read_text())


@pytest.fixture(scope="module")
def production_model(rows):
    return explainability.train_production_model(rows)


@pytest.fixture(scope="module")
def at_risk_row(rows):
    at_risk_rows = [r for r in rows if r["at_risk"]]
    assert at_risk_rows, "expected at least one at_risk=True row in real data"
    # Prefer a later checkpoint (more signal accumulated) for a clean example.
    return max(at_risk_rows, key=lambda r: r["checkpoint_idx"])


@pytest.fixture(scope="module")
def not_at_risk_row(rows):
    not_at_risk_rows = [r for r in rows if not r["at_risk"]]
    assert not_at_risk_rows, "expected at least one at_risk=False row in real data"
    return not_at_risk_rows[0]


def test_train_production_model_runs_on_real_data_and_returns_fitted_objects(production_model):
    model = production_model["model"]
    scaler = production_model["scaler"]

    assert hasattr(model, "coef_")
    assert model.coef_.shape == (1, len(explainability.FEATURE_NAMES))
    assert hasattr(scaler, "mean_")
    assert scaler.mean_.shape == (len(explainability.FEATURE_NAMES),)
    assert production_model["threshold"] == pytest.approx(0.90)
    assert production_model["background_rows"]
    assert production_model["feature_names"] == explainability.FEATURE_NAMES


def _assert_well_shaped_explanation(explanation, threshold):
    required_keys = {
        "predicted_risk_probability",
        "is_flagged",
        "top_risk_factors",
        "protective_factors",
        "human_readable_summary",
    }
    assert required_keys <= set(explanation.keys())

    proba = explanation["predicted_risk_probability"]
    assert isinstance(proba, float)
    assert 0.0 <= proba <= 1.0
    assert explanation["is_flagged"] == (proba >= threshold)

    for key in ("top_risk_factors", "protective_factors"):
        factors = explanation[key]
        assert isinstance(factors, list)
        assert len(factors) <= 3
        for factor in factors:
            assert set(factor.keys()) == {"feature", "shap_value", "feature_value"}
            assert factor["feature"] in explainability.FEATURE_NAMES
            assert isinstance(factor["shap_value"], float)
            assert isinstance(factor["feature_value"], float)

    for factor in explanation["top_risk_factors"]:
        assert factor["shap_value"] > 0
    for factor in explanation["protective_factors"]:
        assert factor["shap_value"] < 0

    assert isinstance(explanation["human_readable_summary"], str)
    assert len(explanation["human_readable_summary"]) > 0


def test_generate_student_explanation_on_at_risk_row(production_model, at_risk_row):
    explanation = explainability.generate_student_explanation(
        at_risk_row,
        production_model["model"],
        production_model["scaler"],
        threshold=production_model["threshold"],
        background_rows=production_model["background_rows"],
    )
    _assert_well_shaped_explanation(explanation, production_model["threshold"])


def test_generate_student_explanation_on_not_at_risk_row(production_model, not_at_risk_row):
    explanation = explainability.generate_student_explanation(
        not_at_risk_row,
        production_model["model"],
        production_model["scaler"],
        threshold=production_model["threshold"],
        background_rows=production_model["background_rows"],
    )
    _assert_well_shaped_explanation(explanation, production_model["threshold"])


def test_generate_student_explanation_without_background_rows_falls_back_cleanly(
    production_model, at_risk_row
):
    """No background_rows supplied -> falls back to the zero-vector scaled
    background (see module docstring), but must still produce a well-formed,
    internally-consistent explanation."""
    explanation = explainability.generate_student_explanation(
        at_risk_row,
        production_model["model"],
        production_model["scaler"],
        threshold=production_model["threshold"],
    )
    _assert_well_shaped_explanation(explanation, production_model["threshold"])


def test_shap_additivity_holds_for_real_rows(production_model, rows):
    """SHAP's core guarantee: expected_value + sum(shap_values) ==
    model's actual log-odds output for the row, within numeric tolerance."""
    model = production_model["model"]
    scaler = production_model["scaler"]
    background_rows = production_model["background_rows"]

    sample_rows = rows[:5] + [r for r in rows if r["at_risk"]][:5]

    explainer = explainability._build_explainer(model, scaler, background_rows)

    for row in sample_rows:
        x = explainability._features_matrix([row], explainability.FEATURE_NAMES)
        x_scaled = scaler.transform(x)

        shap_values = np.asarray(explainer.shap_values(x_scaled))[0]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.asarray(expected_value).reshape(-1)[0])

        reconstructed_log_odds = expected_value + float(np.sum(shap_values))
        actual_log_odds = float(model.decision_function(x_scaled)[0])

        assert reconstructed_log_odds == pytest.approx(actual_log_odds, abs=1e-6)


def test_at_risk_student_explanation_implicates_sensible_features(production_model, at_risk_row):
    """A real, known-declining at-risk row's top risk factors should
    plausibly include the deficit-margin / attendance / marks features that
    directly encode "below the risk threshold" or "declining" - not be
    nonsensical/random."""
    explanation = explainability.generate_student_explanation(
        at_risk_row,
        production_model["model"],
        production_model["scaler"],
        threshold=production_model["threshold"],
        background_rows=production_model["background_rows"],
    )

    plausible_features = {
        "attendance_deficit_margin",
        "marks_deficit_margin",
        "attendance_pct",
        "internal_marks_pct",
        "assignment_marks_pct",
        "attendance_velocity",
        "marks_velocity",
    }
    top_feature_names = {f["feature"] for f in explanation["top_risk_factors"]}
    assert top_feature_names, "expected at least one positive-SHAP risk factor for a real at_risk row"
    assert top_feature_names & plausible_features, (
        f"top risk factors {top_feature_names} did not include any plausible "
        f"at-risk-indicating feature from {plausible_features}"
    )

    # The row really is at-risk and high-signal (late checkpoint); the model
    # should at least lean toward flagging it, even if the fixed 0.90
    # threshold doesn't always flip for every single at-risk row.
    assert explanation["predicted_risk_probability"] > 0.5
