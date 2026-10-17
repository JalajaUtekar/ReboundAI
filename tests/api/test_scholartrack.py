"""Tests for api/_lib/scholartrack.py against the REAL live Supabase
database and the REAL data/training_rows.json dataset.

The most important test here is `test_shap_formula_matches_real_shap_...`:
it verifies, empirically, the claim `predict_risk` relies on to avoid
importing `shap` at runtime -- that `shap_i = coef_i * x_scaled_i` (a zero
background in scaled space) is EXACTLY what `shap.LinearExplainer` itself
produces for its own documented zero-background fallback (see
`scripts/training/explainability.py`'s module docstring and
`_build_explainer`). It is deliberately NOT compared against
`supabase/seed.py`'s seeded `risk_predictions` rows, which were computed
with a *real, non-zero-mean* background sample
(`train_production_model`'s `background_rows`) -- that path is a different,
only-approximately-equal comparison; see the module docstring in
`api/_lib/scholartrack.py` for why this endpoint deliberately uses the
zero-background formula instead of trying to reproduce that sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from api._lib import scholartrack
from api._lib.db import get_db_connection
from scripts.feature_engineering.extraction import FEATURE_NAMES
from scripts.training.explainability import train_production_model

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROWS_PATH = REPO_ROOT / "data" / "training_rows.json"


def _fetch_real_student_subject_ids():
    """Looks up real student_subject_ids live: one flagged (at-risk) and
    one with no risk_predictions row yet, so tests don't hardcode ids that
    could drift if the dataset is reseeded."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "select ss.id from public.student_subjects ss "
            "join public.risk_predictions rp on rp.student_subject_id = ss.id "
            "where rp.is_flagged limit 1"
        )
        flagged_row = cur.fetchone()

        cur.execute(
            "select ss.id from public.student_subjects ss "
            "where not exists ("
            "  select 1 from public.risk_predictions rp where rp.student_subject_id = ss.id"
            ") limit 1"
        )
        unflagged_row = cur.fetchone()
        cur.close()
    finally:
        conn.close()

    if flagged_row is None or unflagged_row is None:
        pytest.skip("expected at least one flagged and one unpredicted student_subject in the live DB")
    return flagged_row[0], unflagged_row[0]


@pytest.fixture(scope="module")
def real_student_subject_ids():
    return _fetch_real_student_subject_ids()


class TestComputeFeaturesAsOfLiveDb:
    def test_flagged_student_subject_features_are_sane(self, real_student_subject_ids):
        flagged_id, _ = real_student_subject_ids
        conn = get_db_connection()
        try:
            features = scholartrack.compute_features_as_of(flagged_id, conn)
        finally:
            conn.close()

        assert set(features.keys()) == set(FEATURE_NAMES)

        assert 0.0 <= features["internal_marks_pct"] <= 100.0
        assert 0.0 <= features["assignment_marks_pct"] <= 100.0
        assert 0.0 <= features["attendance_pct"] <= 100.0
        assert 0.0 <= features["on_time_submission_ratio"] <= 1.0
        assert features["avg_days_late"] >= 0.0
        # Deficit margins are signed (pct - threshold), so no fixed bound,
        # but they must be finite real numbers.
        for name in FEATURE_NAMES:
            assert np.isfinite(features[name])

    def test_unflagged_student_subject_features_are_sane(self, real_student_subject_ids):
        _, unflagged_id = real_student_subject_ids
        conn = get_db_connection()
        try:
            features = scholartrack.compute_features_as_of(unflagged_id, conn)
        finally:
            conn.close()

        assert set(features.keys()) == set(FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert np.isfinite(features[name])

    def test_unknown_student_subject_id_raises(self):
        conn = get_db_connection()
        try:
            with pytest.raises(ValueError):
                scholartrack.compute_features_as_of("00000000-0000-0000-0000-000000000000", conn)
        finally:
            conn.close()


class TestShapEquivalence:
    def test_shap_formula_matches_real_shap_zero_background(self):
        """THE critical test: coef_i * x_scaled_i, computed by hand, must
        match shap.LinearExplainer's own zero-background output exactly
        (not approximately) on real training rows."""
        import shap

        rows = json.loads(TRAINING_ROWS_PATH.read_text(encoding="utf-8"))
        production = train_production_model(rows)
        model = production["model"]
        scaler = production["scaler"]

        coef = model.coef_.reshape(-1)
        mean = scaler.mean_
        scale = scaler.scale_

        sample_rows = (
            [r for r in rows if r["at_risk"]][:6] + [r for r in rows if not r["at_risk"]][:6]
        )
        assert sample_rows, "expected at least some at_risk and not-at_risk rows to sample"

        explainer = shap.LinearExplainer(model, np.zeros((1, len(FEATURE_NAMES))))

        max_abs_diff = 0.0
        for row in sample_rows:
            x = np.array([float(row[name]) for name in FEATURE_NAMES], dtype=float)
            x_scaled = (x - mean) / scale
            my_shap = coef * x_scaled

            real_shap = np.asarray(explainer.shap_values(x_scaled.reshape(1, -1)))[0]
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(real_shap - my_shap))))

        print(f"\n[shap-equivalence] max abs diff vs real shap.LinearExplainer: {max_abs_diff:.3e}")
        assert max_abs_diff < 1e-9

    def test_shap_formula_does_not_match_the_seeded_background_sample(self):
        """Documents the OTHER comparison (seed.py's real background_rows
        sample) deliberately does NOT match the zero-background formula --
        so nobody mistakes the exact-match claim above for applying to the
        already-seeded risk_predictions rows too."""
        import shap

        rows = json.loads(TRAINING_ROWS_PATH.read_text(encoding="utf-8"))
        production = train_production_model(rows)
        model = production["model"]
        scaler = production["scaler"]
        background_rows = production["background_rows"]

        coef = model.coef_.reshape(-1)
        mean = scaler.mean_
        scale = scaler.scale_

        row = next(r for r in rows if r["at_risk"])
        x = np.array([float(row[name]) for name in FEATURE_NAMES], dtype=float)
        x_scaled = (x - mean) / scale
        my_shap = coef * x_scaled

        X_background = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in background_rows], dtype=float)
        X_background_scaled = scaler.transform(X_background)
        explainer = shap.LinearExplainer(model, X_background_scaled)
        real_shap = np.asarray(explainer.shap_values(x_scaled.reshape(1, -1)))[0]

        # Expected to differ (by a per-feature constant offset, since the
        # background sample's mean in scaled space isn't exactly zero) --
        # this is NOT a bug, it's why predict_risk avoids relying on it.
        assert np.max(np.abs(real_shap - my_shap)) > 1e-3


class TestPredictRiskRealModel:
    def test_predict_risk_on_a_real_flagged_student(self, real_student_subject_ids):
        """End-to-end with the ACTUAL deployed npz model and real live
        checkpoint data for a real, currently-flagged student-subject."""
        flagged_id, _ = real_student_subject_ids
        conn = get_db_connection()
        try:
            features = scholartrack.compute_features_as_of(flagged_id, conn)
        finally:
            conn.close()

        result = scholartrack.predict_risk(features)

        assert 0.0 <= result["risk_probability"] <= 1.0
        assert isinstance(result["is_flagged"], bool)
        assert isinstance(result["human_readable_summary"], str)
        assert result["human_readable_summary"]
        assert len(result["top_risk_factors"]) <= 3
        assert len(result["protective_factors"]) <= 3
        for factor in result["top_risk_factors"]:
            assert factor["shap_value"] > 0
        for factor in result["protective_factors"]:
            assert factor["shap_value"] < 0

        is_failing = scholartrack.compute_is_failing_so_far(features)
        assert isinstance(is_failing, bool)

        print(f"\n[real-prediction] student_subject_id={flagged_id} "
              f"risk_probability={result['risk_probability']:.4f} "
              f"is_flagged={result['is_flagged']} is_failing={is_failing}")
        print(f"[real-prediction] summary: {result['human_readable_summary']}")
