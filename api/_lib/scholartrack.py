"""Rebound AI -- ScholarTrack risk-prediction logic (spec section 5).

Two halves:

1. `compute_features_as_of` -- queries `assessment_checkpoints` for one
   `student_subject_id` and recomputes the exact 10-feature vector that
   `scripts/feature_engineering/extraction.py::extract_prefix_features`
   computes, by reassembling the live-queried rows into the same
   `{"unit_tests": [...], "submissions": [...], "attendance_weeks": [...]}`
   shape that function expects and calling it directly -- so there is no
   second, hand-copied implementation of the feature formulas that could
   quietly drift out of sync with the one the model was trained against.

2. `predict_risk` -- loads the deployed `NumPyLogisticRegression` (plain
   NumPy, no sklearn/tensorflow at runtime -- see
   `scripts/inference/numpy_logreg.py`) and computes `risk_probability` /
   `is_flagged`, plus exact per-feature SHAP-equivalent contributions
   computed directly as `shap_i = coef_i * x_scaled_i` -- no `shap` package
   import needed in this serverless path.

   Why that formula is exact, not an approximation: `StandardScaler`
   guarantees the training mean maps to the origin in scaled space, so a
   SHAP explanation relative to an all-zero background row in scaled space
   is mathematically the same as `shap.LinearExplainer`'s own documented
   zero-background fallback (see
   `scripts/training/explainability.py`'s module docstring and
   `_build_explainer`). `tests/api/test_scholartrack.py` proves this
   empirically against the real `shap.LinearExplainer` output on real
   dataset rows (max abs diff 0.0 across every row tested) -- NOT against
   the background-sample path `supabase/seed.py` used to seed
   `risk_predictions` during the initial data load, which uses a
   non-zero-mean background sample and so is only approximately matched by
   this formula (see that test file for the measured difference in that
   case). This endpoint deliberately recomputes predictions fresh rather
   than reading the seeded rows, so the zero-background comparison is the
   relevant one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from scripts.data_generation.generator import CHECKPOINTS_IN_TIME_ORDER
from scripts.feature_engineering.extraction import (
    MAX_PREFIX_LEN,
    MIN_PREFIX_LEN,
    RISK_MARKS_THRESHOLD,
    extract_prefix_features,
)
from scripts.inference.numpy_logreg import NumPyLogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = REPO_ROOT / "scripts" / "inference" / "numpy_logreg_exported.npz"

# ScholarTrack's generated dataset always has exactly these many unit-test
# and submission checkpoints (spec section 4): 2 unit tests, 5 submissions
# (see scripts/data_generation/generator.py's UNIT_TEST_WEEKS/
# SUBMISSION_WEEKS). Derived from CHECKPOINTS_IN_TIME_ORDER rather than
# hardcoded so this stays correct if that generator config ever changes.
_NUM_UNIT_TESTS = sum(1 for kind, _ in CHECKPOINTS_IN_TIME_ORDER if kind == "unit_test")
_NUM_SUBMISSIONS = sum(1 for kind, _ in CHECKPOINTS_IN_TIME_ORDER if kind == "submission")

# "Failing" is the informational, more severe aggregate-marks fact (spec
# section 3/4; models/benchmark_report.md). The real SPPU-passing cutoff is
# 40% of the FULL aggregate INCLUDING the end-term exam (ETE) -- but ETE is
# never stored in assessment_checkpoints (by design: it must never leak
# into a feature, see that table's migration comment), so it is not
# available to any live request. The closest no-leakage approximation is
# the same "aggregate marks so far" quantity extract_prefix_features already
# computes for marks_deficit_margin (internal + assignment marks, no ETE,
# anchored at RISK_MARKS_THRESHOLD=70 there) -- here thresholded at 40
# instead. This is a documented approximation, not the real is_failing
# definition; see FAILING_THRESHOLD_SO_FAR's docstring note below and the
# api/predict_risk.py endpoint, which uses this rather than silently
# reusing a stale seeded value.
FAILING_THRESHOLD_SO_FAR = 40.0

# Narration floor copied from scripts/training/explainability.py -- a
# feature's |SHAP value| (log-odds units) must clear this before it's worth
# naming in the human-readable summary.
_NARRATION_MIN_ABS_SHAP = 0.1

_RISK_PHRASES: Dict[str, str] = {
    "internal_marks_pct": "internal exam marks are low",
    "assignment_marks_pct": "assignment marks are low",
    "attendance_pct": "attendance is low",
    "on_time_submission_ratio": "a low share of submissions have been on time",
    "avg_days_late": "submissions have been arriving late",
    "attendance_velocity": "attendance has been declining",
    "marks_velocity": "marks have been trending downward",
    "delay_acceleration": "submission delays have been getting worse",
    "attendance_deficit_margin": "attendance is well below the risk threshold",
    "marks_deficit_margin": "marks are well below the risk threshold",
}

_PROTECTIVE_PHRASES: Dict[str, str] = {
    "internal_marks_pct": "internal exam marks are strong",
    "assignment_marks_pct": "assignment marks are strong",
    "attendance_pct": "attendance is solid",
    "on_time_submission_ratio": "most submissions have been on time",
    "avg_days_late": "submissions have generally been on time",
    "attendance_velocity": "attendance has been improving",
    "marks_velocity": "marks have been trending upward",
    "delay_acceleration": "submission timeliness has been improving",
    "attendance_deficit_margin": "attendance is comfortably above the risk threshold",
    "marks_deficit_margin": "marks are comfortably above the risk threshold",
}


def compute_features_as_of(student_subject_id: str, db_conn) -> Dict[str, float]:
    """Queries `assessment_checkpoints` for `student_subject_id` and returns
    the 10-feature dict (`scripts/feature_engineering/extraction.FEATURE_NAMES`
    order, as a dict) as of however many checkpoints currently exist for it.

    `db_conn` is any DB-API 2.0 connection (e.g. from
    `api/_lib/db.py::get_db_connection`) with a live transaction/autocommit
    the caller controls -- this function only reads, via its own cursor.
    """
    cur = db_conn.cursor()
    try:
        cur.execute(
            "select checkpoint_type, checkpoint_index, week_number, score, "
            "max_marks, attendance_pct, due_day, submitted_day "
            "from public.assessment_checkpoints "
            "where student_subject_id = %s",
            (student_subject_id,),
        )
        rows = cur.fetchall()
    finally:
        cur.close()

    if not rows:
        raise ValueError(
            f"No assessment_checkpoints rows found for student_subject_id={student_subject_id!r}"
        )

    unit_tests: List[Optional[dict]] = [None] * _NUM_UNIT_TESTS
    submissions: List[Optional[dict]] = [None] * _NUM_SUBMISSIONS
    attendance_weeks: List[dict] = []

    for (
        checkpoint_type,
        checkpoint_index,
        week_number,
        score,
        max_marks,
        attendance_pct,
        due_day,
        submitted_day,
    ) in rows:
        if checkpoint_type == "unit_test":
            idx0 = checkpoint_index - 1
            if 0 <= idx0 < _NUM_UNIT_TESTS:
                unit_tests[idx0] = {
                    "score": float(score),
                    "max_marks": float(max_marks),
                    "week": int(week_number),
                }
        elif checkpoint_type == "submission":
            idx0 = checkpoint_index - 1
            if 0 <= idx0 < _NUM_SUBMISSIONS:
                submissions[idx0] = {
                    "score": float(score),
                    "max_marks": float(max_marks),
                    "due_day": int(due_day),
                    "submitted_day": int(submitted_day),
                }
        elif checkpoint_type == "attendance_week":
            attendance_weeks.append(
                {"week": int(week_number), "attendance_pct": float(attendance_pct)}
            )

    n_unit_tests_present = sum(1 for u in unit_tests if u is not None)
    n_submissions_present = sum(1 for s in submissions if s is not None)
    prefix_len = n_unit_tests_present + n_submissions_present
    prefix_len = max(MIN_PREFIX_LEN, min(prefix_len, MAX_PREFIX_LEN))

    subject = {
        "unit_tests": unit_tests,
        "submissions": submissions,
        "attendance_weeks": attendance_weeks,
    }
    return extract_prefix_features(subject, prefix_len)


def compute_is_failing_so_far(features: Dict[str, float]) -> bool:
    """The documented is_failing approximation described in this module's
    docstring: the same no-ETE "aggregate marks so far" quantity
    `extract_prefix_features` computes for `marks_deficit_margin`
    (anchored at `RISK_MARKS_THRESHOLD`), re-thresholded at
    `FAILING_THRESHOLD_SO_FAR` (40%) instead.
    """
    aggregate_pct_so_far = features["marks_deficit_margin"] + RISK_MARKS_THRESHOLD
    return aggregate_pct_so_far < FAILING_THRESHOLD_SO_FAR


def _narrate(factors: List[dict], phrase_map: Dict[str, str]) -> List[str]:
    phrases = []
    for factor in factors:
        if abs(factor["shap_value"]) >= _NARRATION_MIN_ABS_SHAP:
            phrases.append(phrase_map.get(factor["feature"], factor["feature"].replace("_", " ")))
    return phrases


def _join(phrases: List[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{phrases[0]}, {phrases[1]}, and {phrases[2]}"


def _build_summary(
    subject_label: str,
    risk_factors: List[dict],
    protective_factors: List[dict],
    is_flagged: bool,
    proba: float,
) -> str:
    if is_flagged:
        risk_phrases = _narrate(risk_factors, _RISK_PHRASES)
        if risk_phrases:
            body = _join(risk_phrases)
            body = body[0].upper() + body[1:]
            return (
                f"{body}, putting this student at elevated risk in {subject_label} "
                f"(predicted risk {proba:.0%})."
            )
        return (
            f"This student is flagged as at elevated risk in {subject_label} "
            f"(predicted risk {proba:.0%}), though no single factor dominates "
            "the prediction."
        )

    protective_phrases = _narrate(protective_factors, _PROTECTIVE_PHRASES)
    if protective_phrases:
        body = _join(protective_phrases)
        return (
            f"This student is not currently flagged as at risk in {subject_label}: "
            f"{body} (predicted risk {proba:.0%})."
        )
    return (
        f"This student is not currently flagged as at risk in {subject_label} "
        f"(predicted risk {proba:.0%})."
    )


def predict_risk(features: Dict[str, float]) -> Dict[str, object]:
    """Runs the deployed NumPy logistic-regression model on `features`
    (a dict carrying at least the 10 keys in
    `scripts.feature_engineering.extraction.FEATURE_NAMES`; an optional
    `"subject_slug"` key, if present, is used only for the human-readable
    summary's subject label and is otherwise ignored).

    Returns:
        {
            "risk_probability": float,
            "is_flagged": bool,
            "top_risk_factors": list[dict],       # top 3 by positive SHAP value
            "protective_factors": list[dict],      # top 3 by negative SHAP value
            "human_readable_summary": str,
        }
    """
    model = NumPyLogisticRegression.load(MODEL_PATH)

    x = np.array([float(features[name]) for name in model.feature_names], dtype=float)
    x_scaled = (x - model.mean) / model.scale
    proba = float(model.predict_proba(x)[0])
    is_flagged = bool(proba >= model.threshold)

    # Exact per-feature SHAP contributions relative to a zero (= training
    # mean) background in scaled space -- see module docstring.
    shap_values = model.coef * x_scaled

    feature_rows = [
        {
            "feature": name,
            "shap_value": float(shap_values[i]),
            "feature_value": float(x[i]),
        }
        for i, name in enumerate(model.feature_names)
    ]

    top_risk_factors = sorted(
        (f for f in feature_rows if f["shap_value"] > 0),
        key=lambda f: -f["shap_value"],
    )[:3]
    protective_factors = sorted(
        (f for f in feature_rows if f["shap_value"] < 0),
        key=lambda f: f["shap_value"],
    )[:3]

    subject_label = features.get("subject_slug") or "this subject"
    summary = _build_summary(subject_label, top_risk_factors, protective_factors, is_flagged, proba)

    return {
        "risk_probability": proba,
        "is_flagged": is_flagged,
        "top_risk_factors": top_risk_factors,
        "protective_factors": protective_factors,
        "human_readable_summary": summary,
    }
