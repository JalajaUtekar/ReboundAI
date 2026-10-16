"""
SHAP explainability for ScholarTrack's production risk model - spec section
5, point 9:

    "SHAP on top of the trained model, so every flagged student comes with
    *why* ('attendance trend -32%, 3 late TW submissions') instead of a bare
    probability."

Per `models/benchmark_report.md` (Step 3's final result), the model that
actually ships is Logistic Regression with `C=0.1`,
`oversample_target_ratio=0.5`, decision threshold `0.90` -
`scripts/training/baselines.py::train_and_predict_oof_logreg`'s winning
config. This module:

1. Fits that exact config as a single PRODUCTION model on all 120 students'
   rows (not a per-fold OOF/CV artifact - `train_production_model`), since
   this is the model that will actually be asked to explain future
   predictions, not a cross-validation experiment.
2. Explains individual predictions with `shap.LinearExplainer`
   (`generate_student_explanation`) - the exact, closed-form SHAP explainer
   for linear models, so no sampling approximation is needed the way it
   would be for `KernelExplainer`.

SHAP background dataset choice
-------------------------------
`shap.LinearExplainer` needs a background dataset to define the "expected
value" each prediction is explained relative to. `train_production_model`
stores a `background_rows` sample (a `random.Random(42).sample` of up to 100
of the training rows it fit on, or all of them if there are 100 or fewer -
the 120-student dataset here has only 120 rows after de-oversampling context,
so in practice this keeps every row). We use plain random sampling rather
than `shap.kmeans` because clustering a 10-feature, well-behaved tabular
dataset this small buys no meaningful speed-up (`LinearExplainer` is a closed
form, not a Monte-Carlo estimator) and would blur individual students'
features into cluster centroids for no benefit. If a caller doesn't have a
background sample (e.g. they didn't call `train_production_model`), SHAP
values are computed against a single all-zero background row in the
*scaled* feature space - because `StandardScaler` guarantees the training
mean maps to the origin in scaled space, this is exactly equivalent to
mean-centered attribution without needing the original rows. Callers should
prefer passing the real `background_rows` from `train_production_model`.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.training.baselines import DEFAULT_FEATURE_NAMES
from scripts.training.cv import oversample_training_rows

FEATURE_NAMES: List[str] = DEFAULT_FEATURE_NAMES

DEFAULT_C = 0.1
DEFAULT_OVERSAMPLE_TARGET_RATIO = 0.5
DEFAULT_THRESHOLD = 0.90
DEFAULT_BACKGROUND_SAMPLE_SIZE = 100

# Minimum |SHAP value| (log-odds units) for a feature to be worth narrating
# in the human-readable summary. Below this, a factor's contribution is
# close enough to zero that naming it would overstate its role.
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


def _features_matrix(rows: List[dict], feature_names: List[str]) -> np.ndarray:
    return np.array(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=float,
    )


def _labels_array(rows: List[dict]) -> np.ndarray:
    return np.array([bool(row["at_risk"]) for row in rows], dtype=bool)


def train_production_model(
    rows: List[dict],
    C: float = DEFAULT_C,
    oversample_target_ratio: float = DEFAULT_OVERSAMPLE_TARGET_RATIO,
    threshold: float = DEFAULT_THRESHOLD,
    background_sample_size: int = DEFAULT_BACKGROUND_SAMPLE_SIZE,
    seed: int = 42,
) -> Dict[str, object]:
    """Fit the single production Logistic Regression model on ALL of `rows`.

    Mirrors `baselines.train_and_predict_oof_logreg`'s winning-fold recipe
    exactly, but as one fit over the whole dataset rather than per-CV-fold:
    oversample (`oversample_training_rows`, `target_ratio=0.5` by default),
    fit a fresh `StandardScaler` on the oversampled rows, then fit
    `LogisticRegression(C=0.1, class_weight='balanced', penalty='l2',
    max_iter=1000)` on the scaled, oversampled rows.

    Returns a dict with:
        "model": the fitted LogisticRegression
        "scaler": the fitted StandardScaler
        "threshold": the decision threshold (0.90 by default)
        "background_rows": a sample of up to `background_sample_size` of the
            ORIGINAL (un-oversampled, unscaled) input `rows`, for use as
            `generate_student_explanation`'s `background_rows` argument.
        "feature_names": the 10 feature names, in the order the model
            expects them.
    """
    oversampled_rows = oversample_training_rows(rows, target_ratio=oversample_target_ratio)

    X_train = _features_matrix(oversampled_rows, FEATURE_NAMES)
    y_train = _labels_array(oversampled_rows)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(C=C, class_weight="balanced", penalty="l2", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    rng = random.Random(seed)
    background_rows = list(rows)
    if len(background_rows) > background_sample_size:
        background_rows = rng.sample(background_rows, background_sample_size)

    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "background_rows": background_rows,
        "feature_names": list(FEATURE_NAMES),
    }


def _build_explainer(
    model: LogisticRegression,
    scaler: StandardScaler,
    background_rows: Optional[List[dict]],
) -> shap.LinearExplainer:
    if background_rows:
        X_background = _features_matrix(background_rows, FEATURE_NAMES)
        X_background_scaled = scaler.transform(X_background)
        if X_background_scaled.shape[0] > DEFAULT_BACKGROUND_SAMPLE_SIZE:
            X_background_scaled = shap.sample(
                X_background_scaled, DEFAULT_BACKGROUND_SAMPLE_SIZE
            )
    else:
        # See module docstring: StandardScaler maps the training mean to the
        # origin, so a single all-zero row in scaled space stands in for
        # "the average training student" when no real background is given.
        X_background_scaled = np.zeros((1, len(FEATURE_NAMES)))

    return shap.LinearExplainer(model, X_background_scaled)


def _narrate(factors: List[dict], phrase_map: Dict[str, str]) -> List[str]:
    """Phrases for factors whose |SHAP value| clears the narration floor."""
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
    row: dict,
    risk_factors: List[dict],
    protective_factors: List[dict],
    is_flagged: bool,
    proba: float,
) -> str:
    subject = row.get("subject_slug", "this subject")

    if is_flagged:
        risk_phrases = _narrate(risk_factors, _RISK_PHRASES)
        if risk_phrases:
            body = _join(risk_phrases)
            body = body[0].upper() + body[1:]
            return (
                f"{body}, putting this student at elevated risk in {subject} "
                f"(predicted risk {proba:.0%})."
            )
        return (
            f"This student is flagged as at elevated risk in {subject} "
            f"(predicted risk {proba:.0%}), though no single factor dominates "
            "the prediction."
        )

    protective_phrases = _narrate(protective_factors, _PROTECTIVE_PHRASES)
    if protective_phrases:
        body = _join(protective_phrases)
        return (
            f"This student is not currently flagged as at risk in {subject}: "
            f"{body} (predicted risk {proba:.0%})."
        )
    return (
        f"This student is not currently flagged as at risk in {subject} "
        f"(predicted risk {proba:.0%})."
    )


def generate_student_explanation(
    row: dict,
    model: LogisticRegression,
    scaler: StandardScaler,
    threshold: float = DEFAULT_THRESHOLD,
    background_rows: Optional[List[dict]] = None,
) -> Dict[str, object]:
    """Explain one student-subject-checkpoint prediction with SHAP.

    `row` must carry the 10 feature keys in `FEATURE_NAMES` (plus whatever
    metadata, e.g. `subject_slug`, is used for the human-readable summary).
    `model`/`scaler` are the fitted objects from `train_production_model`
    (its "model"/"scaler" entries, not the whole dict). `background_rows`,
    if given, should be `train_production_model(...)["background_rows"]` (or
    any other list of raw rows) - see the module docstring for the fallback
    used when it's omitted.

    Returns a dict with "predicted_risk_probability", "is_flagged",
    "top_risk_factors" (top 3 by positive SHAP value), "protective_factors"
    (top 3 by negative SHAP value), and "human_readable_summary".
    """
    x = _features_matrix([row], FEATURE_NAMES)
    x_scaled = scaler.transform(x)

    proba = float(model.predict_proba(x_scaled)[0, 1])
    is_flagged = proba >= threshold

    explainer = _build_explainer(model, scaler, background_rows)
    shap_values = np.asarray(explainer.shap_values(x_scaled))[0]

    feature_rows = [
        {
            "feature": name,
            "shap_value": float(shap_values[i]),
            "feature_value": float(row[name]),
        }
        for i, name in enumerate(FEATURE_NAMES)
    ]

    top_risk_factors = sorted(
        (f for f in feature_rows if f["shap_value"] > 0),
        key=lambda f: -f["shap_value"],
    )[:3]
    protective_factors = sorted(
        (f for f in feature_rows if f["shap_value"] < 0),
        key=lambda f: f["shap_value"],
    )[:3]

    summary = _build_summary(row, top_risk_factors, protective_factors, is_flagged, proba)

    return {
        "predicted_risk_probability": proba,
        "is_flagged": is_flagged,
        "top_risk_factors": top_risk_factors,
        "protective_factors": protective_factors,
        "human_readable_summary": summary,
    }
