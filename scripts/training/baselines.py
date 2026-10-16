"""
ScholarTrack baseline models - spec section 5, point 8.

Implements:
    "Baseline comparison: Logistic Regression and Gradient Boosting
    (scikit-learn), trained and evaluated under the identical CV protocol
    as the Conv1D model - the neural net has to beat the baselines, not
    just exist."

Both `train_and_predict_oof_logreg` and `train_and_predict_oof_gbdt` share
the same out-of-fold (OOF) protocol:

    for each (train_student_ids, val_student_ids) fold:
        1. Split `rows` into this fold's train/val rows by student_id.
        2. Oversample the TRAIN rows only, via
           scripts.training.cv.oversample_training_rows (never the val
           rows - spec section 5 point 5: validation stays on the real
           class distribution).
        3. Fit the model on the (possibly oversampled) train rows.
        4. Predict probabilities on the fold's *original* val rows.
        5. Record those val predictions as this fold's OOF contribution.

Every row in `rows` is a member of exactly one fold's val set (folds
produced by `grouped_student_kfold` partition all student_ids across val
sets), so the collected OOF predictions cover the whole input dataset
exactly once, with no duplicates and no leakage: a row is only ever
predicted by a model that did not see its student_id during training.

Logistic Regression additionally standardizes features: a fresh
`StandardScaler` is fit on that fold's train rows only and used to
transform both train and val rows (never refit on val - the same
leakage discipline as the feature-engineering step's
`LeakageFreeStandardizer`, spec section 5 point 7). Gradient Boosting
(HistGradientBoostingClassifier) is tree-based and skips standardization
entirely, since tree splits are invariant to monotonic per-feature
rescaling.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.training.cv import oversample_training_rows

# The 10 numeric feature keys used by both baseline models, matching
# scripts/feature_engineering/standardizer.py's NUMERIC_FEATURE_KEYS and
# scripts/training/cv.py's NUMERIC_FEATURE_KEYS.
DEFAULT_FEATURE_NAMES: List[str] = [
    "internal_marks_pct",
    "assignment_marks_pct",
    "attendance_pct",
    "on_time_submission_ratio",
    "avg_days_late",
    "attendance_velocity",
    "marks_velocity",
    "delay_acceleration",
    "attendance_deficit_margin",
    "marks_deficit_margin",
]


def _rows_for_students(rows: List[dict], student_ids: List[str]) -> List[dict]:
    """Return the subset of `rows` whose student_id is in `student_ids`."""
    wanted = set(student_ids)
    return [row for row in rows if row["student_id"] in wanted]


def _features_matrix(rows: List[dict], feature_names: List[str]) -> np.ndarray:
    return np.array(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=float,
    )


def _labels_array(rows: List[dict]) -> np.ndarray:
    return np.array([bool(row["at_risk"]) for row in rows], dtype=bool)


def _empty_oof_result() -> Dict[str, list]:
    return {
        "student_id": [],
        "subject_slug": [],
        "checkpoint_idx": [],
        "y_true": [],
        "y_proba": [],
    }


def _append_fold_predictions(
    result: Dict[str, list],
    val_rows: List[dict],
    y_proba: np.ndarray,
) -> None:
    for row, proba in zip(val_rows, y_proba):
        result["student_id"].append(row["student_id"])
        result["subject_slug"].append(row["subject_slug"])
        result["checkpoint_idx"].append(row["checkpoint_idx"])
        result["y_true"].append(bool(row["at_risk"]))
        result["y_proba"].append(float(proba))


def train_and_predict_oof_logreg(
    rows: List[dict],
    folds: List[tuple],
    feature_names: Optional[List[str]] = None,
    C: float = 1.0,
    oversample_target_ratio: float = 1.0,
) -> Dict[str, list]:
    """Train/predict LogisticRegression under the grouped-CV OOF protocol.

    Per fold: oversample train rows (cv.oversample_training_rows), fit a
    fresh StandardScaler on those (oversampled) train rows, transform both
    train and val rows with it, then fit
    `LogisticRegression(C=C, class_weight='balanced', penalty='l2',
    max_iter=1000)` on the scaled train rows and predict probabilities on
    the scaled val rows.

    `C` (inverse regularization strength, default 1.0 - scikit-learn's own
    default, so existing callers are unaffected) and
    `oversample_target_ratio` (forwarded as `target_ratio` to
    `oversample_training_rows`, default 1.0 - that function's own default)
    are optional tuning knobs added for scripts/training/tune_logreg.py;
    neither changes behavior for callers that don't pass them.

    Returns parallel lists under keys "student_id", "subject_slug",
    "checkpoint_idx", "y_true", "y_proba" - one entry per row across all
    folds' validation sets (i.e. one entry per input row, in whichever
    order the folds were visited).
    """
    names = feature_names if feature_names is not None else DEFAULT_FEATURE_NAMES
    result = _empty_oof_result()

    for train_student_ids, val_student_ids in folds:
        train_rows = _rows_for_students(rows, train_student_ids)
        val_rows = _rows_for_students(rows, val_student_ids)
        if not val_rows:
            continue

        oversampled_train_rows = oversample_training_rows(
            train_rows, target_ratio=oversample_target_ratio
        )

        X_train = _features_matrix(oversampled_train_rows, names)
        y_train = _labels_array(oversampled_train_rows)
        X_val = _features_matrix(val_rows, names)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(
            C=C, class_weight="balanced", penalty="l2", max_iter=1000
        )
        model.fit(X_train_scaled, y_train)

        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        _append_fold_predictions(result, val_rows, y_proba)

    return result


def train_and_predict_oof_gbdt(
    rows: List[dict],
    folds: List[tuple],
    feature_names: Optional[List[str]] = None,
    max_depth: Optional[int] = 3,
    min_samples_leaf: int = 5,
    learning_rate: float = 0.1,
    l2_regularization: float = 0.0,
    max_iter: int = 100,
    oversample_target_ratio: float = 1.0,
) -> Dict[str, list]:
    """Train/predict HistGradientBoostingClassifier under the grouped-CV
    OOF protocol.

    Same shape/contract as `train_and_predict_oof_logreg`, but trains
    `HistGradientBoostingClassifier(class_weight='balanced', max_depth=3,
    min_samples_leaf=5, random_state=42)` on the (oversampled) train rows
    and predicts on the val rows directly - no feature standardization,
    since tree splits are invariant to monotonic per-feature rescaling.

    `max_depth`, `min_samples_leaf`, `learning_rate`, `l2_regularization`,
    and `max_iter` are optional tuning knobs forwarded straight to
    `HistGradientBoostingClassifier`, added for
    scripts/training/tune_gbdt.py; their defaults match the previous
    hardcoded values (`learning_rate`, `l2_regularization`, and `max_iter`
    match scikit-learn's own HistGradientBoostingClassifier defaults), so
    existing callers that don't pass them are unaffected.
    `oversample_target_ratio` is forwarded as `target_ratio` to
    `oversample_training_rows` (default 1.0 - that function's own
    default), same as the LogisticRegression variant.
    """
    names = feature_names if feature_names is not None else DEFAULT_FEATURE_NAMES
    result = _empty_oof_result()

    for train_student_ids, val_student_ids in folds:
        train_rows = _rows_for_students(rows, train_student_ids)
        val_rows = _rows_for_students(rows, val_student_ids)
        if not val_rows:
            continue

        oversampled_train_rows = oversample_training_rows(
            train_rows, target_ratio=oversample_target_ratio
        )

        X_train = _features_matrix(oversampled_train_rows, names)
        y_train = _labels_array(oversampled_train_rows)
        X_val = _features_matrix(val_rows, names)

        model = HistGradientBoostingClassifier(
            class_weight="balanced",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            max_iter=max_iter,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        _append_fold_predictions(result, val_rows, y_proba)

    return result
