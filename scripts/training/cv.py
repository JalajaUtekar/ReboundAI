"""
ScholarTrack cross-validation utilities - spec section 5, points 2-4.

Implements:
    1. `grouped_student_kfold` - stratified, STUDENT-grouped k-fold splitting
       (spec point 3). Splitting must happen at the student_id level, never
       the row level: `data/training_rows.json` carries ~6 correlated
       "as-of checkpoint" prefix rows per student-subject (see
       docs/superpowers/specs/2026-10-16-rebound-ai-design.md section 4's
       no-leakage construction). If two prefix rows from the same student
       landed in different sides of a split, the model would effectively
       see (a near-duplicate of) its own validation example during
       training - that is the leakage this function exists to prevent.
    2. `oversample_training_rows` - group-aware oversampling of TRAINING
       rows only (spec point 4), operating on whole student-subject
       trajectories rather than individual rows.
    3. `compute_class_weights` - standard "balanced" class-weight formula
       (spec point 2).
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

# The 10 numeric feature keys that get jittered during oversampling. Must
# stay in sync with scripts/feature_engineering/standardizer.py's
# NUMERIC_FEATURE_KEYS - duplicated here (rather than imported) so this
# module has no dependency on the feature-engineering module's import
# path staying stable while multiple agents touch neighboring files.
NUMERIC_FEATURE_KEYS: List[str] = [
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

# Relative jitter magnitude for oversampling: noise is drawn from
# N(0, (_JITTER_STD_FRACTION * |value| or a small floor)^2).
_JITTER_STD_FRACTION = 0.03
_JITTER_STD_FLOOR = 0.5


def grouped_student_kfold(
    student_ids: List[str],
    student_at_risk: Dict[str, bool],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """Return `n_splits` (train_student_ids, val_student_ids) folds.

    Splitting happens at the student level - every row belonging to a given
    student_id lands entirely in train or entirely in val for a given fold,
    which is the leakage guard described in the module docstring.

    Stratification: at-risk and not-at-risk students are shuffled and
    distributed round-robin into folds *separately*, so each fold's
    validation set gets a roughly proportional share of at-risk students
    (not an exact share - with ~12-18 at-risk students over 5 folds exact
    equality isn't meaningful, but no fold should end up with zero at-risk
    students or a wildly disproportionate share).
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    unique_ids = list(dict.fromkeys(student_ids))  # de-dup, preserve order
    at_risk_ids = [sid for sid in unique_ids if student_at_risk.get(sid, False)]
    not_at_risk_ids = [sid for sid in unique_ids if not student_at_risk.get(sid, False)]

    if len(at_risk_ids) < n_splits:
        raise ValueError(
            f"Only {len(at_risk_ids)} at-risk students available for "
            f"{n_splits}-fold stratified splitting. Per spec section 5 "
            "point 3, fall back to leave-one-(positive-student)-out CV "
            "when there are fewer than n_splits at-risk students."
        )

    rng = random.Random(seed)
    at_risk_shuffled = at_risk_ids[:]
    not_at_risk_shuffled = not_at_risk_ids[:]
    rng.shuffle(at_risk_shuffled)
    rng.shuffle(not_at_risk_shuffled)

    # Round-robin assignment into n_splits buckets, done separately for each
    # stratum so each fold's val bucket gets a proportional share of both.
    at_risk_buckets: List[List[str]] = [[] for _ in range(n_splits)]
    for i, sid in enumerate(at_risk_shuffled):
        at_risk_buckets[i % n_splits].append(sid)

    not_at_risk_buckets: List[List[str]] = [[] for _ in range(n_splits)]
    for i, sid in enumerate(not_at_risk_shuffled):
        not_at_risk_buckets[i % n_splits].append(sid)

    folds: List[Tuple[List[str], List[str]]] = []
    for fold_idx in range(n_splits):
        val_ids = at_risk_buckets[fold_idx] + not_at_risk_buckets[fold_idx]
        val_id_set = set(val_ids)
        train_ids = [sid for sid in unique_ids if sid not in val_id_set]
        folds.append((train_ids, val_ids))

    return folds


def compute_class_weights(labels: List[bool]) -> Dict[int, float]:
    """Standard "balanced" class weights: n_samples / (n_classes * count_c).

    Returns {0: weight_for_negative, 1: weight_for_positive}.
    """
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("compute_class_weights requires at least one label")

    n_positive = sum(1 for label in labels if label)
    n_negative = n_samples - n_positive

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "compute_class_weights requires at least one example of each "
            "class (got "
            f"{n_negative} negative, {n_positive} positive)"
        )

    n_classes = 2
    weight_negative = n_samples / (n_classes * n_negative)
    weight_positive = n_samples / (n_classes * n_positive)
    return {0: weight_negative, 1: weight_positive}


def oversample_training_rows(
    train_rows: List[dict],
    target_ratio: float = 1.0,
    seed: int = 42,
) -> List[dict]:
    """Group-aware oversampling of at-risk student-subject trajectories.

    IMPORTANT - training rows only. Call this ONLY on a fold's training
    rows, NEVER on validation/test rows (spec section 5 point 5:
    "Test/validation folds are never oversampled - evaluation stays on the
    real class distribution"). Passing validation rows here would corrupt
    the evaluation by inflating the apparent positive rate and introducing
    duplicated/jittered examples into the metric computation.

    WHY trajectory-level, not row-level (spec section 5 point 4): each
    student-subject contributes ~6 "as-of checkpoint" prefix rows that are
    heavily correlated snapshots of the *same* underlying decline (see
    scripts/build_training_rows.py). Naive row-level SMOTE/duplication
    would pick individual prefix rows and interpolate/duplicate them
    independently, which mostly re-mixes a single student's own trend with
    itself rather than adding genuinely new synthetic diversity, and can
    even break the checkpoint_idx structure (e.g. producing a "trajectory"
    with two checkpoint_idx=3 rows and no checkpoint_idx=1). Instead, this
    function groups by (student_id, subject_slug), duplicates a WHOLE
    matched trajectory (all of its prefix rows together, keeping
    checkpoint_idx structure intact) as one unit, and only jitters the 10
    numeric feature columns - student_id/subject_slug/checkpoint_idx/labels
    are left untouched except for the synthetic student_id suffix below.

    Duplicated trajectories get a synthetic student_id suffix (`_dup1`,
    `_dup2`, ...) appended to the original student_id, so they never
    collide with real student IDs and so tests/callers can identify which
    rows are synthetic.

    target_ratio=1.0 means oversampling continues, duplicating whole
    at-risk trajectories round-robin, until the number of at-risk
    student-subjects is approximately equal to (>= and as close as
    possible to) target_ratio * (number of not-at-risk student-subjects).
    A target_ratio of 0 or an input with no at-risk rows at all returns the
    input unchanged (documented no-op, not an error) - see the function's
    final branch.
    """
    if not train_rows:
        return list(train_rows)

    rng = random.Random(seed)

    # Group rows into trajectories keyed by (student_id, subject_slug).
    trajectories: Dict[Tuple[str, str], List[dict]] = {}
    order: List[Tuple[str, str]] = []
    for row in train_rows:
        key = (row["student_id"], row["subject_slug"])
        if key not in trajectories:
            trajectories[key] = []
            order.append(key)
        trajectories[key].append(row)

    def _is_at_risk_trajectory(rows: List[dict]) -> bool:
        return any(r.get("at_risk", False) for r in rows)

    at_risk_keys = [k for k in order if _is_at_risk_trajectory(trajectories[k])]
    not_at_risk_keys = [k for k in order if not _is_at_risk_trajectory(trajectories[k])]

    output_rows: List[dict] = list(train_rows)  # all originals, unchanged

    if not at_risk_keys or target_ratio <= 0:
        # No-op: nothing to oversample, or caller explicitly asked for none.
        return output_rows

    n_not_at_risk = len(not_at_risk_keys)
    target_at_risk_count = round(target_ratio * n_not_at_risk)

    current_at_risk_count = len(at_risk_keys)
    dup_counter = 0
    cycle_idx = 0
    while current_at_risk_count < target_at_risk_count:
        source_key = at_risk_keys[cycle_idx % len(at_risk_keys)]
        source_rows = trajectories[source_key]

        dup_counter += 1
        new_student_id = f"{source_key[0]}_dup{dup_counter}"

        for row in source_rows:
            new_row = copy.deepcopy(row)
            new_row["student_id"] = new_student_id
            for feature_key in NUMERIC_FEATURE_KEYS:
                if feature_key in new_row:
                    original_value = row[feature_key]
                    noise_std = max(
                        abs(original_value) * _JITTER_STD_FRACTION,
                        _JITTER_STD_FLOOR,
                    )
                    new_row[feature_key] = original_value + rng.gauss(0.0, noise_std)
            output_rows.append(new_row)

        current_at_risk_count += 1
        cycle_idx += 1

    return output_rows
