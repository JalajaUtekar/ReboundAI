"""
ScholarTrack feature engineering - feature skew inspection + leakage-free standardization.

Implements spec section 5, point 7:
    "Feature skew check + standardization (fit strictly on training folds,
    same leakage discipline as Lecture2Revision's existing StandardScaler
    usage) before any model sees the data."

Two pieces:

1. `inspect_feature_skew(rows)` - computes skewness per numeric feature so
   callers can decide (at the training step, not here) whether a log/power
   transform is warranted before fitting the standardizer.
2. `LeakageFreeStandardizer` - a strict fit-then-transform standardizer
   mirroring `Lecture2Revision/lstm_model.py`'s pure-NumPy `StandardScaler`
   fallback: fit exclusively on training-fold rows, store the statistics,
   and never recompute them when transforming other rows (e.g. validation
   or test folds).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from scipy.stats import skew as _scipy_skew
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


# The 10 numeric features that participate in skew inspection and
# standardization. Everything else in a training row is metadata/label and
# must pass through untouched.
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

# Non-numeric metadata/label keys that inspect_feature_skew/standardizer
# must ignore and that transform() must pass through unchanged.
NON_NUMERIC_KEYS: List[str] = [
    "student_id",
    "subject_slug",
    "checkpoint_idx",
    "at_risk",
    "is_failing",
]

# Features naturally expected to be right-skewed: most students submit
# on-time (avg_days_late == 0) or have a stable/decelerating lateness trend
# (delay_acceleration clustered near 0), with a long tail of chronically
# late outliers. This is documentation only - inspect_feature_skew reports
# the actual measured skew for every feature regardless of this expectation.
EXPECTED_RIGHT_SKEWED_FEATURES = {"avg_days_late", "delay_acceleration"}

# Threshold used by the optional suggest_transform() helper below.
_SKEW_TRANSFORM_THRESHOLD = 1.0


def _population_skew(values: np.ndarray) -> float:
    """Population skewness (third standardized moment), used when scipy is
    unavailable. Matches scipy.stats.skew's default bias=True behaviour.
    """
    mean = np.mean(values)
    std = np.std(values)  # population std (ddof=0), matching scipy default
    if std == 0:
        return 0.0
    return float(np.mean((values - mean) ** 3) / std ** 3)


def inspect_feature_skew(rows: List[dict]) -> Dict[str, float]:
    """Compute skewness for each of the 10 numeric features across rows.

    Uses scipy.stats.skew when scipy is installed, otherwise falls back to
    a pure-NumPy implementation of the population skewness formula.
    """
    result: Dict[str, float] = {}
    for key in NUMERIC_FEATURE_KEYS:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        if _SCIPY_AVAILABLE:
            result[key] = float(_scipy_skew(values))
        else:
            result[key] = _population_skew(values)
    return result


def suggest_transform(feature_name: str, skewness: float) -> str:
    """Optional helper: recommend "log1p" for heavily skewed features.

    This is a nice-to-have, not required by the spec item (which leaves
    log/power transformations to the training step). Threshold of
    |skew| > 1 is a common rule of thumb for "substantial" skew.
    """
    if abs(skewness) > _SKEW_TRANSFORM_THRESHOLD:
        return "log1p"
    return "none"


class LeakageFreeStandardizer:
    """Strict fit-then-transform standardizer for the 10 numeric features.

    Mirrors the leakage discipline of Lecture2Revision/lstm_model.py's
    pure-NumPy StandardScaler fallback: `fit()` must be called once on
    training-fold rows only, and `transform()` always applies the stored
    statistics - it never recomputes mean/std from the rows passed to it.
    """

    def __init__(self) -> None:
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self._fitted = False

    def fit(self, rows: List[dict]) -> "LeakageFreeStandardizer":
        """Compute and store per-feature mean/std from the given rows only."""
        for key in NUMERIC_FEATURE_KEYS:
            values = np.array([row[key] for row in rows], dtype=np.float64)
            self.means_[key] = float(np.mean(values))
            # epsilon guards against division by zero for zero-variance
            # features, matching the Lecture2Revision StandardScaler pattern.
            self.stds_[key] = float(np.std(values) + 1e-7)
        self._fitted = True
        return self

    def transform(self, rows: List[dict]) -> List[dict]:
        """Return new row dicts with numeric features standardized using the
        stored fit statistics. Non-numeric keys pass through unchanged.
        """
        if not self._fitted:
            raise RuntimeError(
                "LeakageFreeStandardizer.transform() called before fit(). "
                "Call fit(train_rows) on training-fold rows first, then "
                "transform() on any rows you want standardized - this "
                "class never recomputes statistics from transform() input."
            )

        transformed_rows: List[dict] = []
        for row in rows:
            new_row = dict(row)
            for key in NUMERIC_FEATURE_KEYS:
                if key in row:
                    new_row[key] = (row[key] - self.means_[key]) / self.stds_[key]
            transformed_rows.append(new_row)
        return transformed_rows
