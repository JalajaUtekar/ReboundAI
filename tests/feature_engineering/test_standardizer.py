"""Tests for scripts/feature_engineering/standardizer.py."""

import numpy as np
import pytest

from scripts.feature_engineering.standardizer import (
    NUMERIC_FEATURE_KEYS,
    LeakageFreeStandardizer,
    inspect_feature_skew,
)


def _make_row(value: float, **overrides) -> dict:
    """Build a hand-constructed training row with all 10 numeric features
    set to `value`, plus metadata keys, with optional per-feature overrides.
    """
    row = {key: value for key in NUMERIC_FEATURE_KEYS}
    row.update(
        {
            "student_id": "S001",
            "subject_slug": "data-structures",
            "checkpoint_idx": 2,
            "at_risk": False,
            "is_failing": False,
        }
    )
    row.update(overrides)
    return row


class TestInspectFeatureSkew:
    def test_returns_value_for_every_numeric_key(self):
        rows = [_make_row(float(i)) for i in range(10)]
        result = inspect_feature_skew(rows)
        assert set(result.keys()) == set(NUMERIC_FEATURE_KEYS)
        for key in NUMERIC_FEATURE_KEYS:
            assert isinstance(result[key], float)

    def test_identifies_skewed_feature_as_high_positive_skew(self):
        # avg_days_late: mostly zeros (on-time), one large outlier (chronic
        # lateness) - classic right-skewed shape.
        skewed_values = [0.0] * 19 + [50.0]
        rows = [
            _make_row(5.0, avg_days_late=v) for v in skewed_values
        ]
        result = inspect_feature_skew(rows)
        assert result["avg_days_late"] > 1.5

    def test_identifies_symmetric_feature_as_near_zero_skew(self):
        # attendance_pct: evenly spread around a center -> symmetric.
        symmetric_values = [40.0, 45.0, 50.0, 55.0, 60.0, 45.0, 50.0, 55.0]
        rows = [
            _make_row(5.0, attendance_pct=v) for v in symmetric_values
        ]
        result = inspect_feature_skew(rows)
        assert abs(result["attendance_pct"]) < 0.3


class TestLeakageFreeStandardizer:
    def test_fit_transform_same_rows_gives_mean_zero_std_one(self):
        rows = [_make_row(float(v)) for v in [10, 20, 30, 40, 50, 60, 70, 80]]
        standardizer = LeakageFreeStandardizer()
        standardizer.fit(rows)
        transformed = standardizer.transform(rows)

        for key in NUMERIC_FEATURE_KEYS:
            values = np.array([row[key] for row in transformed])
            assert abs(np.mean(values)) < 1e-6
            assert abs(np.std(values) - 1.0) < 1e-3

    def test_transform_uses_fit_statistics_not_transformed_rows(self):
        # Fit on rows centered around 50.
        train_rows = [_make_row(float(v)) for v in [40, 45, 50, 55, 60]]
        # Transform a DIFFERENT set of rows centered around 80.
        other_rows = [_make_row(float(v)) for v in [75, 78, 80, 82, 85]]

        standardizer = LeakageFreeStandardizer()
        standardizer.fit(train_rows)
        transformed = standardizer.transform(other_rows)

        fit_mean = standardizer.means_["internal_marks_pct"]
        fit_std = standardizer.stds_["internal_marks_pct"]
        assert abs(fit_mean - 50.0) < 1e-6

        # Manually recompute what transform SHOULD produce using the
        # fit-set statistics (never the other_rows statistics).
        expected = [(v - fit_mean) / fit_std for v in [75, 78, 80, 82, 85]]
        actual = [row["internal_marks_pct"] for row in transformed]
        for exp, act in zip(expected, actual):
            assert abs(exp - act) < 1e-9

        # Because other_rows is centered far from the fit mean, the
        # transformed values should NOT be centered near 0 - that would
        # indicate the statistics were (incorrectly) recomputed from
        # other_rows itself.
        assert np.mean(actual) > 3.0

    def test_transform_before_fit_raises_clear_error(self):
        standardizer = LeakageFreeStandardizer()
        rows = [_make_row(10.0)]
        with pytest.raises(RuntimeError, match="fit"):
            standardizer.transform(rows)

    def test_non_numeric_keys_pass_through_unchanged(self):
        rows = [
            _make_row(
                10.0,
                student_id="S042",
                subject_slug="operating-systems",
                checkpoint_idx=3,
                at_risk=True,
                is_failing=False,
            ),
            _make_row(20.0, student_id="S043"),
        ]
        standardizer = LeakageFreeStandardizer()
        standardizer.fit(rows)
        transformed = standardizer.transform(rows)

        assert transformed[0]["student_id"] == "S042"
        assert transformed[0]["subject_slug"] == "operating-systems"
        assert transformed[0]["checkpoint_idx"] == 3
        assert transformed[0]["at_risk"] is True
        assert transformed[0]["is_failing"] is False
        assert transformed[1]["student_id"] == "S043"
