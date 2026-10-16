"""Tests for scripts/training/cv.py."""

from __future__ import annotations

import json
import os

import pytest

from scripts.training.cv import (
    compute_class_weights,
    grouped_student_kfold,
    oversample_training_rows,
)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "training_rows.json",
)


def _load_training_rows():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _student_at_risk_map(rows):
    """A student is at-risk if ANY of their subjects has at_risk=True."""
    mapping = {}
    for row in rows:
        sid = row["student_id"]
        mapping[sid] = mapping.get(sid, False) or bool(row["at_risk"])
    return mapping


def _make_synthetic_rows(n_not_at_risk_students=20, n_at_risk_students=15, n_checkpoints=6):
    """Build small synthetic student-subject data for cheap unit tests."""
    rows = []
    for i in range(n_not_at_risk_students):
        sid = f"NR{i:03d}"
        for cp in range(n_checkpoints):
            rows.append(_make_row(sid, "DSML", cp, at_risk=False))
    for i in range(n_at_risk_students):
        sid = f"AR{i:03d}"
        for cp in range(n_checkpoints):
            rows.append(_make_row(sid, "DSML", cp, at_risk=True))
    return rows


def _make_row(student_id, subject_slug, checkpoint_idx, at_risk):
    return {
        "student_id": student_id,
        "subject_slug": subject_slug,
        "checkpoint_idx": checkpoint_idx,
        "internal_marks_pct": 50.0 + checkpoint_idx,
        "assignment_marks_pct": 60.0 + checkpoint_idx,
        "attendance_pct": 80.0 - checkpoint_idx,
        "on_time_submission_ratio": 0.9,
        "avg_days_late": 1.0,
        "attendance_velocity": -0.5,
        "marks_velocity": 1.0,
        "delay_acceleration": 0.0,
        "attendance_deficit_margin": 5.0,
        "marks_deficit_margin": -10.0 if at_risk else 10.0,
        "at_risk": at_risk,
        "is_failing": False,
    }


class TestGroupedStudentKFold:
    def test_no_student_in_both_train_and_val_same_fold(self):
        rows = _load_training_rows()
        student_ids = sorted({r["student_id"] for r in rows})
        at_risk_map = _student_at_risk_map(rows)

        folds = grouped_student_kfold(student_ids, at_risk_map, n_splits=5, seed=42)

        for train_ids, val_ids in folds:
            overlap = set(train_ids) & set(val_ids)
            assert not overlap, f"Leakage: students in both train and val: {overlap}"

    def test_every_student_in_exactly_one_val_set(self):
        rows = _load_training_rows()
        student_ids = sorted({r["student_id"] for r in rows})
        at_risk_map = _student_at_risk_map(rows)

        folds = grouped_student_kfold(student_ids, at_risk_map, n_splits=5, seed=42)

        val_membership_count = {sid: 0 for sid in student_ids}
        for _, val_ids in folds:
            for sid in val_ids:
                val_membership_count[sid] += 1

        for sid, count in val_membership_count.items():
            assert count == 1, f"Student {sid} appeared in {count} val sets, expected 1"

    def test_train_val_union_is_all_students_each_fold(self):
        rows = _load_training_rows()
        student_ids = sorted({r["student_id"] for r in rows})
        at_risk_map = _student_at_risk_map(rows)

        folds = grouped_student_kfold(student_ids, at_risk_map, n_splits=5, seed=42)

        for train_ids, val_ids in folds:
            assert set(train_ids) | set(val_ids) == set(student_ids)

    def test_at_risk_students_reasonably_distributed_across_folds(self):
        rows = _load_training_rows()
        student_ids = sorted({r["student_id"] for r in rows})
        at_risk_map = _student_at_risk_map(rows)
        total_at_risk = sum(1 for v in at_risk_map.values() if v)

        folds = grouped_student_kfold(student_ids, at_risk_map, n_splits=5, seed=42)

        counts = []
        for _, val_ids in folds:
            n_at_risk_in_val = sum(1 for sid in val_ids if at_risk_map.get(sid, False))
            counts.append(n_at_risk_in_val)

        # No fold should get zero at-risk students.
        assert all(c > 0 for c in counts), f"Some fold got zero at-risk students: {counts}"
        # No fold should get a wildly disproportionate share (sanity: no
        # fold holds more than ~2x the expected even share).
        expected_even_share = total_at_risk / 5
        assert all(c <= expected_even_share * 2.5 for c in counts), counts
        assert sum(counts) == total_at_risk

    def test_raises_when_fewer_at_risk_students_than_n_splits(self):
        student_ids = [f"S{i}" for i in range(10)]
        at_risk_map = {sid: False for sid in student_ids}
        at_risk_map["S0"] = True  # only 1 at-risk student, need 5 folds

        with pytest.raises(ValueError):
            grouped_student_kfold(student_ids, at_risk_map, n_splits=5, seed=42)

    def test_deterministic_given_seed(self):
        rows = _load_training_rows()
        student_ids = sorted({r["student_id"] for r in rows})
        at_risk_map = _student_at_risk_map(rows)

        folds_a = grouped_student_kfold(student_ids, at_risk_map, n_splits=5, seed=7)
        folds_b = grouped_student_kfold(student_ids, at_risk_map, n_splits=5, seed=7)

        assert folds_a == folds_b


class TestOversampleTrainingRows:
    def test_increases_at_risk_student_subject_count(self):
        rows = _make_synthetic_rows(n_not_at_risk_students=20, n_at_risk_students=4)

        def count_at_risk_trajectories(data):
            keys = {
                (r["student_id"], r["subject_slug"])
                for r in data
                if r["at_risk"]
            }
            return len(keys)

        before = count_at_risk_trajectories(rows)
        output = oversample_training_rows(rows, target_ratio=1.0, seed=42)
        after = count_at_risk_trajectories(output)

        assert after > before

    def test_original_rows_present_and_unchanged(self):
        rows = _make_synthetic_rows(n_not_at_risk_students=20, n_at_risk_students=4)
        output = oversample_training_rows(rows, target_ratio=1.0, seed=42)

        # Every original row should appear byte-identical (by value) in output.
        output_signatures = [tuple(sorted(r.items())) for r in output]
        for row in rows:
            sig = tuple(sorted(row.items()))
            assert sig in output_signatures

    def test_duplicated_rows_are_jittered_but_keep_checkpoint_structure(self):
        rows = _make_synthetic_rows(n_not_at_risk_students=20, n_at_risk_students=4)
        output = oversample_training_rows(rows, target_ratio=1.0, seed=42)

        dup_rows = [r for r in output if "_dup" in r["student_id"]]
        assert dup_rows, "Expected at least one duplicated/jittered trajectory"

        # Group dup rows by synthetic student_id and check checkpoint_idx
        # structure matches a real trajectory (0..5, one each).
        by_student: dict = {}
        for r in dup_rows:
            by_student.setdefault(r["student_id"], []).append(r)

        for dup_student_id, dup_group in by_student.items():
            checkpoint_idxs = sorted(r["checkpoint_idx"] for r in dup_group)
            assert checkpoint_idxs == list(range(6))

            # subject_slug/checkpoint_idx/at_risk/is_failing untouched;
            # numeric features should differ from any original row's values
            # for the same checkpoint (jittered), i.e. not byte-identical.
            original_student_id = dup_student_id.split("_dup")[0]
            originals = [
                r for r in rows
                if r["student_id"] == original_student_id
            ]
            for dup_row in dup_group:
                orig_row = next(
                    r for r in originals if r["checkpoint_idx"] == dup_row["checkpoint_idx"]
                )
                assert dup_row["subject_slug"] == orig_row["subject_slug"]
                assert dup_row["at_risk"] == orig_row["at_risk"]
                assert dup_row["is_failing"] == orig_row["is_failing"]
                assert dup_row["internal_marks_pct"] != orig_row["internal_marks_pct"]

    def test_no_at_risk_rows_does_not_crash(self):
        rows = _make_synthetic_rows(n_not_at_risk_students=10, n_at_risk_students=0)
        output = oversample_training_rows(rows, target_ratio=1.0, seed=42)
        # Documented behavior: no-op when there are no at-risk trajectories.
        assert output == rows

    def test_target_ratio_zero_is_noop(self):
        rows = _make_synthetic_rows(n_not_at_risk_students=20, n_at_risk_students=4)
        output = oversample_training_rows(rows, target_ratio=0.0, seed=42)
        assert len(output) == len(rows)


class TestComputeClassWeights:
    def test_known_ratio_hand_verified(self):
        # 8 negative, 2 positive out of 10 total, 2 classes.
        labels = [False] * 8 + [True] * 2
        weights = compute_class_weights(labels)

        # weight_negative = 10 / (2 * 8) = 0.625
        # weight_positive = 10 / (2 * 2) = 2.5
        assert weights[0] == pytest.approx(0.625)
        assert weights[1] == pytest.approx(2.5)

    def test_balanced_labels_give_equal_weights(self):
        labels = [False] * 5 + [True] * 5
        weights = compute_class_weights(labels)
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(1.0)

    def test_raises_on_empty_labels(self):
        with pytest.raises(ValueError):
            compute_class_weights([])

    def test_raises_when_missing_a_class(self):
        with pytest.raises(ValueError):
            compute_class_weights([False, False, False])
