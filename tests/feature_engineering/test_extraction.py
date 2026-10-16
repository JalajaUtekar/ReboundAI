import copy
import json
import random
from pathlib import Path

import pytest

from scripts.data_generation.generator import (
    CHECKPOINTS_IN_TIME_ORDER,
    generate_dataset,
)
from scripts.feature_engineering.extraction import (
    FEATURE_NAMES,
    MIN_PREFIX_LEN,
    MAX_PREFIX_LEN,
    build_training_rows,
    extract_prefix_features,
)

REAL_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic_students.json"


def _tiny_dataset(seed=1, num_students=3):
    return generate_dataset(num_students=num_students, seed=seed)


def test_checkpoints_in_time_order_has_seven_entries():
    # Sanity check on the generator's own contract this module depends on.
    assert len(CHECKPOINTS_IN_TIME_ORDER) == 7


def test_build_training_rows_count_matches_students_times_subjects_times_valid_prefixes():
    dataset = _tiny_dataset(num_students=5)
    rows = build_training_rows(dataset)
    num_subjects = sum(len(s["subjects"]) for s in dataset)
    valid_prefixes = MAX_PREFIX_LEN - MIN_PREFIX_LEN + 1  # checkpoints 2..7 inclusive == 6
    assert valid_prefixes == 6
    assert len(rows) == num_subjects * valid_prefixes


def test_real_dataset_row_count_is_4320():
    dataset = json.loads(REAL_DATASET_PATH.read_text(encoding="utf-8"))
    assert len(dataset) == 120
    rows = build_training_rows(dataset)
    num_subjects = sum(len(s["subjects"]) for s in dataset)
    assert num_subjects == 720
    assert len(rows) == 4320


def test_every_row_has_all_feature_names_and_metadata():
    dataset = _tiny_dataset(num_students=2)
    rows = build_training_rows(dataset)
    for row in rows:
        for name in FEATURE_NAMES:
            assert name in row
        assert "student_id" in row
        assert "subject_slug" in row
        assert "checkpoint_idx" in row
        assert "at_risk" in row
        assert "is_failing" in row


def test_checkpoint_idx_ranges_from_2_to_7_per_subject():
    dataset = _tiny_dataset(num_students=2)
    rows = build_training_rows(dataset)
    student = dataset[0]
    subject = student["subjects"][0]
    sub_rows = [
        r for r in rows if r["student_id"] == student["student_id"] and r["subject_slug"] == subject["subject_slug"]
    ]
    assert sorted(r["checkpoint_idx"] for r in sub_rows) == [2, 3, 4, 5, 6, 7]


def test_label_matches_source_outcome_exactly_for_every_row():
    dataset = _tiny_dataset(num_students=6)
    rows = build_training_rows(dataset)
    outcomes = {
        (s["student_id"], subj["subject_slug"]): subj["outcome"]
        for s in dataset
        for subj in s["subjects"]
    }
    for row in rows:
        outcome = outcomes[(row["student_id"], row["subject_slug"])]
        assert row["at_risk"] == outcome["at_risk"]
        assert row["is_failing"] == outcome["is_failing"]


def test_no_leakage_features_unchanged_when_later_checkpoints_added():
    """The core leakage-freedom property: features computed at a given prefix length
    must be identical whether or not later checkpoints' data exists in the source record.
    """
    dataset = _tiny_dataset(num_students=4)
    student = dataset[0]
    subject = student["subjects"][0]

    full_features = extract_prefix_features(subject, prefix_len=3)

    # Build a truncated copy that only contains data up through the 3rd checkpoint
    # in time order (submission idx0, submission idx1, submission idx2 -- see
    # CHECKPOINTS_IN_TIME_ORDER), simulating a mid-term snapshot where checkpoints
    # 4-7 (unit_test 0, submission 3, submission 4, unit_test 1) have not happened yet.
    truncated = copy.deepcopy(subject)
    truncated["unit_tests"] = []  # neither unit test has occurred by checkpoint 3
    truncated["submissions"] = truncated["submissions"][:3]  # only first 3 submissions
    # Attendance weeks beyond the 3rd checkpoint's week should not exist either.
    checkpoint_3_week = 6  # week of submission idx2 (see SUBMISSION_WEEKS)
    truncated["attendance_weeks"] = [
        w for w in truncated["attendance_weeks"] if w["week"] <= checkpoint_3_week
    ]

    truncated_features = extract_prefix_features(truncated, prefix_len=3)

    assert full_features == truncated_features


def test_no_leakage_across_all_valid_prefixes_for_several_subjects():
    dataset = _tiny_dataset(num_students=3)
    for student in dataset:
        for subject in student["subjects"]:
            for prefix_len in range(MIN_PREFIX_LEN, MAX_PREFIX_LEN + 1):
                features_full = extract_prefix_features(subject, prefix_len)

                truncated = copy.deepcopy(subject)
                prefix_checkpoints = CHECKPOINTS_IN_TIME_ORDER[:prefix_len]
                kept_ut_indices = sorted(idx for kind, idx in prefix_checkpoints if kind == "unit_test")
                kept_sub_indices = sorted(idx for kind, idx in prefix_checkpoints if kind == "submission")
                truncated["unit_tests"] = [truncated["unit_tests"][i] for i in kept_ut_indices]
                truncated["submissions"] = [truncated["submissions"][i] for i in kept_sub_indices]
                last_week = max(
                    (
                        truncated_orig_week(subject, kind, idx)
                        for kind, idx in prefix_checkpoints
                    )
                )
                truncated["attendance_weeks"] = [
                    w for w in subject["attendance_weeks"] if w["week"] <= last_week
                ]

                features_truncated = extract_prefix_features(truncated, prefix_len)
                assert features_full == features_truncated, (
                    f"Leakage detected at prefix_len={prefix_len}"
                )


def truncated_orig_week(subject, kind, idx):
    if kind == "unit_test":
        return subject["unit_tests"][idx]["week"]
    return subject["submissions"][idx]["due_day"] // 7


def test_marks_deficit_margin_anchored_to_70_not_40():
    dataset = _tiny_dataset(num_students=3)
    subject = dataset[0]["subjects"][0]
    features = extract_prefix_features(subject, prefix_len=4)

    th_so_far = sum(ut["score"] for ut in subject["unit_tests"][:1])  # ut0 included by checkpoint 4
    max_th_so_far = sum(ut["max_marks"] for ut in subject["unit_tests"][:1])
    pr_tw_so_far = sum(s["score"] for s in subject["submissions"][:3])
    max_pr_tw_so_far = sum(s["max_marks"] for s in subject["submissions"][:3])
    aggregate_pct_so_far = (th_so_far + pr_tw_so_far) / (max_th_so_far + max_pr_tw_so_far) * 100

    assert features["marks_deficit_margin"] == pytest.approx(aggregate_pct_so_far - 70.0)
    assert features["marks_deficit_margin"] != pytest.approx(aggregate_pct_so_far - 40.0)


def test_velocity_features_hand_constructed_example():
    """Hand-constructed subject record with known checkpoint values, to verify the
    exact deltas/slopes the engineered features are meant to compute.
    """
    subject = {
        "subject_slug": "test_subj",
        "components": {"TH": 40, "ETE": 60, "PR": 25},
        "pr_tw_component": "PR",
        "unit_tests": [
            {"index": 1, "week": 6, "max_marks": 20.0, "score": 10.0},  # 50%
            {"index": 2, "week": 12, "max_marks": 20.0, "score": 16.0},  # 80%
        ],
        "submissions": [
            {"index": 1, "due_day": 14, "submitted_day": 14, "max_marks": 5.0, "score": 2.5},  # 50%, on time
            {"index": 2, "due_day": 28, "submitted_day": 30, "max_marks": 5.0, "score": 3.0},  # 60%, 2 days late
            {"index": 3, "due_day": 42, "submitted_day": 44, "max_marks": 5.0, "score": 4.0},  # 80%, 2 days late
            {"index": 4, "due_day": 56, "submitted_day": 56, "max_marks": 5.0, "score": 4.5},  # 90%, on time
            {"index": 5, "due_day": 70, "submitted_day": 75, "max_marks": 5.0, "score": 5.0},  # 100%, 5 days late
        ],
        "attendance_weeks": [{"week": w, "attendance_pct": 80.0 + w} for w in range(1, 15)],
        "outcome": {"at_risk": False, "is_failing": False},
    }

    # CHECKPOINTS_IN_TIME_ORDER = sub0(w2), sub1(w4), sub2(w6), ut0(w6), sub3(w8), sub4(w10), ut1(w12)

    # prefix_len=2: sub0, sub1. marks_pct series so far (own checkpoints) = [50%, 60%]
    features_2 = extract_prefix_features(subject, prefix_len=2)
    assert features_2["marks_velocity"] == pytest.approx(60.0 - 50.0)
    assert features_2["delay_acceleration"] == pytest.approx(2.0 - 0.0)  # sub1 late=2, sub0 late=0

    # prefix_len=4: sub0, sub1, sub2, ut0 -> marks series = [50, 60, 80, 50] (ut0 own pct = 50%)
    features_4 = extract_prefix_features(subject, prefix_len=4)
    assert features_4["marks_velocity"] == pytest.approx(50.0 - 80.0)
    # delay_acceleration only concerns submissions: last two submissions so far are sub1 (late 2), sub2 (late 2)
    assert features_4["delay_acceleration"] == pytest.approx(2.0 - 2.0)

    # Attendance velocity at prefix_len=2: cumulative attendance through week2 vs week... (two-point slope)
    # attendance_pct at step1 (through week2) = mean(attendance weeks 1-2) = mean(81,82)=81.5
    # attendance_pct at step2 (through week4, since sub1 is week4) = mean(weeks1-4)=mean(81,82,83,84)=82.5
    assert features_2["attendance_velocity"] == pytest.approx(82.5 - 81.5)


def test_internal_and_assignment_marks_pct_use_only_observed_components():
    dataset = _tiny_dataset(num_students=3)
    subject = dataset[0]["subjects"][0]
    features = extract_prefix_features(subject, prefix_len=2)
    # prefix_len=2 only covers sub0, sub1 -- no unit test has happened yet.
    assert features["internal_marks_pct"] == pytest.approx(0.0)
    sub_score = sum(s["score"] for s in subject["submissions"][:2])
    sub_max = sum(s["max_marks"] for s in subject["submissions"][:2])
    assert features["assignment_marks_pct"] == pytest.approx(sub_score / sub_max * 100)


def test_on_time_ratio_and_avg_days_late():
    dataset = _tiny_dataset(num_students=3)
    subject = dataset[0]["subjects"][0]
    features = extract_prefix_features(subject, prefix_len=3)
    subs = subject["submissions"][:3]
    expected_on_time = sum(1 for s in subs if s["submitted_day"] <= s["due_day"]) / len(subs)
    expected_avg_late = sum(max(0, s["submitted_day"] - s["due_day"]) for s in subs) / len(subs)
    assert features["on_time_submission_ratio"] == pytest.approx(expected_on_time)
    assert features["avg_days_late"] == pytest.approx(expected_avg_late)


def test_rejects_invalid_prefix_len():
    dataset = _tiny_dataset(num_students=1)
    subject = dataset[0]["subjects"][0]
    with pytest.raises(ValueError):
        extract_prefix_features(subject, prefix_len=1)
    with pytest.raises(ValueError):
        extract_prefix_features(subject, prefix_len=8)
