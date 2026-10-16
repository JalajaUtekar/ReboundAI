"""Tests for scripts/training/cnn_model.py -- the CNN sequence model."""

import copy
import json
import math
import os

import numpy as np
import pytest

from scripts.training.cnn_model import (
    MAX_PREFIX_LEN,
    MIN_PREFIX_LEN,
    NUM_FEATURES,
    SEQ_LEN,
    binary_focal_loss,
    build_model,
    build_sequence,
    build_sequences,
    train_and_predict_oof,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_DATASET_PATH = os.path.join(REPO_ROOT, "data", "synthetic_students.json")


def _make_subject(subject_slug="demo-subj", at_risk=False, num_submissions=5, num_unit_tests=2):
    """Hand-constructed fake subject record matching the generator's shape."""
    submission_weeks = [2, 4, 6, 8, 10][:num_submissions]
    unit_test_weeks = [6, 12][:num_unit_tests]

    submissions = [
        {
            "index": i + 1,
            "due_day": w * 7,
            "submitted_day": w * 7 + (0 if i % 2 == 0 else 2),
            "max_marks": 5.0,
            "score": 4.0 - i * 0.2,
        }
        for i, w in enumerate(submission_weeks)
    ]
    unit_tests = [
        {"index": i + 1, "week": w, "max_marks": 20.0, "score": 15.0 - i * 2.0}
        for i, w in enumerate(unit_test_weeks)
    ]
    attendance_weeks = [
        {"week": wk, "attendance_pct": 90.0 - wk * 0.5} for wk in range(1, 15)
    ]

    return {
        "subject_slug": subject_slug,
        "subject_name": "Demo Subject",
        "components": {"TH": 20, "PR": 25, "ETE": 60},
        "unit_tests": unit_tests,
        "submissions": submissions,
        "attendance_weeks": attendance_weeks,
        "pr_tw_component": "PR",
        "th_score": sum(u["score"] for u in unit_tests),
        "pr_tw_score": sum(s["score"] for s in submissions),
        "outcome": {
            "ete_score": 10.0 if at_risk else 45.0,
            "aggregate_pct": 35.0 if at_risk else 80.0,
            "final_attendance_pct": 70.0 if at_risk else 90.0,
            "at_risk": at_risk,
            "is_failing": False,
        },
    }


def _make_student(student_id, subjects):
    return {
        "student_id": student_id,
        "name": "Test Student",
        "prn": "PRN0001",
        "email": "test@example.com",
        "branch": "CE",
        "academic_year": "TY",
        "subjects": subjects,
    }


def _make_fake_dataset(num_students=3):
    dataset = []
    for i in range(num_students):
        sid = f"S{i:03d}"
        at_risk = i == 0  # first student's subject is at-risk, rest are not
        subj = _make_subject(subject_slug="demo-subj", at_risk=at_risk)
        dataset.append(_make_student(sid, [subj]))
    return dataset


# ---------------------------------------------------------------------------
# build_sequence / build_sequences shape + padding tests
# ---------------------------------------------------------------------------


def test_build_sequence_shape():
    subj = _make_subject()
    seq = build_sequence(subj, checkpoint_idx=3)
    assert seq.shape == (SEQ_LEN, NUM_FEATURES)
    assert seq.dtype == np.float32


def test_build_sequence_zero_padding_for_short_prefix():
    subj = _make_subject()
    seq = build_sequence(subj, checkpoint_idx=2)
    # Only first 2 rows should be nonzero (real), rest zero-padded.
    assert not np.allclose(seq[0], 0.0)
    assert not np.allclose(seq[1], 0.0)
    for row in seq[2:]:
        assert np.allclose(row, 0.0)


def test_build_sequence_full_prefix_no_padding_but_drops_oldest():
    """checkpoint_idx=7 means all 7 checkpoints have occurred, but SEQ_LEN=6,
    so by design we keep the 6 MOST RECENT (drop checkpoint index 0)."""
    subj = _make_subject()
    seq = build_sequence(subj, checkpoint_idx=7)
    assert seq.shape == (SEQ_LEN, NUM_FEATURES)
    for row in seq:
        assert not np.allclose(row, 0.0)


def test_build_sequence_invalid_checkpoint_idx_raises():
    subj = _make_subject()
    with pytest.raises(ValueError):
        build_sequence(subj, checkpoint_idx=1)
    with pytest.raises(ValueError):
        build_sequence(subj, checkpoint_idx=8)


def test_build_sequences_shapes_and_metadata():
    dataset = _make_fake_dataset(num_students=3)
    built = build_sequences(dataset)
    num_prefixes = MAX_PREFIX_LEN - MIN_PREFIX_LEN + 1  # 6
    expected_n = 3 * 1 * num_prefixes  # 3 students x 1 subject x 6 prefixes
    assert built["X"].shape == (expected_n, SEQ_LEN, NUM_FEATURES)
    assert built["y"].shape == (expected_n,)
    assert len(built["student_id"]) == expected_n
    assert len(built["subject_slug"]) == expected_n
    assert len(built["checkpoint_idx"]) == expected_n
    # y values are 0./1. floats matching at_risk
    assert set(np.unique(built["y"]).tolist()) <= {0.0, 1.0}
    # student S000's subject was built at_risk=True
    s000_mask = [sid == "S000" for sid in built["student_id"]]
    assert all(y == 1.0 for y, m in zip(built["y"], s000_mask) if m)
    s001_mask = [sid == "S001" for sid in built["student_id"]]
    assert all(y == 0.0 for y, m in zip(built["y"], s001_mask) if m)


# ---------------------------------------------------------------------------
# No-leakage test: truncating the source record must not change the sequence
# for a given checkpoint_idx.
# ---------------------------------------------------------------------------


def _truncate_subject(subject, checkpoint_idx):
    """Returns a copy of `subject` with only the unit_tests/submissions/attendance
    data available as of `checkpoint_idx` checkpoints elapsed -- simulating a
    mid-term snapshot where later checkpoints haven't happened yet."""
    from scripts.data_generation.generator import CHECKPOINTS_IN_TIME_ORDER

    prefix = CHECKPOINTS_IN_TIME_ORDER[:checkpoint_idx]
    ut_indices = sorted(idx for kind, idx in prefix if kind == "unit_test")
    sub_indices = sorted(idx for kind, idx in prefix if kind == "submission")
    last_week = max(
        (subject["unit_tests"][i]["week"] if kind == "unit_test" else subject["submissions"][i]["due_day"] // 7)
        for kind, i in prefix
    )

    truncated = copy.deepcopy(subject)
    truncated["unit_tests"] = [truncated["unit_tests"][i] for i in ut_indices]
    truncated["submissions"] = [truncated["submissions"][i] for i in sub_indices]
    truncated["attendance_weeks"] = [w for w in truncated["attendance_weeks"] if w["week"] <= last_week]
    # Blow away outcome to make sure build_sequence never reads it anyway.
    truncated["outcome"] = {"at_risk": None, "is_failing": None}
    return truncated


def test_no_leakage_truncated_record_matches():
    subj_full = _make_subject(at_risk=True)
    for checkpoint_idx in range(MIN_PREFIX_LEN, MAX_PREFIX_LEN + 1):
        seq_full = build_sequence(subj_full, checkpoint_idx)
        subj_truncated = _truncate_subject(subj_full, checkpoint_idx)
        seq_truncated = build_sequence(subj_truncated, checkpoint_idx)
        assert np.allclose(seq_full, seq_truncated), (
            f"Leakage detected at checkpoint_idx={checkpoint_idx}: sequence differs "
            "depending on whether later checkpoints exist in the source record."
        )


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------


def test_focal_loss_matches_hand_computed_value():
    alpha, gamma = 0.25, 2.0
    loss_fn = binary_focal_loss(alpha=alpha, gamma=gamma)

    # Single example: y_true=1, y_pred=0.8 -> p_t=0.8, alpha_t=alpha
    y_true = np.array([1.0], dtype=np.float32)
    y_pred = np.array([0.8], dtype=np.float32)
    expected = -alpha * (1 - 0.8) ** gamma * math.log(0.8)
    got = float(loss_fn(y_true, y_pred).numpy())
    assert math.isclose(got, expected, rel_tol=1e-4)

    # Single example: y_true=0, y_pred=0.3 -> p_t=1-0.3=0.7, alpha_t=1-alpha
    y_true2 = np.array([0.0], dtype=np.float32)
    y_pred2 = np.array([0.3], dtype=np.float32)
    expected2 = -(1 - alpha) * (1 - 0.7) ** gamma * math.log(0.7)
    got2 = float(loss_fn(y_true2, y_pred2).numpy())
    assert math.isclose(got2, expected2, rel_tol=1e-4)

    # Batch of the two above should give the mean.
    y_true_b = np.array([1.0, 0.0], dtype=np.float32)
    y_pred_b = np.array([0.8, 0.3], dtype=np.float32)
    expected_mean = (expected + expected2) / 2.0
    got_mean = float(loss_fn(y_true_b, y_pred_b).numpy())
    assert math.isclose(got_mean, expected_mean, rel_tol=1e-4)


def test_focal_loss_well_classified_lower_than_poorly_classified():
    loss_fn = binary_focal_loss()
    y_true = np.array([1.0], dtype=np.float32)
    confident_correct = loss_fn(y_true, np.array([0.95], dtype=np.float32)).numpy()
    confident_wrong = loss_fn(y_true, np.array([0.05], dtype=np.float32)).numpy()
    assert confident_correct < confident_wrong


# ---------------------------------------------------------------------------
# build_model sanity
# ---------------------------------------------------------------------------


def test_build_model_output_shape():
    model = build_model()
    x = np.zeros((4, SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    out = model(x, training=False).numpy()
    assert out.shape == (4, 1)
    assert np.all((out >= 0.0) & (out <= 1.0))


# ---------------------------------------------------------------------------
# End-to-end train_and_predict_oof on a tiny fake dataset
# ---------------------------------------------------------------------------


def _make_bigger_fake_dataset():
    """A few more students so each fold has both classes to train/predict on."""
    dataset = []
    for i in range(8):
        sid = f"S{i:03d}"
        at_risk = i < 3  # 3 at-risk, 5 not
        subj = _make_subject(subject_slug="demo-subj", at_risk=at_risk)
        dataset.append(_make_student(sid, [subj]))
    return dataset


@pytest.mark.parametrize("loss", ["weighted_bce", "focal"])
def test_train_and_predict_oof_end_to_end(loss):
    dataset = _make_bigger_fake_dataset()
    student_ids = [s["student_id"] for s in dataset]
    # Simple 2-fold split by hand (mirrors cv.grouped_student_kfold's output shape).
    fold1 = (student_ids[:4], student_ids[4:])
    fold2 = (student_ids[4:], student_ids[:4])
    folds = [fold1, fold2]

    result = train_and_predict_oof(dataset, folds, loss=loss, epochs=2, batch_size=8)

    num_prefixes = MAX_PREFIX_LEN - MIN_PREFIX_LEN + 1  # 6
    expected_n = len(dataset) * 1 * num_prefixes  # every student appears in exactly one val fold
    assert len(result["student_id"]) == expected_n
    assert len(result["subject_slug"]) == expected_n
    assert len(result["checkpoint_idx"]) == expected_n
    assert result["y_true"].shape == (expected_n,)
    assert result["y_proba"].shape == (expected_n,)
    assert np.all((result["y_proba"] >= 0.0) & (result["y_proba"] <= 1.0))
    assert set(np.unique(result["y_true"]).tolist()) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(REAL_DATASET_PATH), reason="data/synthetic_students.json not found")
def test_real_data_extraction_row_count_and_label_correctness():
    with open(REAL_DATASET_PATH) as f:
        dataset = json.load(f)

    built = build_sequences(dataset)
    num_students = len(dataset)
    num_subjects = len(dataset[0]["subjects"])
    num_prefixes = MAX_PREFIX_LEN - MIN_PREFIX_LEN + 1  # 6
    expected_n = num_students * num_subjects * num_prefixes
    assert built["X"].shape == (expected_n, SEQ_LEN, NUM_FEATURES)
    assert len(built["y"]) == expected_n

    # Spot-check that y matches outcome.at_risk for a sample of rows, by
    # reconstructing the (student_id, subject_slug) -> at_risk lookup.
    at_risk_lookup = {}
    for student in dataset:
        for subject in student["subjects"]:
            at_risk_lookup[(student["student_id"], subject["subject_slug"])] = subject["outcome"]["at_risk"]

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(expected_n, size=min(200, expected_n), replace=False)
    for i in sample_idx:
        key = (built["student_id"][i], built["subject_slug"][i])
        expected_label = 1.0 if at_risk_lookup[key] else 0.0
        assert built["y"][i] == expected_label
