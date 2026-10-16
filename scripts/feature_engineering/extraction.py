"""No-leakage prefix-row feature extraction for Rebound AI's ScholarTrack training data.

For each student-subject record, builds one training row per "as-of checkpoint"
prefix (checkpoints 2 through 7 of the 7 in `CHECKPOINTS_IN_TIME_ORDER` -- checkpoint
1 alone is too little signal). Every feature for a given prefix is computed strictly
from data available as of that checkpoint: earlier checkpoints' unit tests,
submissions, and attendance weeks only, and the attendance window is bounded by the
prefix's own last checkpoint week. ETE is never read here -- only the final `outcome`
dict's `at_risk`/`is_failing` fields are used, and only as the row's label/metadata,
never as a feature input.
"""

from typing import Dict, List

from scripts.data_generation.generator import CHECKPOINTS_IN_TIME_ORDER

MIN_PREFIX_LEN = 2
MAX_PREFIX_LEN = len(CHECKPOINTS_IN_TIME_ORDER)  # 7

BASE_FEATURE_NAMES = [
    "internal_marks_pct",
    "assignment_marks_pct",
    "attendance_pct",
    "on_time_submission_ratio",
    "avg_days_late",
]

ENGINEERED_FEATURE_NAMES = [
    "attendance_velocity",
    "marks_velocity",
    "delay_acceleration",
    "attendance_deficit_margin",
    "marks_deficit_margin",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES

ATTENDANCE_RISK_THRESHOLD = 75.0
RISK_MARKS_THRESHOLD = 70.0  # the "at risk" aggregate cutoff -- marks_deficit_margin is anchored here, not 40.0


def _checkpoint_week(subject: Dict, kind: str, idx: int) -> int:
    """The calendar week a given (kind, idx) checkpoint falls in, read from the
    subject's own records (never from the module-level SUBMISSION_WEEKS/UNIT_TEST_WEEKS
    constants) so this keeps working on a truncated/mid-term copy of the record.
    """
    if kind == "unit_test":
        return subject["unit_tests"][idx]["week"]
    return subject["submissions"][idx]["due_day"] // 7


def _own_pct(subject: Dict, kind: str, idx: int) -> float:
    """A checkpoint's own score as a percentage of its own max marks (not cumulative)."""
    record = subject["unit_tests"][idx] if kind == "unit_test" else subject["submissions"][idx]
    if record["max_marks"] <= 0:
        return 0.0
    return record["score"] / record["max_marks"] * 100.0


def _cumulative_attendance_pct(subject: Dict, through_week: int) -> float:
    weeks = [w for w in subject["attendance_weeks"] if w["week"] <= through_week]
    if not weeks:
        return 0.0
    return sum(w["attendance_pct"] for w in weeks) / len(weeks)


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    """Least-squares slope of ys against xs. For exactly two points this reduces to
    the simple two-point delta, which is the only case that occurs at the first
    valid prefix (checkpoint 2, which has exactly 2 observed points).
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def extract_prefix_features(subject: Dict, prefix_len: int) -> Dict[str, float]:
    """Computes the 10-feature vector for one student-subject as of the given prefix length.

    `prefix_len` counts checkpoints in `CHECKPOINTS_IN_TIME_ORDER` (1-indexed count,
    i.e. prefix_len=3 means "the first 3 checkpoints have happened"). Only valid for
    MIN_PREFIX_LEN..MAX_PREFIX_LEN (2..7) -- checkpoint 1 alone is too little signal,
    and there are only 7 checkpoints total.

    Reads only `subject["unit_tests"]`, `subject["submissions"]`, and
    `subject["attendance_weeks"]` entries that fall within the prefix; never reads
    `subject["outcome"]` (the ETE-derived label lives there and must never leak into
    a feature). This function is deliberately index-bounded rather than length-aware,
    so it produces identical output whether or not the source record contains data
    for checkpoints beyond `prefix_len` -- see tests/feature_engineering/test_extraction.py
    for the leakage-freedom tests this property is designed to satisfy.
    """
    if not (MIN_PREFIX_LEN <= prefix_len <= MAX_PREFIX_LEN):
        raise ValueError(f"prefix_len must be between {MIN_PREFIX_LEN} and {MAX_PREFIX_LEN}, got {prefix_len}")

    prefix_checkpoints = CHECKPOINTS_IN_TIME_ORDER[:prefix_len]
    ut_indices = sorted(idx for kind, idx in prefix_checkpoints if kind == "unit_test")
    sub_indices = sorted(idx for kind, idx in prefix_checkpoints if kind == "submission")

    unit_tests_so_far = [subject["unit_tests"][i] for i in ut_indices]
    submissions_so_far = [subject["submissions"][i] for i in sub_indices]

    th_so_far = sum(ut["score"] for ut in unit_tests_so_far)
    max_th_so_far = sum(ut["max_marks"] for ut in unit_tests_so_far)
    pr_tw_so_far = sum(s["score"] for s in submissions_so_far)
    max_pr_tw_so_far = sum(s["max_marks"] for s in submissions_so_far)

    internal_marks_pct = (th_so_far / max_th_so_far * 100.0) if max_th_so_far > 0 else 0.0
    assignment_marks_pct = (pr_tw_so_far / max_pr_tw_so_far * 100.0) if max_pr_tw_so_far > 0 else 0.0

    last_week = max(_checkpoint_week(subject, kind, idx) for kind, idx in prefix_checkpoints)
    attendance_pct = _cumulative_attendance_pct(subject, last_week)

    if submissions_so_far:
        on_time_submission_ratio = sum(
            1 for s in submissions_so_far if s["submitted_day"] <= s["due_day"]
        ) / len(submissions_so_far)
        avg_days_late = sum(
            max(0, s["submitted_day"] - s["due_day"]) for s in submissions_so_far
        ) / len(submissions_so_far)
    else:
        on_time_submission_ratio = 0.0
        avg_days_late = 0.0

    # --- Engineered velocity/kinematic features ---

    # marks_velocity: delta between the latest and preceding checkpoint's own pct score
    # (own-marks-event percentage, not the cumulative aggregate).
    last_two = prefix_checkpoints[-2:]
    marks_series = [_own_pct(subject, kind, idx) for kind, idx in last_two]
    marks_velocity = marks_series[-1] - marks_series[0] if len(marks_series) == 2 else 0.0

    # delay_acceleration: change in lateness between the two most recent submissions
    # observed so far. Every valid prefix (>=2) includes at least 2 submissions
    # (checkpoints 1 and 2 in time order are both submissions), so this is always defined.
    if len(submissions_so_far) >= 2:
        last_two_subs = submissions_so_far[-2:]
        latenesses = [max(0, s["submitted_day"] - s["due_day"]) for s in last_two_subs]
        delay_acceleration = latenesses[-1] - latenesses[0]
    else:
        delay_acceleration = 0.0

    # attendance_velocity: slope of the cumulative attendance_pct trajectory across
    # every prefix step observed so far (step j's value = attendance_pct computed as
    # of checkpoint j's week), regressed against step index. For prefix_len=2 this is
    # exactly 2 points, which reduces to a simple two-point delta.
    attendance_trajectory = []
    for step in range(1, prefix_len + 1):
        step_checkpoints = CHECKPOINTS_IN_TIME_ORDER[:step]
        step_week = max(_checkpoint_week(subject, kind, idx) for kind, idx in step_checkpoints)
        attendance_trajectory.append(_cumulative_attendance_pct(subject, step_week))
    attendance_velocity = _linear_slope(list(range(len(attendance_trajectory))), attendance_trajectory)

    attendance_deficit_margin = attendance_pct - ATTENDANCE_RISK_THRESHOLD

    aggregate_denominator = max_th_so_far + max_pr_tw_so_far
    aggregate_pct_so_far = (
        (th_so_far + pr_tw_so_far) / aggregate_denominator * 100.0 if aggregate_denominator > 0 else 0.0
    )
    marks_deficit_margin = aggregate_pct_so_far - RISK_MARKS_THRESHOLD

    return {
        "internal_marks_pct": internal_marks_pct,
        "assignment_marks_pct": assignment_marks_pct,
        "attendance_pct": attendance_pct,
        "on_time_submission_ratio": on_time_submission_ratio,
        "avg_days_late": avg_days_late,
        "attendance_velocity": attendance_velocity,
        "marks_velocity": marks_velocity,
        "delay_acceleration": delay_acceleration,
        "attendance_deficit_margin": attendance_deficit_margin,
        "marks_deficit_margin": marks_deficit_margin,
    }


def build_training_rows(dataset: List[Dict]) -> List[Dict]:
    """Builds the full list of no-leakage training rows for a generated dataset.

    One row per (student, subject, valid prefix) -- at 120 students x 6 subjects x
    6 valid prefixes (checkpoints 2..7) this is 4,320 rows. Each row carries its
    features plus `student_id`/`subject_slug`/`checkpoint_idx` metadata and the
    subject's real `outcome.at_risk` (the training target) and `outcome.is_failing`
    (informational only, never a feature and never the target).
    """
    rows = []
    for student in dataset:
        student_id = student["student_id"]
        for subject in student["subjects"]:
            outcome = subject["outcome"]
            for prefix_len in range(MIN_PREFIX_LEN, MAX_PREFIX_LEN + 1):
                features = extract_prefix_features(subject, prefix_len)
                row = {
                    "student_id": student_id,
                    "subject_slug": subject["subject_slug"],
                    "checkpoint_idx": prefix_len,
                    **features,
                    "at_risk": outcome["at_risk"],
                    "is_failing": outcome["is_failing"],
                }
                rows.append(row)
    return rows
