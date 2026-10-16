"""Synthetic dataset generator for Rebound AI's ScholarTrack training data."""

import math
import random
from typing import Dict, List

from .subjects import (
    ELECTIVE_OPTIONS,
    FIXED_SUBJECTS,
    MDM_OPTIONS,
    OEL_OPTIONS,
    subject_slug,
)

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Ananya", "Diya", "Saanvi", "Myra", "Aadhya", "Kiara",
    "Pari", "Anika", "Navya", "Riya", "Om", "Atharv", "Yash", "Rohan", "Kabir",
    "Prisha", "Siya", "Avni", "Ira", "Zara",
]

LAST_NAMES = [
    "Sharma", "Verma", "Patil", "Deshmukh", "Kulkarni", "Joshi", "Kale",
    "Shinde", "Pawar", "Jadhav", "Gaikwad", "Naik", "Rao", "Iyer", "Nair",
    "Reddy", "Mehta", "Shah", "Gupta", "Chavan",
]


def generate_roster(num_students: int, rng: random.Random) -> List[Dict]:
    """Generates `num_students` MMCOE TY Computer Engineering student records."""
    roster = []
    for i in range(1, num_students + 1):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        roster.append(
            {
                "student_id": f"S{i:03d}",
                "name": f"{first} {last}",
                "prn": f"PRN{i:04d}",
                "email": f"{first.lower()}.{last.lower()}{i}@mmcoe.edu.in",
                "branch": "CE",
                "academic_year": "TY",
            }
        )
    return roster


def assign_subjects(rng: random.Random) -> List[Dict]:
    """Builds one student's 6-subject list: the 3 fixed subjects plus one chosen MDM/OEL/Elective."""
    chosen = [
        {"subject_slug": slug, "subject_name": meta["name"], "components": dict(meta["components"])}
        for slug, meta in FIXED_SUBJECTS.items()
    ]

    for pool in (MDM_OPTIONS, OEL_OPTIONS, ELECTIVE_OPTIONS):
        name = rng.choice(list(pool))
        meta = pool[name]
        chosen.append(
            {
                "subject_slug": subject_slug(name),
                "subject_name": name,
                "components": dict(meta["components"]),
            }
        )

    return chosen


def select_at_risk_students(
    student_ids: List[str],
    rng: random.Random,
    min_rate: float = 0.05,
    max_rate: float = 0.15,
) -> List[str]:
    """Picks a 5-15% (by default) minority of students to be genuinely falling behind."""
    n = len(student_ids)
    min_count = max(1, math.ceil(min_rate * n))
    max_count = max(min_count, math.floor(max_rate * n))
    count = rng.randint(min_count, max_count)
    return rng.sample(student_ids, count)


def declining_series(start: float, end: float, n: int, noise: float, rng: random.Random) -> List[float]:
    """A linear ramp from `start` to `end` over `n` points, with noise, clipped to [0, 100].

    Used for at-risk trajectories: the ramp spreads the decline across every
    point rather than cliff-dropping between two checkpoints.
    """
    values = []
    for idx in range(n):
        t = idx / max(1, n - 1)
        base = start + (end - start) * t
        noisy = base + rng.gauss(0, noise)
        values.append(max(0.0, min(100.0, noisy)))
    return values


def stable_series(level: float, n: int, noise: float, rng: random.Random) -> List[float]:
    """`n` points hovering around `level` with noise, clipped to [0, 100]. Used for students in good standing."""
    return [max(0.0, min(100.0, level + rng.gauss(0, noise))) for _ in range(n)]


ATTENDANCE_RISK_THRESHOLD = 75.0
AGGREGATE_RISK_THRESHOLD = 40.0
NUM_WEEKS = 14
UNIT_TEST_WEEKS = [6, 12]
SUBMISSION_WEEKS = [2, 4, 6, 8, 10]

# The 7 assessment checkpoints in chronological order: ("submission", i) or ("unit_test", i).
# On a tie in week, submissions come before unit tests (so week 6 submission precedes week 6 UT1).
_TYPE_RANK = {"submission": 0, "unit_test": 1}


def _checkpoint_week(checkpoint):
    kind, idx = checkpoint
    return SUBMISSION_WEEKS[idx] if kind == "submission" else UNIT_TEST_WEEKS[idx]


CHECKPOINTS_IN_TIME_ORDER = sorted(
    [("submission", i) for i in range(len(SUBMISSION_WEEKS))] + [("unit_test", i) for i in range(2)],
    key=lambda cp: (_checkpoint_week(cp), _TYPE_RANK[cp[0]]),
)


def generate_subject_instance(subject: Dict, at_risk: bool, rng: random.Random) -> Dict:
    """Generates one student's full-term record for one subject, and computes its risk label.

    ETE is only ever used below to compute the label -- it is never part of the
    per-checkpoint feature trajectory a forecasting model would train on.
    """
    components = subject["components"]
    pr_tw_key = "PR" if "PR" in components else "TW"
    num_marks_points = 2 + len(SUBMISSION_WEEKS)

    if at_risk:
        marks_pct_series = declining_series(rng.uniform(65, 85), rng.uniform(20, 38), num_marks_points, 4.0, rng)
        attendance_series = declining_series(rng.uniform(80, 95), rng.uniform(60, 74), NUM_WEEKS, 2.5, rng)
        lateness_series = declining_series(rng.uniform(0, 1), rng.uniform(3, 7), len(SUBMISSION_WEEKS), 1.0, rng)
    else:
        if rng.random() < 0.35:
            # Realistic moderation bump: a chunk of "fine" students cluster just above the pass cutoff.
            level = rng.uniform(AGGREGATE_RISK_THRESHOLD, AGGREGATE_RISK_THRESHOLD + 10)
        else:
            level = rng.uniform(AGGREGATE_RISK_THRESHOLD + 10, 95)
        marks_pct_series = stable_series(level, num_marks_points, 4.0, rng)
        attendance_series = stable_series(rng.uniform(78, 97), NUM_WEEKS, 2.0, rng)
        lateness_series = stable_series(rng.uniform(0, 1.5), len(SUBMISSION_WEEKS), 0.5, rng)

    # Assign the declining series in chronological order so the trend is monotone over time.
    pct_by_checkpoint = {cp: marks_pct_series[k] for k, cp in enumerate(CHECKPOINTS_IN_TIME_ORDER)}
    unit_test_pcts = [pct_by_checkpoint[("unit_test", i)] for i in range(2)]
    submission_pcts = [pct_by_checkpoint[("submission", i)] for i in range(len(SUBMISSION_WEEKS))]

    unit_tests = [
        {"index": i + 1, "week": UNIT_TEST_WEEKS[i], "max_marks": 20.0, "score": round(unit_test_pcts[i] / 100 * 20, 1)}
        for i in range(2)
    ]

    submission_max = round(25 / len(SUBMISSION_WEEKS), 2)
    submissions = [
        {
            "index": i + 1,
            "due_day": SUBMISSION_WEEKS[i] * 7,
            "submitted_day": SUBMISSION_WEEKS[i] * 7 + max(0, round(lateness_series[i])),
            "max_marks": submission_max,
            "score": round(submission_pcts[i] / 100 * submission_max, 2),
        }
        for i in range(len(SUBMISSION_WEEKS))
    ]

    attendance_weeks = [{"week": i + 1, "attendance_pct": round(attendance_series[i], 1)} for i in range(NUM_WEEKS)]

    th_score = round(sum(ut["score"] for ut in unit_tests), 1)
    pr_tw_score = round(sum(s["score"] for s in submissions), 2)
    final_attendance_pct = round(
        sum(w["attendance_pct"] for w in attendance_weeks) / len(attendance_weeks), 1
    )

    if at_risk:
        ete_pct = rng.uniform(15, 38)
    else:
        ete_pct = max(0.0, min(100.0, marks_pct_series[-1] + rng.gauss(0, 5)))
    ete_score = round(ete_pct / 100 * 60, 1)

    aggregate_pct = round((th_score + ete_score + pr_tw_score) / 125 * 100, 1)
    computed_at_risk = (final_attendance_pct < ATTENDANCE_RISK_THRESHOLD) or (aggregate_pct < AGGREGATE_RISK_THRESHOLD)

    return {
        "subject_slug": subject["subject_slug"],
        "subject_name": subject["subject_name"],
        "components": components,
        "unit_tests": unit_tests,
        "submissions": submissions,
        "attendance_weeks": attendance_weeks,
        "pr_tw_component": pr_tw_key,
        "th_score": th_score,
        "pr_tw_score": pr_tw_score,
        "outcome": {
            "ete_score": ete_score,
            "aggregate_pct": aggregate_pct,
            "final_attendance_pct": final_attendance_pct,
            "at_risk": computed_at_risk,
        },
    }


def generate_subject_instance_with_target_label(
    subject: Dict,
    target_at_risk: bool,
    rng: random.Random,
    max_attempts: int = 20,
) -> Dict:
    """Retries `generate_subject_instance` until its computed label matches the intended one.

    The label is always computed from the generated numbers (never hand-set),
    so this just rejects the rare draw that lands on the wrong side of a
    threshold rather than overriding it.
    """
    for _ in range(max_attempts):
        instance = generate_subject_instance(subject, at_risk=target_at_risk, rng=rng)
        if instance["outcome"]["at_risk"] == target_at_risk:
            return instance
    raise RuntimeError(
        f"Could not generate a '{subject['subject_slug']}' instance with at_risk={target_at_risk} "
        f"after {max_attempts} attempts"
    )


def generate_dataset(num_students: int = 75, seed: int = 42) -> List[Dict]:
    """Generates the full synthetic TY Computer Engineering cohort, deterministic for a given seed."""
    rng = random.Random(seed)

    roster = generate_roster(num_students, rng)
    student_ids = [s["student_id"] for s in roster]
    at_risk_ids = set(select_at_risk_students(student_ids, rng))

    dataset = []
    for student in roster:
        subject_specs = assign_subjects(rng)
        is_at_risk_student = student["student_id"] in at_risk_ids
        flagged_index = rng.randrange(len(subject_specs)) if is_at_risk_student else None

        subjects = []
        for idx, subject in enumerate(subject_specs):
            target_at_risk = is_at_risk_student and idx == flagged_index
            subjects.append(generate_subject_instance_with_target_label(subject, target_at_risk, rng))

        dataset.append({**student, "subjects": subjects})

    return dataset
