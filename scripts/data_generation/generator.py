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
