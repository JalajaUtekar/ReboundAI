"""Synthetic dataset generator for Rebound AI's ScholarTrack training data."""

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
