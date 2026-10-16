# Synthetic Dataset Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a realistic, reproducible 75-student TY Computer Engineering dataset (6 subjects per student) with a 5-15% genuinely at-risk minority, which every later Rebound AI subsystem (ScholarTrack training, the Supabase seed data, the frontend demo) builds on.

**Architecture:** A dependency-free Python package (`scripts/data_generation/`) with pure-stdlib generation logic (no numpy/pandas needed for this stage), driven by a seeded `random.Random` for reproducibility, producing one JSON file. A thin CLI script wraps it for one-command regeneration.

**Tech Stack:** Python 3.11+ stdlib only (`random`, `json`, `math`), `pytest` for tests.

**Spec:** [docs/superpowers/specs/2026-10-16-rebound-ai-design.md](../specs/2026-10-16-rebound-ai-design.md) — Sections 2 (registration/subjects), 3 (grading schema), 4 (synthetic dataset).

## Global Constraints

- 75 students total, branch fixed to `"CE"`, academic_year fixed to `"TY"` (Spec §2).
- Every student has exactly 6 subjects: the 3 fixed (`DSML`, `CN`, `SPOS`) + 1 chosen MDM + 1 chosen OEL + 1 chosen Elective (Spec §2, §3).
- Marks components per subject: `TH(40) + ETE(60) + PR(25)` for DSML/SPOS/MDM; `TH(40) + ETE(60) + TW(25)` for CN/OEL/Elective (Spec §3).
- At-risk label rule: attendance ends below 75% **OR** weighted aggregate (TH+ETE+PR/TW, normalized to 100) ends below 40% (Spec §4). ETE is only ever used to compute this label — it must never appear as an input feature for later modeling.
- 5-15% of the 75 students are designated at-risk overall (Spec §4) — one of their 6 subjects carries the at-risk trajectory, the rest of their subjects and all of a non-at-risk student's subjects use the majority generator.
- At-risk trajectories must be a gradual multi-checkpoint decline, never a single-step cliff (Spec §4).
- Majority-population marks must show a realistic cluster just above the 40% pass threshold, not a smooth/uniform spread (Spec §4).

---

### Task 1: Subject catalog

**Files:**
- Create: `conftest.py` (repo root)
- Create: `scripts/data_generation/__init__.py`
- Create: `scripts/data_generation/subjects.py`
- Test: `tests/data_generation/__init__.py`
- Test: `tests/data_generation/test_subjects.py`

**Interfaces:**
- Produces: `FIXED_SUBJECTS: dict[str, dict]` (keys `"DSML"`, `"CN"`, `"SPOS"`, each `{"name": str, "components": dict[str, int]}`), `MDM_OPTIONS`, `OEL_OPTIONS`, `ELECTIVE_OPTIONS` (same shape, keyed by subject name), `subject_slug(name: str) -> str`.

- [ ] **Step 0: Make the repo root importable for tests**

Create an empty `conftest.py` at the repo root (`D:\GenHack v1.0\conftest.py`):

```python
# Empty on purpose. Its presence makes pytest insert the repo root onto
# sys.path during collection, so test modules under tests/ (which has its
# own __init__.py chain) can `import scripts....` regardless of cwd.
```

- [ ] **Step 1: Write the failing test**

Create `tests/data_generation/__init__.py` (empty file) and `tests/data_generation/test_subjects.py`:

```python
from scripts.data_generation.subjects import (
    FIXED_SUBJECTS,
    MDM_OPTIONS,
    OEL_OPTIONS,
    ELECTIVE_OPTIONS,
    subject_slug,
)


def test_fixed_subjects_are_dsml_cn_spos():
    assert set(FIXED_SUBJECTS.keys()) == {"DSML", "CN", "SPOS"}


def test_fixed_subject_components_sum_to_125():
    for slug, meta in FIXED_SUBJECTS.items():
        assert sum(meta["components"].values()) == 125, slug


def test_dsml_and_spos_use_practical_not_termwork():
    assert "PR" in FIXED_SUBJECTS["DSML"]["components"]
    assert "PR" in FIXED_SUBJECTS["SPOS"]["components"]


def test_cn_uses_termwork_not_practical():
    assert "TW" in FIXED_SUBJECTS["CN"]["components"]
    assert "PR" not in FIXED_SUBJECTS["CN"]["components"]


def test_option_pool_sizes_match_spec():
    assert len(MDM_OPTIONS) == 6
    assert len(OEL_OPTIONS) == 6
    assert len(ELECTIVE_OPTIONS) == 4


def test_mdm_options_use_practical_component():
    for name, meta in MDM_OPTIONS.items():
        assert "PR" in meta["components"], name
        assert sum(meta["components"].values()) == 125, name


def test_oel_and_elective_options_use_termwork_component():
    for pool in (OEL_OPTIONS, ELECTIVE_OPTIONS):
        for name, meta in pool.items():
            assert "TW" in meta["components"], name
            assert sum(meta["components"].values()) == 125, name


def test_subject_slug_normalizes_name():
    assert subject_slug("Data Science and Machine Learning") == "data_science_and_machine_learning"
    assert subject_slug("Electric Vehicle Technology") == "electric_vehicle_technology"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_generation/test_subjects.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.data_generation.subjects'` (or `'scripts'`).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/data_generation/__init__.py` (empty file).

Create `scripts/data_generation/subjects.py`:

```python
"""Subject catalog for Rebound AI's TY Computer Engineering cohort (MMCOE)."""

_PR_COMPONENTS = {"TH": 40, "ETE": 60, "PR": 25}
_TW_COMPONENTS = {"TH": 40, "ETE": 60, "TW": 25}

FIXED_SUBJECTS = {
    "DSML": {"name": "Data Science and Machine Learning", "components": dict(_PR_COMPONENTS)},
    "CN": {"name": "Computer Networks", "components": dict(_TW_COMPONENTS)},
    "SPOS": {"name": "System Programming and Operating Systems", "components": dict(_PR_COMPONENTS)},
}

MDM_OPTIONS = {
    "GenAI": {"components": dict(_PR_COMPONENTS)},
    "Privacy and Security Engineering": {"components": dict(_PR_COMPONENTS)},
    "Robotics and Automation": {"components": dict(_PR_COMPONENTS)},
    "Electric Vehicle Technology": {"components": dict(_PR_COMPONENTS)},
    "Sustainable and Smart Systems": {"components": dict(_PR_COMPONENTS)},
    "Embedded Systems": {"components": dict(_PR_COMPONENTS)},
}

OEL_OPTIONS = {
    "Practical Japanese Communication": {"components": dict(_TW_COMPONENTS)},
    "Workplace Ethics in German Language": {"components": dict(_TW_COMPONENTS)},
    "Product and Brand Management": {"components": dict(_TW_COMPONENTS)},
    "Strategic Management": {"components": dict(_TW_COMPONENTS)},
    "Operations and Supply Chain Management": {"components": dict(_TW_COMPONENTS)},
    "Organizational Behaviour": {"components": dict(_TW_COMPONENTS)},
}

ELECTIVE_OPTIONS = {
    "Cloud Computing": {"components": dict(_TW_COMPONENTS)},
    "Information Systems": {"components": dict(_TW_COMPONENTS)},
    "Prompt Engineering": {"components": dict(_TW_COMPONENTS)},
    "Game Development": {"components": dict(_TW_COMPONENTS)},
}


def subject_slug(name: str) -> str:
    """Converts a human-readable subject name into a stable lowercase, underscore-separated key."""
    return name.strip().lower().replace(" ", "_").replace("&", "and")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_generation/test_subjects.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add conftest.py scripts/data_generation/__init__.py scripts/data_generation/subjects.py tests/data_generation/__init__.py tests/data_generation/test_subjects.py
git commit -m "feat: add subject catalog for synthetic data generation"
```

---

### Task 2: Student roster and subject assignment

**Files:**
- Modify: `scripts/data_generation/generator.py` (create)
- Test: `tests/data_generation/test_generator.py` (create)

**Interfaces:**
- Consumes: `FIXED_SUBJECTS`, `MDM_OPTIONS`, `OEL_OPTIONS`, `ELECTIVE_OPTIONS`, `subject_slug` from `scripts.data_generation.subjects` (Task 1).
- Produces: `generate_roster(num_students: int, rng: random.Random) -> list[dict]` (each dict has keys `student_id, name, prn, email, branch, academic_year`), `assign_subjects(rng: random.Random) -> list[dict]` (each dict has keys `subject_slug, subject_name, components`, length 6, first three always DSML/CN/SPOS in that order).

- [ ] **Step 1: Write the failing test**

Create `tests/data_generation/test_generator.py`:

```python
import random

from scripts.data_generation.generator import generate_roster, assign_subjects


def test_generate_roster_returns_requested_count():
    rng = random.Random(42)
    roster = generate_roster(75, rng)
    assert len(roster) == 75


def test_generate_roster_ids_and_prns_and_emails_are_unique():
    rng = random.Random(42)
    roster = generate_roster(75, rng)
    assert len({s["student_id"] for s in roster}) == 75
    assert len({s["prn"] for s in roster}) == 75
    assert len({s["email"] for s in roster}) == 75


def test_generate_roster_fixes_branch_and_year():
    rng = random.Random(42)
    roster = generate_roster(75, rng)
    assert all(s["branch"] == "CE" for s in roster)
    assert all(s["academic_year"] == "TY" for s in roster)


def test_assign_subjects_returns_six_subjects_fixed_first():
    rng = random.Random(1)
    subjects = assign_subjects(rng)
    assert len(subjects) == 6
    assert [s["subject_slug"] for s in subjects[:3]] == ["DSML", "CN", "SPOS"]


def test_assign_subjects_choice_slots_come_from_their_pools():
    from scripts.data_generation.subjects import MDM_OPTIONS, OEL_OPTIONS, ELECTIVE_OPTIONS

    rng = random.Random(7)
    subjects = assign_subjects(rng)
    mdm_slug, oel_slug, elective_slug = (s["subject_slug"] for s in subjects[3:])

    from scripts.data_generation.subjects import subject_slug as slugify

    assert mdm_slug in {slugify(n) for n in MDM_OPTIONS}
    assert oel_slug in {slugify(n) for n in OEL_OPTIONS}
    assert elective_slug in {slugify(n) for n in ELECTIVE_OPTIONS}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.data_generation.generator'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/data_generation/generator.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/data_generation/generator.py tests/data_generation/test_generator.py
git commit -m "feat: generate student roster and 6-subject assignment"
```

---

### Task 3: At-risk student selection (5-15% minority)

**Files:**
- Modify: `scripts/data_generation/generator.py`
- Modify: `tests/data_generation/test_generator.py`

**Interfaces:**
- Produces: `select_at_risk_students(student_ids: list[str], rng: random.Random, min_rate: float = 0.05, max_rate: float = 0.15) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/data_generation/test_generator.py`:

```python
from scripts.data_generation.generator import select_at_risk_students


def test_select_at_risk_students_count_within_5_to_15_percent():
    student_ids = [f"S{i:03d}" for i in range(1, 76)]
    rng = random.Random(3)
    at_risk = select_at_risk_students(student_ids, rng)
    assert 4 <= len(at_risk) <= 11  # ceil(5% of 75) .. floor(15% of 75)


def test_select_at_risk_students_returns_subset_with_no_duplicates():
    student_ids = [f"S{i:03d}" for i in range(1, 76)]
    rng = random.Random(9)
    at_risk = select_at_risk_students(student_ids, rng)
    assert len(at_risk) == len(set(at_risk))
    assert set(at_risk).issubset(set(student_ids))


def test_select_at_risk_students_respects_custom_rate_bounds():
    student_ids = [f"S{i:03d}" for i in range(1, 101)]
    rng = random.Random(11)
    at_risk = select_at_risk_students(student_ids, rng, min_rate=0.10, max_rate=0.10)
    assert len(at_risk) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_at_risk_students'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/data_generation/generator.py` (near the top, with the other imports):

```python
import math
```

Append to `scripts/data_generation/generator.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/data_generation/generator.py tests/data_generation/test_generator.py
git commit -m "feat: select a 5-15% at-risk student minority"
```

---

### Task 4: Gradual-decline and pass-threshold-clustering series helpers

**Files:**
- Modify: `scripts/data_generation/generator.py`
- Modify: `tests/data_generation/test_generator.py`

**Interfaces:**
- Produces: `declining_series(start: float, end: float, n: int, noise: float, rng: random.Random) -> list[float]`, `stable_series(level: float, n: int, noise: float, rng: random.Random) -> list[float]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/data_generation/test_generator.py`:

```python
from scripts.data_generation.generator import declining_series, stable_series


def test_declining_series_has_correct_length_and_bounds():
    rng = random.Random(5)
    series = declining_series(start=85, end=30, n=7, noise=0.0, rng=rng)
    assert len(series) == 7
    assert all(0.0 <= v <= 100.0 for v in series)


def test_declining_series_actually_declines_overall():
    rng = random.Random(5)
    series = declining_series(start=85, end=30, n=7, noise=0.0, rng=rng)
    assert series[0] > series[-1] + 30  # a real, large drop end to end


def test_declining_series_is_gradual_not_a_cliff():
    # With zero noise the ramp is linear, so no single step should account
    # for most of the total drop -- the decline must be spread across steps.
    rng = random.Random(5)
    series = declining_series(start=90, end=20, n=8, noise=0.0, rng=rng)
    total_drop = series[0] - series[-1]
    biggest_single_step = max(series[i] - series[i + 1] for i in range(len(series) - 1))
    assert biggest_single_step < total_drop * 0.5


def test_stable_series_stays_near_level():
    rng = random.Random(2)
    series = stable_series(level=82.0, n=10, noise=0.0, rng=rng)
    assert all(v == 82.0 for v in series)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: FAIL with `ImportError: cannot import name 'declining_series'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/data_generation/generator.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/data_generation/generator.py tests/data_generation/test_generator.py
git commit -m "feat: add gradual-decline and stable trajectory helpers"
```

---

### Task 5: Per-subject instance generation with no-leakage label computation

**Files:**
- Modify: `scripts/data_generation/generator.py`
- Modify: `tests/data_generation/test_generator.py`

**Interfaces:**
- Consumes: `declining_series`, `stable_series` (Task 4).
- Produces: `generate_subject_instance(subject: dict, at_risk: bool, rng: random.Random) -> dict` (keys: `subject_slug, subject_name, components, unit_tests, submissions, attendance_weeks, pr_tw_component, th_score, pr_tw_score, ete_score, aggregate_pct, final_attendance_pct, at_risk`), `generate_subject_instance_with_target_label(subject: dict, target_at_risk: bool, rng: random.Random, max_attempts: int = 20) -> dict`.

- [ ] **Step 1: Write the failing test**

Append to `tests/data_generation/test_generator.py`:

```python
from scripts.data_generation.generator import (
    generate_subject_instance,
    generate_subject_instance_with_target_label,
)

_SAMPLE_SUBJECT = {
    "subject_slug": "DSML",
    "subject_name": "Data Science and Machine Learning",
    "components": {"TH": 40, "ETE": 60, "PR": 25},
}


def test_generate_subject_instance_has_two_unit_tests_and_five_submissions():
    rng = random.Random(4)
    instance = generate_subject_instance(_SAMPLE_SUBJECT, at_risk=False, rng=rng)
    assert len(instance["unit_tests"]) == 2
    assert len(instance["submissions"]) == 5
    assert len(instance["attendance_weeks"]) == 14


def test_generate_subject_instance_component_scores_are_within_max_marks():
    rng = random.Random(4)
    instance = generate_subject_instance(_SAMPLE_SUBJECT, at_risk=False, rng=rng)
    assert 0 <= instance["th_score"] <= 40
    assert 0 <= instance["pr_tw_score"] <= 25
    assert 0 <= instance["ete_score"] <= 60


def test_generate_subject_instance_at_risk_label_matches_computed_thresholds():
    rng = random.Random(4)
    instance = generate_subject_instance(_SAMPLE_SUBJECT, at_risk=True, rng=rng)
    expected = (instance["final_attendance_pct"] < 75.0) or (instance["aggregate_pct"] < 40.0)
    assert instance["at_risk"] == expected


def test_generate_subject_instance_with_target_label_guarantees_at_risk_true():
    rng = random.Random(123)
    for _ in range(10):
        instance = generate_subject_instance_with_target_label(_SAMPLE_SUBJECT, True, rng)
        assert instance["at_risk"] is True


def test_generate_subject_instance_with_target_label_guarantees_at_risk_false():
    rng = random.Random(321)
    for _ in range(10):
        instance = generate_subject_instance_with_target_label(_SAMPLE_SUBJECT, False, rng)
        assert instance["at_risk"] is False


def test_at_risk_instance_does_not_start_already_failing():
    # The decline must be visible over the term, not already broken at checkpoint 1.
    rng = random.Random(123)
    instance = generate_subject_instance_with_target_label(_SAMPLE_SUBJECT, True, rng)
    first_unit_test_pct = instance["unit_tests"][0]["score"] / instance["unit_tests"][0]["max_marks"] * 100
    assert first_unit_test_pct > 45.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_subject_instance'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/data_generation/generator.py`:

```python
ATTENDANCE_RISK_THRESHOLD = 75.0
AGGREGATE_RISK_THRESHOLD = 40.0
NUM_WEEKS = 14
UNIT_TEST_WEEKS = [6, 12]
SUBMISSION_WEEKS = [2, 4, 6, 8, 10]


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
        attendance_series = declining_series(rng.uniform(80, 95), rng.uniform(55, 72), NUM_WEEKS, 2.5, rng)
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

    unit_test_pcts = marks_pct_series[:2]
    submission_pcts = marks_pct_series[2:]

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
    final_attendance_pct = attendance_weeks[-1]["attendance_pct"]

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
        "ete_score": ete_score,
        "aggregate_pct": aggregate_pct,
        "final_attendance_pct": final_attendance_pct,
        "at_risk": computed_at_risk,
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
        if instance["at_risk"] == target_at_risk:
            return instance
    raise RuntimeError(
        f"Could not generate a '{subject['subject_slug']}' instance with at_risk={target_at_risk} "
        f"after {max_attempts} attempts"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: PASS (18 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/data_generation/generator.py tests/data_generation/test_generator.py
git commit -m "feat: generate per-subject trajectories with no-leakage risk labels"
```

---

### Task 6: Full dataset assembly and JSON export

**Files:**
- Modify: `scripts/data_generation/generator.py`
- Modify: `tests/data_generation/test_generator.py`

**Interfaces:**
- Consumes: `generate_roster`, `assign_subjects`, `select_at_risk_students`, `generate_subject_instance_with_target_label` (Tasks 2-5).
- Produces: `generate_dataset(num_students: int = 75, seed: int = 42) -> list[dict]` (each entry: student fields from `generate_roster` plus `"subjects": list[dict]`, each subject dict the output of `generate_subject_instance_with_target_label`).

- [ ] **Step 1: Write the failing test**

Append to `tests/data_generation/test_generator.py`:

```python
from scripts.data_generation.generator import generate_dataset


def test_generate_dataset_produces_75_students_with_6_subjects_each():
    dataset = generate_dataset(num_students=75, seed=42)
    assert len(dataset) == 75
    assert all(len(student["subjects"]) == 6 for student in dataset)


def test_generate_dataset_is_reproducible_for_a_fixed_seed():
    first = generate_dataset(num_students=75, seed=42)
    second = generate_dataset(num_students=75, seed=42)
    assert first == second


def test_generate_dataset_at_risk_student_rate_is_within_5_to_15_percent():
    dataset = generate_dataset(num_students=75, seed=42)
    at_risk_students = [s for s in dataset if any(subj["at_risk"] for subj in s["subjects"])]
    rate = len(at_risk_students) / len(dataset)
    assert 0.05 <= rate <= 0.15


def test_generate_dataset_at_risk_students_have_exactly_one_flagged_subject():
    dataset = generate_dataset(num_students=75, seed=42)
    for student in dataset:
        flagged = [subj for subj in student["subjects"] if subj["at_risk"]]
        assert len(flagged) in (0, 1), student["student_id"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_dataset'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/data_generation/generator.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_generation/test_generator.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/data_generation/generator.py tests/data_generation/test_generator.py
git commit -m "feat: assemble the full 75-student synthetic dataset"
```

---

### Task 7: CLI entrypoint and end-to-end smoke test

**Files:**
- Create: `scripts/generate_synthetic_data.py`
- Test: `tests/data_generation/test_cli.py`

**Interfaces:**
- Consumes: `generate_dataset` (Task 6).
- Produces: `data/synthetic_students.json` on disk when run as a script.

- [ ] **Step 1: Write the failing test**

Create `tests/data_generation/test_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path


def test_cli_writes_valid_json_dataset(tmp_path):
    output_path = tmp_path / "synthetic_students.json"
    result = subprocess.run(
        [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(output_path), "--seed", "42"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 75
    assert len(data[0]["subjects"]) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data_generation/test_cli.py -v`
Expected: FAIL with a non-zero return code (`scripts/generate_synthetic_data.py` does not exist yet)

- [ ] **Step 3: Write minimal implementation**

Create `scripts/generate_synthetic_data.py`:

```python
"""CLI entrypoint: regenerate Rebound AI's synthetic ScholarTrack dataset.

Usage:
    python scripts/generate_synthetic_data.py [--output data/synthetic_students.json] [--seed 42]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data_generation.generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Rebound AI's synthetic student dataset.")
    parser.add_argument("--output", default="data/synthetic_students.json")
    parser.add_argument("--num-students", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate_dataset(num_students=args.num_students, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    at_risk_count = sum(1 for s in dataset if any(subj["at_risk"] for subj in s["subjects"]))
    print(f"Wrote {len(dataset)} students to {output_path} ({at_risk_count} flagged at-risk).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data_generation/test_cli.py -v`
Expected: PASS (1 test)

Then run the full suite to confirm nothing regressed:

Run: `pytest tests/data_generation/ -v`
Expected: PASS (23 tests)

- [ ] **Step 5: Generate the real demo dataset and commit**

```bash
python scripts/generate_synthetic_data.py
git add scripts/generate_synthetic_data.py tests/data_generation/test_cli.py data/synthetic_students.json
git commit -m "feat: add CLI entrypoint and generate the demo synthetic dataset"
```
