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
