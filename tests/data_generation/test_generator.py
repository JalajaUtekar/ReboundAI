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
