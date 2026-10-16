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


def test_select_at_risk_students_count_within_10_to_15_percent():
    student_ids = [f"S{i:03d}" for i in range(1, 121)]
    rng = random.Random(3)
    at_risk = select_at_risk_students(student_ids, rng)
    assert 12 <= len(at_risk) <= 18  # ceil(10% of 120) .. floor(15% of 120)


def test_select_at_risk_students_returns_subset_with_no_duplicates():
    student_ids = [f"S{i:03d}" for i in range(1, 121)]
    rng = random.Random(9)
    at_risk = select_at_risk_students(student_ids, rng)
    assert len(at_risk) == len(set(at_risk))
    assert set(at_risk).issubset(set(student_ids))


def test_select_at_risk_students_respects_custom_rate_bounds():
    student_ids = [f"S{i:03d}" for i in range(1, 101)]
    rng = random.Random(11)
    at_risk = select_at_risk_students(student_ids, rng, min_rate=0.10, max_rate=0.10)
    assert len(at_risk) == 10


from scripts.data_generation.generator import select_failing_students


def test_select_failing_students_is_a_smaller_nonempty_proper_subset():
    at_risk_ids = [f"S{i:03d}" for i in range(1, 16)]  # 15 at-risk students
    rng = random.Random(13)
    failing = select_failing_students(at_risk_ids, rng)
    assert 1 <= len(failing) < len(at_risk_ids)
    assert set(failing).issubset(set(at_risk_ids))


def test_select_failing_students_returns_empty_for_empty_input():
    rng = random.Random(13)
    assert select_failing_students([], rng) == []


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
    assert 0 <= instance["outcome"]["ete_score"] <= 60


def test_generate_subject_instance_at_risk_label_matches_computed_thresholds():
    rng = random.Random(4)
    instance = generate_subject_instance(_SAMPLE_SUBJECT, at_risk=True, rng=rng)
    outcome = instance["outcome"]
    expected = (outcome["final_attendance_pct"] < 75.0) or (outcome["aggregate_pct"] < 70.0)
    assert outcome["at_risk"] == expected


def test_generate_subject_instance_is_failing_label_matches_computed_threshold():
    rng = random.Random(4)
    instance = generate_subject_instance(_SAMPLE_SUBJECT, at_risk=True, rng=rng, failing=True)
    outcome = instance["outcome"]
    assert outcome["is_failing"] == (outcome["aggregate_pct"] < 40.0)


def test_is_failing_implies_at_risk_invariant():
    # 40% < 70%, so a failing aggregate must also be an at-risk aggregate -- checked
    # across a range of seeds and both at_risk/failing combinations, not just one draw.
    for seed in range(100):
        rng = random.Random(seed)
        at_risk_flag = seed % 2 == 0
        failing_flag = seed % 3 == 0
        instance = generate_subject_instance(_SAMPLE_SUBJECT, at_risk=at_risk_flag, rng=rng, failing=failing_flag)
        outcome = instance["outcome"]
        if outcome["is_failing"]:
            assert outcome["at_risk"] is True, (seed, outcome)


def test_generate_subject_instance_with_target_label_guarantees_at_risk_true():
    rng = random.Random(123)
    for _ in range(10):
        instance = generate_subject_instance_with_target_label(_SAMPLE_SUBJECT, True, rng)
        assert instance["outcome"]["at_risk"] is True


def test_generate_subject_instance_with_target_label_guarantees_at_risk_false():
    rng = random.Random(321)
    for _ in range(10):
        instance = generate_subject_instance_with_target_label(_SAMPLE_SUBJECT, False, rng)
        assert instance["outcome"]["at_risk"] is False


def test_generate_subject_instance_with_target_label_guarantees_failing_true():
    rng = random.Random(555)
    for _ in range(10):
        instance = generate_subject_instance_with_target_label(_SAMPLE_SUBJECT, True, rng, target_failing=True)
        assert instance["outcome"]["at_risk"] is True
        assert instance["outcome"]["is_failing"] is True


def test_generate_subject_instance_with_target_label_guarantees_at_risk_not_failing():
    rng = random.Random(777)
    for _ in range(10):
        instance = generate_subject_instance_with_target_label(_SAMPLE_SUBJECT, True, rng, target_failing=False)
        assert instance["outcome"]["at_risk"] is True
        assert instance["outcome"]["is_failing"] is False


def test_at_risk_instance_does_not_start_already_failing():
    # The decline must be visible over the term, not already broken at the first
    # chronological checkpoint, which is the week-2 submission.
    rng = random.Random(123)
    instance = generate_subject_instance_with_target_label(_SAMPLE_SUBJECT, True, rng)
    first_checkpoint = instance["submissions"][0]
    assert first_checkpoint["due_day"] == 14  # week 2
    first_pct = first_checkpoint["score"] / first_checkpoint["max_marks"] * 100
    assert first_pct > 45.0


def test_at_risk_label_is_computed_not_echoed_from_input_flag():
    # Seed 28 with at_risk=False computes to at_risk=True via the aggregate<70 threshold,
    # with attendance remaining above the 75% cutoff.
    # If the return value ever echoed the input flag, this would fail.
    instance = generate_subject_instance(_SAMPLE_SUBJECT, at_risk=False, rng=random.Random(28))
    outcome = instance["outcome"]
    assert outcome["final_attendance_pct"] >= 75.0
    assert outcome["aggregate_pct"] < 70.0
    assert outcome["at_risk"] is True


def test_aggregate_pct_matches_component_formula():
    rng = random.Random(4)
    instance = generate_subject_instance(_SAMPLE_SUBJECT, at_risk=False, rng=rng)
    outcome = instance["outcome"]
    expected = round((instance["th_score"] + outcome["ete_score"] + instance["pr_tw_score"]) / 125 * 100, 1)
    assert outcome["aggregate_pct"] == expected


from scripts.data_generation.generator import generate_dataset


def test_generate_dataset_produces_120_students_with_6_subjects_each():
    dataset = generate_dataset(num_students=120, seed=42)
    assert len(dataset) == 120
    assert all(len(student["subjects"]) == 6 for student in dataset)


def test_generate_dataset_defaults_to_120_students():
    dataset = generate_dataset(seed=42)
    assert len(dataset) == 120


def test_generate_dataset_is_reproducible_for_a_fixed_seed():
    first = generate_dataset(num_students=120, seed=42)
    second = generate_dataset(num_students=120, seed=42)
    assert first == second


def test_generate_dataset_at_risk_student_rate_is_within_10_to_15_percent():
    dataset = generate_dataset(num_students=120, seed=42)
    at_risk_students = [s for s in dataset if any(subj["outcome"]["at_risk"] for subj in s["subjects"])]
    rate = len(at_risk_students) / len(dataset)
    assert 0.10 <= rate <= 0.15


def test_generate_dataset_at_risk_students_have_exactly_one_flagged_subject():
    dataset = generate_dataset(num_students=120, seed=42)
    for student in dataset:
        flagged = [subj for subj in student["subjects"] if subj["outcome"]["at_risk"]]
        assert len(flagged) in (0, 1), student["student_id"]


def test_generate_dataset_has_at_least_one_but_not_all_failing_instances():
    dataset = generate_dataset(num_students=120, seed=42)
    all_subjects = [subj for student in dataset for subj in student["subjects"]]
    failing = [subj for subj in all_subjects if subj["outcome"]["is_failing"]]
    assert 1 <= len(failing) < len(all_subjects)


def test_generate_dataset_is_failing_implies_at_risk_across_all_instances():
    dataset = generate_dataset(num_students=120, seed=42)
    for student in dataset:
        for subj in student["subjects"]:
            outcome = subj["outcome"]
            if outcome["is_failing"]:
                assert outcome["at_risk"] is True, (student["student_id"], subj["subject_slug"])


from scripts.data_generation.generator import CHECKPOINTS_IN_TIME_ORDER


def test_checkpoints_are_in_exact_chronological_order():
    # Regression test for the Task-5 chronological-ordering fix: direct, literal
    # assertion on the structure itself, not an indirect behavioral check.
    assert CHECKPOINTS_IN_TIME_ORDER == [
        ("submission", 0),  # week 2
        ("submission", 1),  # week 4
        ("submission", 2),  # week 6
        ("unit_test", 0),   # week 6 (submissions break ties before unit tests)
        ("submission", 3),  # week 8
        ("submission", 4),  # week 10
        ("unit_test", 1),   # week 12
    ]
