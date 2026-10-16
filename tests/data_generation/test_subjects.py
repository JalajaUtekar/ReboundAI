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
