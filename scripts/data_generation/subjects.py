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
