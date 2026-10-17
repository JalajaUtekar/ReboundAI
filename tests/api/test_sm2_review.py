"""
Tests for api/rebound/sm2_review.py.

Two layers:
  1. Pure SM-2 math -- calls Lecture2Revision/flashcard_service.py's
     calculate_sm2_update directly with known inputs and checks known
     outputs. No DB, no mocking, this is the ported algorithm itself.
  2. Handler-level logic (ownership check, status codes, DB writes) --
     exercised against FAKE api._lib.db / api._lib.auth modules injected
     into sys.modules, since api/_lib/db.py and api/_lib/auth.py are being
     built by another agent in parallel and may not exist yet (only
     api/_lib/__init__.py exists at the time this was written). If/when the
     real modules land with the documented signatures, these fakes are
     simply replaced for this test run via sys.modules and the real
     production code path is unaffected.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
L2R_DIR = REPO_ROOT / "Lecture2Revision"
if str(L2R_DIR) not in sys.path:
    sys.path.insert(0, str(L2R_DIR))

from flashcard_service import calculate_sm2_update  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Pure SM-2 algorithm tests (no mocking needed)
# ---------------------------------------------------------------------------

class TestCalculateSm2Update:
    def test_first_remembered_review_sets_interval_to_1_day(self):
        result = calculate_sm2_update(
            current_interval=1.0, current_ease=2.5, current_repetitions=0, rating="remembered"
        )
        assert result["interval_days"] == 1.0
        assert result["repetitions"] == 1
        # EF' = 2.5 + (0.1 - 1*(0.08+1*0.02)) = 2.5 + (0.1-0.1) = 2.5
        assert result["ease_factor"] == 2.5

    def test_second_remembered_review_sets_interval_to_6_days(self):
        result = calculate_sm2_update(
            current_interval=1.0, current_ease=2.5, current_repetitions=1, rating="remembered"
        )
        assert result["interval_days"] == 6.0
        assert result["repetitions"] == 2
        assert result["ease_factor"] == 2.5

    def test_third_remembered_review_multiplies_interval_by_ease(self):
        result = calculate_sm2_update(
            current_interval=6.0, current_ease=2.5, current_repetitions=2, rating="remembered"
        )
        # new_interval = round(6.0 * 2.5, 1) = 15.0
        assert result["interval_days"] == 15.0
        assert result["repetitions"] == 3
        assert result["ease_factor"] == 2.5

    def test_needs_practice_resets_repetitions_and_interval(self):
        result = calculate_sm2_update(
            current_interval=15.0, current_ease=2.5, current_repetitions=3, rating="needs_practice"
        )
        assert result["interval_days"] == 1.0
        assert result["repetitions"] == 0
        # ease reduced by 0.2
        assert result["ease_factor"] == 2.3

    def test_ease_factor_never_drops_below_minimum(self):
        result = calculate_sm2_update(
            current_interval=1.0, current_ease=1.35, current_repetitions=0, rating="needs_practice"
        )
        # 1.35 - 0.2 = 1.15, clamped to SM2_MIN_EASE_FACTOR (1.3)
        assert result["ease_factor"] == 1.3

    def test_defaults_applied_when_current_state_is_none(self):
        result = calculate_sm2_update(
            current_interval=None, current_ease=None, current_repetitions=None, rating="remembered"
        )
        assert result["repetitions"] == 1
        assert result["interval_days"] == 1.0
        assert result["ease_factor"] == 2.5

    def test_next_review_date_is_a_date_string(self):
        result = calculate_sm2_update(
            current_interval=1.0, current_ease=2.5, current_repetitions=0, rating="remembered"
        )
        assert len(result["next_review_date"]) == 10
        assert result["next_review_date"].count("-") == 2


# ---------------------------------------------------------------------------
# 2. Handler-level logic, against fake api._lib.db / api._lib.auth
# ---------------------------------------------------------------------------

STUDENT_ID = "11111111-1111-1111-1111-111111111111"
OWNED_SUBJECT_ID = "22222222-2222-2222-2222-222222222222"
OTHER_SUBJECT_ID = "33333333-3333-3333-3333-333333333333"
FLASHCARD_ID = "44444444-4444-4444-4444-444444444444"


class FakeCursor:
    def __init__(self, flashcard_row):
        self._flashcard_row = flashcard_row
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql.strip().lower(), params))

    def fetchone(self):
        sql, _params = self.executed[-1]
        if sql.startswith("select f.id"):
            return self._flashcard_row
        return None


class FakeConnection:
    def __init__(self, flashcard_row):
        self.cursor_obj = FakeCursor(flashcard_row)
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _install_fake_lib_modules(monkeypatch, *, profile, flashcard_row, raise_auth_error=False):
    """Injects fake api._lib.db / api._lib.auth modules (matching the
    documented interface) into sys.modules, then (re-)imports
    api.rebound.sm2_review against them."""
    fake_auth = types.ModuleType("api._lib.auth")

    class AuthError(Exception):
        pass

    def get_current_profile(authorization_header):
        if raise_auth_error or not authorization_header:
            raise AuthError("invalid or missing token")
        return profile

    fake_auth.AuthError = AuthError
    fake_auth.get_current_profile = get_current_profile

    fake_db = types.ModuleType("api._lib.db")
    connection_holder = {}

    def get_db_connection():
        conn = FakeConnection(flashcard_row)
        connection_holder["conn"] = conn
        return conn

    fake_db.get_db_connection = get_db_connection

    monkeypatch.setitem(sys.modules, "api._lib.auth", fake_auth)
    monkeypatch.setitem(sys.modules, "api._lib.db", fake_db)
    monkeypatch.delitem(sys.modules, "api.rebound.sm2_review", raising=False)

    import importlib

    module = importlib.import_module("api.rebound.sm2_review")
    return module, connection_holder


@pytest.fixture
def owning_profile():
    return {
        "user_id": "auth-user-id",
        "role": "student",
        "department": "CE",
        "student_id": STUDENT_ID,
        "student_subjects": [{"id": OWNED_SUBJECT_ID, "subject_slug": "DSML"}],
    }


def test_rejects_missing_authorization(monkeypatch, owning_profile):
    module, _ = _install_fake_lib_modules(
        monkeypatch,
        profile=owning_profile,
        flashcard_row=(FLASHCARD_ID, 1.0, 2.5, 0, OWNED_SUBJECT_ID),
        raise_auth_error=True,
    )
    with pytest.raises(module._ApiError) as exc_info:
        module.handle_sm2_review(None, {"flashcard_id": FLASHCARD_ID, "rating": "remembered"})
    assert exc_info.value.status == 401


def test_rejects_invalid_rating(monkeypatch, owning_profile):
    module, _ = _install_fake_lib_modules(
        monkeypatch,
        profile=owning_profile,
        flashcard_row=(FLASHCARD_ID, 1.0, 2.5, 0, OWNED_SUBJECT_ID),
    )
    with pytest.raises(module._ApiError) as exc_info:
        module.handle_sm2_review(
            "Bearer faketoken", {"flashcard_id": FLASHCARD_ID, "rating": "sort_of"}
        )
    assert exc_info.value.status == 400


def test_rejects_flashcard_not_owned_by_caller(monkeypatch, owning_profile):
    module, _ = _install_fake_lib_modules(
        monkeypatch,
        profile=owning_profile,
        flashcard_row=(FLASHCARD_ID, 1.0, 2.5, 0, OTHER_SUBJECT_ID),
    )
    with pytest.raises(module._ApiError) as exc_info:
        module.handle_sm2_review(
            "Bearer faketoken", {"flashcard_id": FLASHCARD_ID, "rating": "remembered"}
        )
    assert exc_info.value.status == 403


def test_returns_404_when_flashcard_missing(monkeypatch, owning_profile):
    module, _ = _install_fake_lib_modules(
        monkeypatch, profile=owning_profile, flashcard_row=None
    )
    with pytest.raises(module._ApiError) as exc_info:
        module.handle_sm2_review(
            "Bearer faketoken", {"flashcard_id": FLASHCARD_ID, "rating": "remembered"}
        )
    assert exc_info.value.status == 404


def test_successful_review_updates_and_commits(monkeypatch, owning_profile):
    module, connections = _install_fake_lib_modules(
        monkeypatch,
        profile=owning_profile,
        flashcard_row=(FLASHCARD_ID, 1.0, 2.5, 0, OWNED_SUBJECT_ID),
    )
    result = module.handle_sm2_review(
        "Bearer faketoken", {"flashcard_id": FLASHCARD_ID, "rating": "remembered"}
    )
    assert result["flashcard_id"] == FLASHCARD_ID
    assert result["interval_days"] == 1.0
    assert result["repetitions"] == 1
    assert result["needs_practice"] is False
    assert connections["conn"].committed is True
    assert connections["conn"].closed is True
