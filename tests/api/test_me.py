"""Tests for api/me.py's request-handling logic (`handle_me`) against the
REAL live Supabase database.

Only the GoTrue network call inside `get_current_profile` is stubbed (see
tests/api/test_auth.py for why); the profiles/students/student_subjects
lookup that follows runs for real. `handle_me` itself is a thin wrapper
around `get_current_profile`, so these tests mostly confirm the HTTP-error
mapping (401 on AuthError) and that the real profile shape passes through
untouched.
"""

from __future__ import annotations

import pytest

from api._lib.db import get_db_connection
from api.me import _ApiError, handle_me


class _FakeGoTrueResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _fetch_real_student_profile_id() -> str:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "select p.id from public.profiles p "
            "join public.students s on s.profile_id = p.id "
            "order by s.prn limit 1"
        )
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()
    if row is None:
        pytest.skip("no seeded student profile found in the live database")
    return row[0]


@pytest.fixture(scope="module")
def real_student_profile_id() -> str:
    return _fetch_real_student_profile_id()


def test_missing_authorization_header_is_401():
    with pytest.raises(_ApiError) as exc_info:
        handle_me(None)
    assert exc_info.value.status == 401


def test_gotrue_rejecting_the_token_is_401(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return _FakeGoTrueResponse(401, {"error": "invalid_token"})

    monkeypatch.setattr("requests.get", fake_get)
    with pytest.raises(_ApiError) as exc_info:
        handle_me("Bearer not-a-real-token")
    assert exc_info.value.status == 401


def test_real_student_profile_returned_against_live_db(monkeypatch, real_student_profile_id):
    def fake_get(url, headers=None, timeout=None):
        assert url.endswith("/auth/v1/user")
        return _FakeGoTrueResponse(200, {"id": real_student_profile_id})

    monkeypatch.setattr("requests.get", fake_get)

    profile = handle_me("Bearer fake-but-whatever")

    assert profile["user_id"] == real_student_profile_id
    assert profile["role"] == "student"
    assert profile["student_id"] is not None
    assert isinstance(profile["student_subjects"], list)
    for subject in profile["student_subjects"]:
        assert {"id", "subject_slug", "subject_name", "subject_type", "components"} <= subject.keys()


def test_user_with_no_profile_row_is_401(monkeypatch):
    """A brand-new auth user with no `profiles` row yet (hasn't registered)
    -- `get_current_profile` raises AuthError for this, which `handle_me`
    maps to 401 (the frontend treats this as "needs to register")."""
    import uuid

    nonexistent_user_id = str(uuid.uuid4())

    def fake_get(url, headers=None, timeout=None):
        return _FakeGoTrueResponse(200, {"id": nonexistent_user_id})

    monkeypatch.setattr("requests.get", fake_get)
    with pytest.raises(_ApiError) as exc_info:
        handle_me("Bearer fake-but-whatever")
    assert exc_info.value.status == 401
