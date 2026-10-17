"""Tests for api/_lib/auth.py::get_current_profile against the REAL live
Supabase database (120 real seeded students).

Getting a real bearer JWT without a literal browser login flow has two
options per the task brief: mint a one-off session via the Admin API, or
monkeypatch only the network call to GoTrue (`requests.get`) with a known
real `auth.users.id` and let everything downstream -- the `profiles`/
`students`/`student_subjects` lookups -- run for real against the live
database. This file takes the second option: it is far less brittle (no
extra admin-API round trip, no token TTL to worry about, nothing to clean
up afterward) and it still exercises 100% of `get_current_profile`'s own
logic -- the only thing being stubbed out is GoTrue's own JWT verification,
which is Supabase's code, not ours, and is not what this module is
responsible for testing.
"""

from __future__ import annotations

import pytest

from api._lib.auth import AuthError, get_current_profile
from api._lib.db import get_db_connection

# Real seeded profile ids (profiles.id == auth.users.id), looked up live
# from the real database at collection time so this file has no hardcoded
# ids that could drift if the dataset is reseeded.


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


class _FakeGoTrueResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_missing_authorization_header_raises_auth_error():
    with pytest.raises(AuthError):
        get_current_profile(None)
    with pytest.raises(AuthError):
        get_current_profile("")


def test_gotrue_rejecting_the_token_raises_auth_error(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return _FakeGoTrueResponse(401, {"error": "invalid_token"})

    monkeypatch.setattr("requests.get", fake_get)
    with pytest.raises(AuthError):
        get_current_profile("Bearer not-a-real-token")


def test_real_student_profile_lookup_against_live_db(monkeypatch, real_student_profile_id):
    """Stubs only the GoTrue network call (to return the real user id);
    the profiles/students/student_subjects lookups that follow run for
    real against the live database."""

    def fake_get(url, headers=None, timeout=None):
        assert url.endswith("/auth/v1/user")
        assert headers.get("Authorization") == "Bearer fake-but-whatever"
        return _FakeGoTrueResponse(200, {"id": real_student_profile_id})

    monkeypatch.setattr("requests.get", fake_get)

    profile = get_current_profile("Bearer fake-but-whatever")

    assert profile["user_id"] == real_student_profile_id
    assert profile["role"] == "student"
    assert profile["department"] in ("AI&DS", "CE", "IT", "E&TC", "ME", "EE")
    assert profile["student_id"] is not None
    assert isinstance(profile["student_subjects"], list)
    # Real TY CE-style cohort: 6 subjects/student per spec section 3.
    assert len(profile["student_subjects"]) >= 1
    for subject in profile["student_subjects"]:
        assert {"id", "subject_slug", "subject_name", "subject_type", "components"} <= subject.keys()
