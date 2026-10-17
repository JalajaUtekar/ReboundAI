"""Tests for api/register.py's request-handling logic (`handle_register`)
against the REAL live Supabase database.

Unlike the other `tests/api/*.py` files, this one needs a REAL
`auth.users` row (not just a stubbed GoTrue response) because
`handle_register` inserts `profiles.id` as a foreign key to
`auth.users.id` -- there is no existing seeded auth user we're free to
mutate (the 120-student demo cohort's profiles already exist and
re-registering them would 409), so this creates one throwaway test user
via the Supabase Admin API (`SUPABASE_SERVICE_ROLE_KEY`, same approach as
`supabase/seed.py::_ensure_auth_user`), with a clearly-fake email/PRN
(`register-test-<uuid>@example.test` / `TEST-<uuid>`), and deletes that
auth user again in a module-scoped fixture teardown. Deleting the
`auth.users` row cascades (`on delete cascade`) through `profiles` ->
`students` -> `student_subjects`, so no separate row-level cleanup is
needed -- the live database is left exactly as it was before this test
module ran.

Only the GoTrue network call inside `verify_jwt_only` is stubbed (same
reasoning as tests/api/test_auth.py), returning this real throwaway
user's real id; every DB write/read below runs for real.
"""

from __future__ import annotations

import os
import uuid

import pytest

from api._lib.db import get_db_connection
from api.register import _ApiError, handle_register


def _env():
    from api._lib.auth import _env as load_env

    return load_env()


class _FakeGoTrueResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(scope="module")
def throwaway_auth_user():
    """Creates a real, throwaway `auth.users` row via the Supabase Admin
    API and deletes it (cascading to profiles/students/student_subjects)
    at teardown. Skips the whole module if SUPABASE_SERVICE_ROLE_KEY isn't
    configured, rather than failing -- this is infra the local/CI
    environment may not have."""
    import requests

    env = _env()
    supabase_url = env.get("SUPABASE_URL")
    service_role_key = env.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        pytest.skip("SUPABASE_SERVICE_ROLE_KEY not configured; cannot create a throwaway auth user")

    token = uuid.uuid4().hex[:10]
    email = f"register-test-{token}@example.test"

    headers = {
        "Authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key,
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{supabase_url}/auth/v1/admin/users",
        headers=headers,
        json={"email": email, "password": str(uuid.uuid4()), "email_confirm": True},
        timeout=20,
    )
    assert resp.status_code in (200, 201), f"failed to create throwaway auth user: {resp.status_code} {resp.text[:200]!r}"
    user_id = resp.json()["id"]

    yield {"user_id": user_id, "email": email, "token": token}

    requests.delete(
        f"{supabase_url}/auth/v1/admin/users/{user_id}",
        headers=headers,
        timeout=20,
    )


def _patch_gotrue_to_return(monkeypatch, user_id: str):
    def fake_get(url, headers=None, timeout=None):
        assert url.endswith("/auth/v1/user")
        return _FakeGoTrueResponse(200, {"id": user_id})

    monkeypatch.setattr("requests.get", fake_get)


def _valid_body(throwaway_auth_user, suffix=""):
    token = throwaway_auth_user["token"] + suffix
    return {
        "name": "Test Student",
        "prn": f"TEST-{token}",
        "email": throwaway_auth_user["email"],
        "branch": "CE",
        "academic_year": "TY",
        "mdm_slug": "genai",
        "oel_slug": "practical_japanese_communication",
        "elective_slug": "cloud_computing",
    }


def test_missing_authorization_header_is_401():
    with pytest.raises(_ApiError) as exc_info:
        handle_register(None, {})
    assert exc_info.value.status == 401


def test_missing_required_field_is_400(monkeypatch, throwaway_auth_user):
    _patch_gotrue_to_return(monkeypatch, throwaway_auth_user["user_id"])
    body = _valid_body(throwaway_auth_user)
    del body["prn"]
    with pytest.raises(_ApiError) as exc_info:
        handle_register("Bearer fake-but-whatever", body)
    assert exc_info.value.status == 400


def test_non_ce_branch_is_rejected(monkeypatch, throwaway_auth_user):
    _patch_gotrue_to_return(monkeypatch, throwaway_auth_user["user_id"])
    body = _valid_body(throwaway_auth_user)
    body["branch"] = "IT"
    with pytest.raises(_ApiError) as exc_info:
        handle_register("Bearer fake-but-whatever", body)
    assert exc_info.value.status == 400


def test_non_ty_academic_year_is_rejected(monkeypatch, throwaway_auth_user):
    _patch_gotrue_to_return(monkeypatch, throwaway_auth_user["user_id"])
    body = _valid_body(throwaway_auth_user)
    body["academic_year"] = "SY"
    with pytest.raises(_ApiError) as exc_info:
        handle_register("Bearer fake-but-whatever", body)
    assert exc_info.value.status == 400


def test_unknown_subject_slug_is_rejected(monkeypatch, throwaway_auth_user):
    _patch_gotrue_to_return(monkeypatch, throwaway_auth_user["user_id"])
    body = _valid_body(throwaway_auth_user)
    body["mdm_slug"] = "not_a_real_subject"
    with pytest.raises(_ApiError) as exc_info:
        handle_register("Bearer fake-but-whatever", body)
    assert exc_info.value.status == 400


def test_full_registration_lands_correct_rows_and_get_current_profile_matches(
    monkeypatch, throwaway_auth_user
):
    """The real end-to-end path: register a throwaway student, confirm
    profiles/students/student_subjects land correctly, then confirm
    `get_current_profile` (the same logic `/api/me` wraps) returns them."""
    _patch_gotrue_to_return(monkeypatch, throwaway_auth_user["user_id"])
    body = _valid_body(throwaway_auth_user)

    result = handle_register("Bearer fake-but-whatever", body)

    assert result["user_id"] == throwaway_auth_user["user_id"]
    assert result["role"] == "student"
    assert result["department"] == "CE"
    assert result["student_id"] is not None
    assert len(result["student_subjects"]) == 6
    slugs = {s["subject_slug"] for s in result["student_subjects"]}
    assert {"DSML", "CN", "SPOS", "genai", "practical_japanese_communication", "cloud_computing"} == slugs

    # Confirm the rows landed correctly, reading directly from the live DB.
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("select role, department from public.profiles where id = %s", (throwaway_auth_user["user_id"],))
        role, department = cur.fetchone()
        assert role == "student"
        assert department == "CE"

        cur.execute(
            "select name, prn, email, branch, academic_year from public.students where profile_id = %s",
            (throwaway_auth_user["user_id"],),
        )
        name, prn, email, branch, academic_year = cur.fetchone()
        assert name == "Test Student"
        assert prn == body["prn"]
        assert email == throwaway_auth_user["email"]
        assert branch == "CE"
        assert academic_year == "TY"

        cur.execute(
            "select subject_slug, subject_type, components from public.student_subjects "
            "where student_id = %s order by subject_slug",
            (result["student_id"],),
        )
        rows = cur.fetchall()
        assert len(rows) == 6
        by_slug = {slug: (subject_type, components) for slug, subject_type, components in rows}
        assert by_slug["DSML"][0] == "fixed"
        assert by_slug["genai"][0] == "mdm"
        assert by_slug["practical_japanese_communication"][0] == "oel"
        assert by_slug["cloud_computing"][0] == "elective"
        # Components are the real grading-weight dicts, e.g. {"TH": 40, "ETE": 60, "PR": 25}.
        assert by_slug["DSML"][1]["TH"] == 40
        assert by_slug["DSML"][1]["ETE"] == 60
        cur.close()
    finally:
        conn.close()

    # get_current_profile (the logic /api/me wraps) should now resolve this
    # user too, since the profiles row now exists.
    from api._lib.auth import get_current_profile

    profile = get_current_profile("Bearer fake-but-whatever")
    assert profile["user_id"] == throwaway_auth_user["user_id"]
    assert profile["role"] == "student"
    assert profile["department"] == "CE"
    assert profile["student_id"] == result["student_id"]
    assert len(profile["student_subjects"]) == 6


def test_re_registering_the_same_user_is_409(monkeypatch, throwaway_auth_user):
    """Depends on the previous test having already registered this user
    (module-scoped fixture, same throwaway_auth_user across the file)."""
    _patch_gotrue_to_return(monkeypatch, throwaway_auth_user["user_id"])
    body = _valid_body(throwaway_auth_user, suffix="-again")
    with pytest.raises(_ApiError) as exc_info:
        handle_register("Bearer fake-but-whatever", body)
    assert exc_info.value.status == 409
