"""Rebound AI -- request authentication for Vercel Python Serverless Functions.

Verifies a Supabase Auth JWT by delegating to GoTrue itself (`GET
{SUPABASE_URL}/auth/v1/user` with the caller's own `Authorization` header
plus `apikey: {SUPABASE_ANON_KEY}`) rather than re-implementing JWT
signature verification here -- this is the standard way to verify a
Supabase session server-side without needing the JWT secret, and it means
token expiry/revocation is always checked against the real, live auth
server instead of a locally cached secret.

Once the caller's `auth.users.id` is known, `get_current_profile` looks up
the matching `profiles` row (role/department) and, for students, their
`students` row and all of their own `student_subjects` rows -- the shape
every other `api/*.py` endpoint needs for RLS-equivalent scoping checks at
the application layer (spec section 8: a student may only ever act on their
own rows).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class AuthError(Exception):
    """Raised when the Authorization header is missing/malformed, the
    Supabase Auth JWT fails verification (expired, revoked, garbage token),
    or the authenticated user has no matching `profiles` row yet. Calling
    endpoints should catch this and respond 401 (or, for the "no profile
    row" case, arguably 404/409 -- but 401 is the safe default since the
    caller's identity couldn't be resolved to anything we're willing to
    act on)."""


def _load_dotenv() -> Dict[str, str]:
    # Duplicated from api/_lib/db.py rather than imported, so this module
    # has no import-order dependency on db.py -- both are tiny and this
    # keeps each file independently readable/testable.
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import dotenv_values  # type: ignore

        return {k: v for k, v in dotenv_values(env_path).items() if v is not None}
    except ImportError:
        values: Dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
        return values


def _env() -> Dict[str, str]:
    return {**_load_dotenv(), **os.environ}


def get_current_profile(authorization_header: Optional[str]) -> dict:
    """Verifies `authorization_header` (e.g. `"Bearer eyJ..."`) against
    Supabase Auth and returns the caller's profile.

    Returns:
        {
            "user_id": str,            # auth.users.id / profiles.id
            "role": str,                # 'student' | 'professor' | 'hod' | 'principal'
            "department": str | None,
            "student_id": str | None,   # students.id, only set when role == 'student'
            "student_subjects": list[dict],  # the student's own rows; [] when role != 'student'
        }

    Raises `AuthError` for a missing/invalid/expired token, or a valid token
    whose user has no `profiles` row.

    Per spec section 8, only the student path is fully built out today:
    professor/hod/principal callers get `student_id=None` and
    `student_subjects=[]` without erroring, but their own department-scoped
    read paths are deliberately not implemented here yet.
    """
    if not authorization_header:
        raise AuthError("Missing Authorization header")

    env = _env()
    supabase_url = env.get("SUPABASE_URL")
    anon_key = env.get("SUPABASE_ANON_KEY")
    if not supabase_url or not anon_key:
        raise AuthError(
            "Server misconfigured: SUPABASE_URL/SUPABASE_ANON_KEY not set"
        )

    import requests

    resp = requests.get(
        f"{supabase_url}/auth/v1/user",
        headers={"Authorization": authorization_header, "apikey": anon_key},
        timeout=10,
    )
    if resp.status_code != 200:
        raise AuthError(
            f"Invalid or expired session (GET /auth/v1/user returned {resp.status_code})"
        )

    try:
        auth_user = resp.json()
    except ValueError as exc:
        raise AuthError("Supabase auth response was not valid JSON") from exc

    user_id = auth_user.get("id")
    if not user_id:
        raise AuthError("Supabase auth response did not include a user id")

    from .db import get_db_connection

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "select role, department from public.profiles where id = %s",
                (user_id,),
            )
            profile_row = cur.fetchone()
            if profile_row is None:
                raise AuthError(
                    f"Authenticated user {user_id!r} has no profiles row yet"
                )
            role, department = profile_row

            student_id = None
            student_subjects: List[dict] = []

            if role == "student":
                cur.execute(
                    "select id from public.students where profile_id = %s",
                    (user_id,),
                )
                student_row = cur.fetchone()
                if student_row is not None:
                    student_id = student_row[0]
                    cur.execute(
                        "select id, subject_slug, subject_name, subject_type, components "
                        "from public.student_subjects where student_id = %s "
                        "order by subject_slug",
                        (student_id,),
                    )
                    for ss_id, slug, name, subject_type, components in cur.fetchall():
                        student_subjects.append(
                            {
                                "id": ss_id,
                                "subject_slug": slug,
                                "subject_name": name,
                                "subject_type": subject_type,
                                "components": components,
                            }
                        )
        finally:
            cur.close()
    finally:
        conn.close()

    return {
        "user_id": user_id,
        "role": role,
        "department": department,
        "student_id": student_id,
        "student_subjects": student_subjects,
    }
