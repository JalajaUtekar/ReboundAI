"""
Rebound AI -- POST /api/register

Vercel Python Serverless Function. Completes a brand-new Supabase Auth
user's registration (spec section 2's 4-step flow, submitted as one final
POST from the frontend): creates their `profiles` row (role='student',
department=branch), their `students` row, and their 6 `student_subjects`
rows (3 fixed: DSML/CN/SPOS + their chosen MDM/OEL/Elective).

Request:
    POST /api/register
    Headers: Authorization: Bearer <supabase JWT>
    Body (JSON): {
        "name": "<str>", "prn": "<str>", "email": "<str>",
        "branch": "CE", "academic_year": "TY",
        "mdm_slug": "<slug>", "oel_slug": "<slug>", "elective_slug": "<slug>"
    }
    `*_slug` values are `subject_slug(name)` for one of the options in
    `scripts/data_generation/subjects.py` (MDM_OPTIONS/OEL_OPTIONS/
    ELECTIVE_OPTIONS) -- see `web/lib/subjects.ts::subjectSlug` for the
    frontend's matching slug derivation.

Response 201 (JSON): the profile, in the same shape
    `api/_lib/auth.py::get_current_profile` returns (so the frontend can
    reuse it directly / `GET /api/me` immediately after will match).

Error responses (JSON body {"error": "<message>"}):
    400 - malformed body / missing or invalid field / unknown subject slug /
          branch or academic_year not yet launched (spec section 2: only
          CE + TY are enabled today, enforced server-side as well as in
          the UI) / PRN or email already registered
    401 - missing/invalid/expired bearer token
    409 - this auth user already has a profiles row (registration is a
          one-time action; re-registering is not supported)
    405 - method other than POST
    500 - unexpected server error

Note on auth: this uses `verify_jwt_only` (not `get_current_profile`),
since a brand-new user has no `profiles` row yet -- that's exactly what
this endpoint creates.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict

ENABLED_BRANCH = "CE"
ENABLED_ACADEMIC_YEAR = "TY"

_FIXED_SUBJECTS = ["DSML", "CN", "SPOS"]


class _ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _resolve_choice_subject(slug: str, options: Dict[str, dict], field_name: str):
    """Looks up `slug` among `options`' derived slugs (same derivation as
    `web/lib/subjects.ts::subjectSlug`) and returns (subject_name, components)."""
    from scripts.data_generation.subjects import subject_slug as to_slug

    for name, meta in options.items():
        if to_slug(name) == slug:
            return name, meta["components"]
    raise _ApiError(400, f"{field_name} {slug!r} is not a recognized subject")


def handle_register(authorization_header: "str | None", body: Dict[str, Any]) -> Dict[str, Any]:
    """Core request handler, factored out of the HTTP plumbing so it can be
    unit-tested directly without going through BaseHTTPRequestHandler."""
    from scripts.data_generation.subjects import ELECTIVE_OPTIONS, FIXED_SUBJECTS, MDM_OPTIONS, OEL_OPTIONS

    from api._lib.auth import AuthError, verify_jwt_only
    from api._lib.db import get_db_connection

    try:
        user_id = verify_jwt_only(authorization_header)
    except AuthError as exc:
        raise _ApiError(401, str(exc) or "invalid or missing authorization token")

    name = body.get("name")
    prn = body.get("prn")
    email = body.get("email")
    branch = body.get("branch")
    academic_year = body.get("academic_year")
    mdm_slug = body.get("mdm_slug")
    oel_slug = body.get("oel_slug")
    elective_slug = body.get("elective_slug")

    for field_name, value in (
        ("name", name),
        ("prn", prn),
        ("email", email),
        ("branch", branch),
        ("academic_year", academic_year),
        ("mdm_slug", mdm_slug),
        ("oel_slug", oel_slug),
        ("elective_slug", elective_slug),
    ):
        if not value or not isinstance(value, str):
            raise _ApiError(400, f"{field_name} is required")

    # Spec section 2: only CE / TY are launched today. The UI already
    # disables the other options, but the backend enforces it too rather
    # than trusting the client.
    if branch != ENABLED_BRANCH:
        raise _ApiError(400, f"branch must be {ENABLED_BRANCH!r} (other branches are coming soon)")
    if academic_year != ENABLED_ACADEMIC_YEAR:
        raise _ApiError(
            400, f"academic_year must be {ENABLED_ACADEMIC_YEAR!r} (other years are coming soon)"
        )

    mdm_name, mdm_components = _resolve_choice_subject(mdm_slug, MDM_OPTIONS, "mdm_slug")
    oel_name, oel_components = _resolve_choice_subject(oel_slug, OEL_OPTIONS, "oel_slug")
    elective_name, elective_components = _resolve_choice_subject(
        elective_slug, ELECTIVE_OPTIONS, "elective_slug"
    )

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("select 1 from public.profiles where id = %s", (user_id,))
        if cursor.fetchone() is not None:
            raise _ApiError(409, "this account has already completed registration")

        cursor.execute("select 1 from public.students where prn = %s", (prn,))
        if cursor.fetchone() is not None:
            raise _ApiError(400, f"PRN {prn!r} is already registered")

        cursor.execute("select 1 from public.students where email = %s", (email,))
        if cursor.fetchone() is not None:
            raise _ApiError(400, f"email {email!r} is already registered")

        cursor.execute(
            "insert into public.profiles (id, role, department) values (%s, 'student', %s)",
            (user_id, branch),
        )

        cursor.execute(
            """
            insert into public.students (profile_id, name, prn, email, photo_url, branch, academic_year)
            values (%s, %s, %s, %s, %s, %s, %s)
            returning id
            """,
            (user_id, name, prn, email, None, branch, academic_year),
        )
        student_id = cursor.fetchone()[0]

        subjects_to_insert = []
        for slug in _FIXED_SUBJECTS:
            meta = FIXED_SUBJECTS[slug]
            subjects_to_insert.append((slug, meta["name"], "fixed", meta["components"]))
        subjects_to_insert.append((mdm_slug, mdm_name, "mdm", mdm_components))
        subjects_to_insert.append((oel_slug, oel_name, "oel", oel_components))
        subjects_to_insert.append((elective_slug, elective_name, "elective", elective_components))

        for slug, subject_name, subject_type, components in subjects_to_insert:
            cursor.execute(
                """
                insert into public.student_subjects
                    (student_id, subject_slug, subject_name, subject_type, components)
                values (%s, %s, %s, %s, %s::jsonb)
                """,
                (student_id, slug, subject_name, subject_type, json.dumps(components)),
            )

        conn.commit()
    except _ApiError:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Returned in the same shape get_current_profile() produces (minus the
    # "id" field on each subject, which get_current_profile adds from the
    # DB -- the frontend's immediate follow-up GET /api/me call will have
    # it; this response is just the "registration succeeded" confirmation).
    return {
        "user_id": user_id,
        "role": "student",
        "department": branch,
        "student_id": str(student_id),
        "student_subjects": [
            {
                "subject_slug": slug,
                "subject_name": subject_name,
                "subject_type": subject_type,
                "components": components,
            }
            for slug, subject_name, subject_type, components in subjects_to_insert
        ],
    }


class handler(BaseHTTPRequestHandler):  # noqa: N801 -- Vercel Python runtime convention
    """Vercel Python Serverless Function entrypoint for POST /api/register."""

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            content_length = int(self.headers.get("Content-Length", 0) or 0)
            raw_body = self.rfile.read(content_length) if content_length else b"{}"
            try:
                body = json.loads(raw_body or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "request body must be valid JSON"})
                return

            auth_header = self.headers.get("Authorization")
            result = handle_register(auth_header, body)
            self._send_json(201, result)
        except _ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive catch-all
            self._send_json(500, {"error": f"internal server error: {exc}"})

    def do_GET(self) -> None:  # noqa: N802
        self._send_json(405, {"error": "method not allowed, use POST"})
