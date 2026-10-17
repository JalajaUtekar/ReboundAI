"""
Rebound AI -- GET /api/me

Vercel Python Serverless Function. Thin wrapper around
`api/_lib/auth.py::get_current_profile`: verifies the caller's Supabase
Auth JWT and returns their profile (role/department, and for students,
`student_id` + `student_subjects`) -- the shape `web/contexts/AuthContext.tsx`
fetches on every session change.

Request:
    GET /api/me
    Headers: Authorization: Bearer <supabase JWT>

Response 200 (JSON): the profile, as returned by `get_current_profile`:
    {
        "user_id": "<uuid>",
        "role": "student" | "professor" | "hod" | "principal",
        "department": "<str>" | null,
        "student_id": "<uuid>" | null,
        "student_subjects": [...]
    }

Error responses (JSON body {"error": "<message>"}):
    401 - missing/invalid/expired bearer token, OR a valid token whose user
          has no `profiles` row yet (brand-new user who hasn't registered --
          the frontend treats this as "needs to register", not a hard error)
    405 - method other than GET
    500 - unexpected server error
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict


class _ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def handle_me(authorization_header: "str | None") -> Dict[str, Any]:
    """Core request handler, factored out of the HTTP plumbing so it can be
    unit-tested directly without going through BaseHTTPRequestHandler."""
    from api._lib.auth import AuthError, get_current_profile

    try:
        profile = get_current_profile(authorization_header)
    except AuthError as exc:
        raise _ApiError(401, str(exc) or "invalid or missing authorization token")

    return profile


class handler(BaseHTTPRequestHandler):  # noqa: N801 -- Vercel Python runtime convention
    """Vercel Python Serverless Function entrypoint for GET /api/me."""

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        try:
            auth_header = self.headers.get("Authorization")
            result = handle_me(auth_header)
            self._send_json(200, result)
        except _ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive catch-all
            self._send_json(500, {"error": f"internal server error: {exc}"})

    def do_POST(self) -> None:  # noqa: N802
        self._send_json(405, {"error": "method not allowed, use GET"})
