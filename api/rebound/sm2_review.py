"""
Rebound AI -- POST /api/rebound/sm2_review

Vercel Python Serverless Function. Records a flashcard review ('remembered' /
'needs_practice') and advances that flashcard's SM-2 spaced-repetition state.

Ported from Lecture2Revision/flashcard_service.py::calculate_sm2_update (the
SM-2 math itself is UNCHANGED -- see spec section 6). What's new here is the
Postgres/Supabase plumbing: auth, ownership check, and persisting the updated
state to the real `flashcards` / `flashcard_reviews` tables (Postgres-ified
schema, see supabase/migrations/20261017000000_rebound_core_schema.sql)
instead of the original SQLite tables.

Request:
    POST /api/rebound/sm2_review
    Headers: Authorization: Bearer <supabase JWT>
    Body (JSON): {"flashcard_id": "<uuid>", "rating": "remembered" | "needs_practice"}

Response 200 (JSON): the flashcard's updated SM-2 state:
    {
        "flashcard_id": "<uuid>",
        "interval_days": <float>,
        "ease_factor": <float>,
        "repetitions": <int>,
        "next_review_date": "YYYY-MM-DD",
        "needs_practice": <bool>
    }

Error responses (JSON body {"error": "<message>"}):
    400 - malformed body / missing fields / invalid rating
    401 - missing/invalid/expired bearer token
    403 - flashcard does not belong to one of the caller's own student_subjects
    404 - flashcard_id not found
    405 - method other than POST
    500 - unexpected server error
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict

# --- make Lecture2Revision's (unmodified) SM-2 module importable ----------
# Lecture2Revision/flashcard_service.py does `from config import ...`, so we
# add the Lecture2Revision/ directory itself to sys.path (not just its
# parent), which also makes its sibling config.py resolvable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_L2R_DIR = _REPO_ROOT / "Lecture2Revision"
if str(_L2R_DIR) not in sys.path:
    sys.path.insert(0, str(_L2R_DIR))

from flashcard_service import calculate_sm2_update  # noqa: E402

# --- shared backend helpers (built by another agent in parallel) ----------
# Imported lazily inside handle_sm2_review() rather than at module level:
# api/_lib/db.py and api/_lib/auth.py are being built concurrently by
# another agent and may not exist yet. A deferred import also lets tests
# inject fake versions via sys.modules without needing the real files
# present at collection time.

VALID_RATINGS = {"remembered", "needs_practice"}


class _ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _load_flashcard_for_update(cursor, flashcard_id: str) -> Dict[str, Any]:
    """Fetches the flashcard's current SM-2 state plus the student_subject_id
    it belongs to (via its lecture), for the ownership check."""
    cursor.execute(
        """
        select f.id, f.interval_days, f.ease_factor, f.repetitions,
               l.student_subject_id
        from flashcards f
        join lectures l on l.id = f.lecture_id
        where f.id = %s
        """,
        (flashcard_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise _ApiError(404, "flashcard not found")

    # Support both tuple-cursors and RealDictCursor-style row mappings.
    if isinstance(row, dict):
        return row
    return {
        "id": row[0],
        "interval_days": row[1],
        "ease_factor": row[2],
        "repetitions": row[3],
        "student_subject_id": row[4],
    }


def handle_sm2_review(authorization_header: str | None, body: Dict[str, Any]) -> Dict[str, Any]:
    """Core request handler, factored out of the HTTP plumbing so it can be
    unit-tested directly without going through BaseHTTPRequestHandler."""
    from api._lib.auth import AuthError, get_current_profile
    from api._lib.db import get_db_connection

    try:
        profile = get_current_profile(authorization_header)
    except AuthError as exc:
        raise _ApiError(401, str(exc) or "invalid or missing authorization token")

    flashcard_id = body.get("flashcard_id")
    rating = body.get("rating")

    if not flashcard_id or not isinstance(flashcard_id, str):
        raise _ApiError(400, "flashcard_id is required")
    if rating not in VALID_RATINGS:
        raise _ApiError(400, "rating must be 'remembered' or 'needs_practice'")

    student_id = profile.get("student_id")
    owned_subject_ids = {s["id"] for s in profile.get("student_subjects", [])}
    if not student_id:
        raise _ApiError(403, "only students can review flashcards")

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        flashcard = _load_flashcard_for_update(cursor, flashcard_id)

        if flashcard["student_subject_id"] not in owned_subject_ids:
            raise _ApiError(403, "this flashcard does not belong to one of your subjects")

        update = calculate_sm2_update(
            current_interval=flashcard["interval_days"],
            current_ease=flashcard["ease_factor"],
            current_repetitions=flashcard["repetitions"],
            rating=rating,
        )

        needs_practice = rating == "needs_practice"

        cursor.execute(
            """
            update flashcards
            set interval_days = %s,
                ease_factor = %s,
                repetitions = %s,
                next_review_date = %s,
                needs_practice = %s,
                last_reviewed_at = now()
            where id = %s
            """,
            (
                update["interval_days"],
                update["ease_factor"],
                update["repetitions"],
                update["next_review_date"],
                needs_practice,
                flashcard_id,
            ),
        )

        cursor.execute(
            """
            insert into flashcard_reviews (flashcard_id, student_id, rating)
            values (%s, %s, %s)
            """,
            (flashcard_id, student_id, rating),
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

    return {
        "flashcard_id": flashcard_id,
        "interval_days": update["interval_days"],
        "ease_factor": update["ease_factor"],
        "repetitions": update["repetitions"],
        "next_review_date": update["next_review_date"],
        "needs_practice": needs_practice,
    }


class handler(BaseHTTPRequestHandler):  # noqa: N801 -- Vercel Python runtime convention
    """Vercel Python Serverless Function entrypoint for POST /api/rebound/sm2_review."""

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
            result = handle_sm2_review(auth_header, body)
            self._send_json(200, result)
        except _ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive catch-all
            self._send_json(500, {"error": f"internal server error: {exc}"})

    def do_GET(self) -> None:  # noqa: N802
        self._send_json(405, {"error": "method not allowed, use POST"})
