"""
POST /api/rebound/process_lecture

Vercel Python Serverless Function. Uploads a lecture PDF for one of the
authenticated student's own `student_subjects`, and only for a subject that
ScholarTrack has currently flagged at risk (spec section 6's activation
rule), runs the ported Lecture2Revision pipeline (PyMuPDF extraction +
`offline_extractive` revision-kit generation) and persists the results into
the Postgres schema from
`supabase/migrations/20261017000000_rebound_core_schema.sql`
(`lectures`, `learning_materials`, `flashcards`, `quiz_questions`).

Request
-------
Method: POST
Headers:
    Authorization: Bearer <supabase JWT>   (required)
    Content-Type: application/json
Body (JSON):
    {
      "student_subject_id": "<uuid>",   # required - a row in the caller's own student_subjects
      "title": "<string>",              # required - lecture title, also used as the generation topic
      "pdf_base64": "<string>"          # required - base64-encoded PDF bytes (no "data:" prefix)
    }

A base64 JSON body was chosen over multipart/form-data because Vercel's
Python runtime gives you the raw request body and nothing else — handling
multipart here would mean hand-rolling a MIME parser for no real benefit at
hackathon-upload sizes. The client (Next.js) reads the file as
base64 and posts it as JSON.

Response
--------
201 Created (JSON): the generated revision kit, plus the new lecture_id:
    {
      "lecture_id": "<uuid>",
      "summary": "...",
      "revision_notes": "...",
      "key_definitions": [ {"term", "definition", "source_context"}, ... ],
      "exam_questions": [ {"question", "suggested_answer", "difficulty"}, ... ],
      "flashcards": [ {"id", "front_question", "back_answer", "difficulty"}, ... ],
      "quiz_questions": [ {"id", "question_text", "option_a".."option_d",
                            "correct_option", "explanation", "difficulty"}, ... ],
      "backend_used": "Offline Extractive NLP (Grounded, Local Heuristic)"
    }

Error responses (JSON body {"error": "<message>"}):
    400 - missing/invalid body fields, corrupt PDF, or no extractable text
    401 - missing/invalid/expired Authorization bearer token
    403 - student_subject_id is not owned by the caller, OR the subject has
          no current risk_predictions row with is_flagged = true (the
          Lecture2Revision activation rule, spec section 6)
    500 - unexpected server error, or api/_lib (db.py / auth.py, built by a
          parallel workstream) is not yet available in this deployment

Gating rule (spec section 6)
-----------------------------
Lecture2Revision only activates for a subject ScholarTrack has flagged.
`_is_subject_flagged` checks the most recent (highest checkpoint_index)
`risk_predictions` row for the given `student_subject_id` and returns its
`is_flagged` value; a subject with no risk_predictions row at all is treated
as not flagged (120-student seed data confirms every risk_predictions row
currently has is_flagged = true — "not flagged" in practice means "no row").
"""

from __future__ import annotations

import base64
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
LECTURE2REVISION_DIR = REPO_ROOT / "Lecture2Revision"

if str(LECTURE2REVISION_DIR) not in sys.path:
    sys.path.insert(0, str(LECTURE2REVISION_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lecture_processor import (  # noqa: E402
    CorruptPDFError,
    NoExtractableTextError,
    chunk_document,
    extract_text_from_pdf,
)
from transformer_service import generate_learning_materials  # noqa: E402

# api/_lib/{db,auth}.py are built by a parallel workstream. Import
# defensively so this module (and its pure-logic helpers below) stays
# importable and unit-testable before those land; do_POST reports a clear
# 500 if they are genuinely missing at request time.
try:
    from api._lib.db import get_db_connection  # type: ignore
except ImportError:  # pragma: no cover - exercised only before api/_lib lands
    get_db_connection = None  # type: ignore

try:
    from api._lib.auth import AuthError, get_current_profile  # type: ignore
except ImportError:  # pragma: no cover - exercised only before api/_lib lands
    class AuthError(Exception):
        """Stand-in for api._lib.auth.AuthError until that module lands."""

    get_current_profile = None  # type: ignore


REVISION_BACKEND = "offline_extractive"


class RequestError(Exception):
    """Raised for any client-facing error; carries the HTTP status to use."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# ==============================================================================
# Pure logic helpers (independently unit-testable, no HTTP involved)
# ==============================================================================

def _decode_pdf_base64(pdf_base64: str) -> bytes:
    """Decodes a base64-encoded PDF payload, raising RequestError(400) on bad input."""
    if not pdf_base64 or not isinstance(pdf_base64, str):
        raise RequestError(400, "pdf_base64 is required and must be a non-empty string.")
    # Tolerate an accidental data: URL prefix from a naive client.
    if pdf_base64.startswith("data:") and "," in pdf_base64:
        pdf_base64 = pdf_base64.split(",", 1)[1]
    try:
        pdf_bytes = base64.b64decode(pdf_base64, validate=False)
    except Exception as exc:
        raise RequestError(400, f"pdf_base64 could not be decoded: {exc}") from exc
    if not pdf_bytes:
        raise RequestError(400, "Decoded PDF is empty.")
    return pdf_bytes


def _is_subject_flagged(student_subject_id: str, db_conn: Any) -> bool:
    """
    Returns True iff the most recent risk_predictions row (by checkpoint_index)
    for this student_subject_id has is_flagged = true. A subject with no
    risk_predictions row at all is not flagged.
    """
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            select is_flagged
            from risk_predictions
            where student_subject_id = %s
            order by checkpoint_index desc, predicted_at desc
            limit 1
            """,
            (student_subject_id,),
        )
        row = cur.fetchone()
    finally:
        cur.close()
    if row is None:
        return False
    return bool(row[0])


def _owns_student_subject(student_subject_id: str, profile: Dict[str, Any]) -> bool:
    """Checks student_subject_id against the authenticated student's own student_subjects rows."""
    for row in profile.get("student_subjects") or []:
        if str(row.get("id")) == str(student_subject_id):
            return True
    return False


def _extract_and_chunk(pdf_bytes: bytes) -> Dict[str, Any]:
    """Runs PyMuPDF extraction + chunking; raises RequestError(400) on bad PDFs."""
    try:
        cleaned_text, page_records = extract_text_from_pdf(pdf_bytes)
    except (NoExtractableTextError, CorruptPDFError) as exc:
        raise RequestError(400, str(exc)) from exc

    chunks = chunk_document(cleaned_text)
    word_count = len(cleaned_text.split())
    return {
        "cleaned_text": cleaned_text,
        "page_records": page_records,
        "chunks": chunks,
        "word_count": word_count,
        "chunk_count": max(1, len(chunks)),
    }


def _generate_revision_kit(title: str, cleaned_text: str) -> Dict[str, Any]:
    """Calls the ported transformer_service with the offline_extractive backend."""
    parsed, backend_used_label = generate_learning_materials(
        lecture_topic=title,
        lecture_text=cleaned_text,
        backend=REVISION_BACKEND,
    )
    parsed = dict(parsed)
    parsed["backend_used"] = backend_used_label
    return parsed


def _insert_revision_kit(
    db_conn: Any,
    student_subject_id: str,
    title: str,
    filename: str,
    extraction: Dict[str, Any],
    kit: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inserts the lecture + generated kit into lectures / learning_materials /
    flashcards / quiz_questions per the real migration column names, and
    returns the response payload (with DB-assigned ids).
    """
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            insert into lectures
                (student_subject_id, title, topic, filename, file_path,
                 raw_text, cleaned_text, word_count, chunk_count)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            returning id
            """,
            (
                student_subject_id,
                title,
                title,
                filename,
                f"rebound/{student_subject_id}/{filename}",
                extraction["cleaned_text"],
                extraction["cleaned_text"],
                extraction["word_count"],
                extraction["chunk_count"],
            ),
        )
        lecture_id = cur.fetchone()[0]

        cur.execute(
            """
            insert into learning_materials
                (lecture_id, summary, revision_notes, key_definitions,
                 exam_questions, source_grounding_notes, backend_used)
            values (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                lecture_id,
                kit["summary"],
                kit["revision_notes"],
                json.dumps(kit["key_definitions"]),
                json.dumps(kit["exam_questions"]),
                "Generated strictly from the uploaded lecture text (offline_extractive).",
                kit["backend_used"],
            ),
        )

        flashcards_out: List[Dict[str, Any]] = []
        for fc in kit["flashcards"]:
            cur.execute(
                """
                insert into flashcards
                    (lecture_id, topic, front_question, back_answer, difficulty)
                values (%s, %s, %s, %s, %s)
                returning id
                """,
                (lecture_id, title, fc["front_question"], fc["back_answer"], fc.get("difficulty", "Medium")),
            )
            fc_id = cur.fetchone()[0]
            flashcards_out.append({**fc, "id": str(fc_id)})

        quiz_questions_out: List[Dict[str, Any]] = []
        for qq in kit["quiz_questions"]:
            cur.execute(
                """
                insert into quiz_questions
                    (lecture_id, topic, question_text, option_a, option_b,
                     option_c, option_d, correct_option, explanation, difficulty)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                returning id
                """,
                (
                    lecture_id,
                    title,
                    qq["question_text"],
                    qq["option_a"],
                    qq["option_b"],
                    qq["option_c"],
                    qq["option_d"],
                    qq["correct_option"],
                    qq["explanation"],
                    qq.get("difficulty", "Medium"),
                ),
            )
            qq_id = cur.fetchone()[0]
            quiz_questions_out.append({**qq, "id": str(qq_id)})

        db_conn.commit()
    except Exception:
        db_conn.rollback()
        raise
    finally:
        cur.close()

    return {
        "lecture_id": str(lecture_id),
        "summary": kit["summary"],
        "revision_notes": kit["revision_notes"],
        "key_definitions": kit["key_definitions"],
        "exam_questions": kit["exam_questions"],
        "flashcards": flashcards_out,
        "quiz_questions": quiz_questions_out,
        "backend_used": kit["backend_used"],
    }


def process_lecture_request(
    authorization_header: Optional[str],
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Orchestrates one process_lecture request end to end: auth, ownership +
    gating checks, PDF extraction, generation, and DB persistence. Raises
    RequestError for any client-facing failure. Used directly by do_POST, and
    independently testable once api/_lib lands.
    """
    if get_current_profile is None or get_db_connection is None:
        raise RequestError(500, "api/_lib is not available yet in this deployment.")

    if not authorization_header:
        raise RequestError(401, "Missing Authorization header.")

    try:
        profile = get_current_profile(authorization_header)
    except AuthError as exc:
        raise RequestError(401, str(exc)) from exc

    student_subject_id = body.get("student_subject_id")
    title = body.get("title")
    pdf_base64 = body.get("pdf_base64")

    if not student_subject_id or not isinstance(student_subject_id, str):
        raise RequestError(400, "student_subject_id is required.")
    if not title or not isinstance(title, str):
        raise RequestError(400, "title is required.")

    if not _owns_student_subject(student_subject_id, profile):
        raise RequestError(403, "student_subject_id does not belong to the authenticated student.")

    pdf_bytes = _decode_pdf_base64(pdf_base64)

    conn = get_db_connection()
    try:
        if not _is_subject_flagged(student_subject_id, conn):
            raise RequestError(
                403,
                "Lecture2Revision is only available once ScholarTrack has flagged this subject at risk.",
            )

        extraction = _extract_and_chunk(pdf_bytes)
        kit = _generate_revision_kit(title, extraction["cleaned_text"])
        result = _insert_revision_kit(
            conn, student_subject_id, title, f"{title}.pdf", extraction, kit
        )
    finally:
        conn.close()

    return result


# ==============================================================================
# Vercel Python Serverless Function entrypoint
# ==============================================================================

class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except json.JSONDecodeError as exc:
                raise RequestError(400, f"Request body is not valid JSON: {exc}") from exc

            auth_header = self.headers.get("Authorization")
            result = process_lecture_request(auth_header, body)
            self._send_json(201, result)
        except RequestError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - last-resort safety net
            self._send_json(500, {"error": f"Internal server error: {exc}"})

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
