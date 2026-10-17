"""
Rebound AI -- POST /api/predict_risk

Vercel Python Serverless Function. ScholarTrack's live risk-prediction
endpoint (spec section 5/9): computes the 10-feature vector for one
student-subject from its live `assessment_checkpoints` rows, runs the
deployed NumPy logistic-regression model
(`scripts/inference/numpy_logreg_exported.npz`), UPSERTs the result into
`risk_predictions`, and returns it.

Request:
    POST /api/predict_risk
    Headers: Authorization: Bearer <supabase JWT>
    Body (JSON): {"student_subject_id": "<uuid>"}

Response 200 (JSON): the prediction, as stored:
    {
        "id": "<uuid>",
        "student_subject_id": "<uuid>",
        "checkpoint_index": <int>,
        "risk_probability": <float>,
        "is_flagged": <bool>,
        "is_failing": <bool>,
        "top_risk_factors": [...],
        "protective_factors": [...],
        "human_readable_summary": "<str>",
        "model_version": "logreg_v1"
    }

Error responses (JSON body {"error": "<message>"}):
    400 - malformed body / missing student_subject_id
    401 - missing/invalid/expired bearer token
    403 - student_subject_id does not belong to the caller's own student_subjects
    404 - student_subject_id not found / has no assessment_checkpoints yet
    405 - method other than POST
    500 - unexpected server error

Auth/scoping note (spec section 8): only the student path is enforced here.
A student may request a prediction only for one of their OWN
`student_subjects` (checked against `get_current_profile`'s
`student_subjects` list). Professor/HOD/Principal callers are not yet
wired up to a scoped read path (see `api/_lib/auth.py`), so they are
rejected the same way an unrecognized role would be: 403.

is_failing: recomputed here from the live features via
`api/_lib/scholartrack.py::compute_is_failing_so_far` rather than read back
from any previously seeded row -- see that function's docstring for why
this is a documented approximation (ETE marks are never stored, by
design, so the real 40%-of-full-aggregate fact cannot be computed from a
live request; this uses the same no-ETE "aggregate so far" quantity the
marks_deficit_margin feature already computes, re-thresholded at 40
instead of 70).
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict

MODEL_VERSION = "logreg_v1"


class _ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _load_student_subject_checkpoint_count(cursor, student_subject_id: str) -> int:
    """Returns how many unit_test/submission checkpoints exist for this
    student_subject (used only to confirm the row exists at all before
    attempting feature extraction -- compute_features_as_of raises its own
    ValueError on zero rows, which this turns into a clean 404)."""
    cursor.execute(
        "select count(*) from public.assessment_checkpoints where student_subject_id = %s",
        (student_subject_id,),
    )
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def handle_predict_risk(authorization_header: "str | None", body: Dict[str, Any]) -> Dict[str, Any]:
    """Core request handler, factored out of the HTTP plumbing so it can be
    unit-tested directly without going through BaseHTTPRequestHandler."""
    from api._lib.auth import AuthError, get_current_profile
    from api._lib.db import get_db_connection
    from api._lib.scholartrack import compute_features_as_of, compute_is_failing_so_far, predict_risk

    try:
        profile = get_current_profile(authorization_header)
    except AuthError as exc:
        raise _ApiError(401, str(exc) or "invalid or missing authorization token")

    student_subject_id = body.get("student_subject_id")
    if not student_subject_id or not isinstance(student_subject_id, str):
        raise _ApiError(400, "student_subject_id is required")

    owned_subjects = {s["id"]: s for s in profile.get("student_subjects", [])}
    if student_subject_id not in owned_subjects:
        raise _ApiError(403, "student_subject_id does not belong to one of your own subjects")

    subject = owned_subjects[student_subject_id]

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        if _load_student_subject_checkpoint_count(cursor, student_subject_id) == 0:
            raise _ApiError(404, "no assessment_checkpoints found for this student_subject_id")

        features = compute_features_as_of(student_subject_id, conn)
        features_with_subject = {**features, "subject_slug": subject.get("subject_slug")}

        prediction = predict_risk(features_with_subject)
        is_failing = compute_is_failing_so_far(features)

        # checkpoint_index mirrors scripts/feature_engineering/extraction.py's
        # prefix_len (2..7), which compute_features_as_of derives from the
        # count of unit_test/submission rows (attendance_week rows don't
        # count toward it) -- recompute the same way here rather than have
        # compute_features_as_of return it as a side channel.
        cursor.execute(
            "select count(*) from public.assessment_checkpoints "
            "where student_subject_id = %s and checkpoint_type in ('unit_test', 'submission')",
            (student_subject_id,),
        )
        checkpoint_index = max(2, min(7, int(cursor.fetchone()[0])))

        # UPSERT by (student_subject_id, checkpoint_index): the schema has no
        # unique constraint on that pair (unlike supabase/seed.py's
        # deterministic-id ON CONFLICT upsert), so this does the equivalent
        # explicitly -- select the existing row for this as-of checkpoint, if
        # any, and UPDATE it in place rather than accumulating a new row on
        # every re-prediction of the same checkpoint. A genuinely new
        # checkpoint_index (more assessment data has landed since the last
        # call) still gets its own new row, which is the intended history.
        cursor.execute(
            "select id from public.risk_predictions "
            "where student_subject_id = %s and checkpoint_index = %s",
            (student_subject_id, checkpoint_index),
        )
        existing = cursor.fetchone()

        params = (
            prediction["risk_probability"],
            prediction["is_flagged"],
            is_failing,
            json.dumps(prediction["top_risk_factors"]),
            json.dumps(prediction["protective_factors"]),
            prediction["human_readable_summary"],
            MODEL_VERSION,
        )
        if existing is not None:
            cursor.execute(
                """
                update public.risk_predictions
                set risk_probability = %s, is_flagged = %s, is_failing = %s,
                    top_risk_factors = %s::jsonb, protective_factors = %s::jsonb,
                    human_readable_summary = %s, model_version = %s, predicted_at = now()
                where id = %s
                returning id, predicted_at
                """,
                params + (existing[0],),
            )
        else:
            cursor.execute(
                """
                insert into public.risk_predictions
                    (student_subject_id, checkpoint_index, risk_probability, is_flagged,
                     is_failing, top_risk_factors, protective_factors, human_readable_summary,
                     model_version)
                values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                returning id, predicted_at
                """,
                (student_subject_id, checkpoint_index) + params,
            )
        new_id, predicted_at = cursor.fetchone()
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
        "id": str(new_id),
        "student_subject_id": student_subject_id,
        "checkpoint_index": checkpoint_index,
        "risk_probability": prediction["risk_probability"],
        "is_flagged": prediction["is_flagged"],
        "is_failing": is_failing,
        "top_risk_factors": prediction["top_risk_factors"],
        "protective_factors": prediction["protective_factors"],
        "human_readable_summary": prediction["human_readable_summary"],
        "model_version": MODEL_VERSION,
        "predicted_at": str(predicted_at),
    }


class handler(BaseHTTPRequestHandler):  # noqa: N801 -- Vercel Python runtime convention
    """Vercel Python Serverless Function entrypoint for POST /api/predict_risk."""

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
            result = handle_predict_risk(auth_header, body)
            self._send_json(200, result)
        except _ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive catch-all
            self._send_json(500, {"error": f"internal server error: {exc}"})

    def do_GET(self) -> None:  # noqa: N802
        self._send_json(405, {"error": "method not allowed, use POST"})
