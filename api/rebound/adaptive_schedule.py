"""
Rebound AI -- GET /api/rebound/adaptive_schedule

Vercel Python Serverless Function. Returns the authenticated student's
adaptive Lecture2Revision revision schedule, but ONLY for subjects
ScholarTrack currently flags at risk (spec section 1/6: Lecture2Revision only
activates for flagged subjects -- a subject in good standing never gets a
schedule entry here).

Ported from Lecture2Revision/adaptive_engine.py:
    - calculate_topic_priority(...)      -- UNCHANGED rule-based + optional
                                             LSTM blend.
    - generate_adaptive_schedule(...)    -- re-implemented against Postgres
                                             (the real schema, joined through
                                             student_subjects) instead of the
                                             original's SQLite queries. The
                                             original grouped by lecture
                                             "topic" within a single demo
                                             student; here, since the
                                             activation/gating unit is the
                                             *subject* (student_subjects
                                             row), inputs are aggregated per
                                             subject across all of that
                                             subject's lectures/flashcards/
                                             quiz attempts.

New risk-boost term (spec section 1/6 -- the actual ScholarTrack <->
Lecture2Revision integration point):
    calculate_topic_priority already optionally blends in an LSTM
    `final_priority = base*0.70 + lstm_urgency*0.30`. We add the subject's
    ScholarTrack `risk_predictions.risk_probability` (0.0-1.0, from the most
    recent checkpoint) as an ADDITIONAL boost on top of that existing blend,
    using the exact same 70/30 ratio so both signals are combined
    consistently rather than inventing a new, arbitrary weighting:

        final_priority = final_priority * 0.7 + (risk_probability * 100) * 0.3

    See `apply_risk_boost` below -- it's factored out standalone so it can be
    unit-tested with hand-picked values independent of the DB/HTTP plumbing.

Request:
    GET /api/rebound/adaptive_schedule
    Headers: Authorization: Bearer <supabase JWT>

Response 200 (JSON): a list, sorted descending by priority_score, of:
    {
        "subject_slug": str,
        "subject_name": str,
        "student_subject_id": str,
        "priority_score": float,        # 0-100, risk-boosted
        "priority_reason": str,         # includes the ScholarTrack risk line
        "scheduled_date": "YYYY-MM-DD",
        "risk_probability": float,      # 0.0-1.0, the raw ScholarTrack signal
        "recent_score": float,
        "days_overdue": float,
        "cards_needing_practice": int,
        "lstm_predicted_score": float | None,
        "status": "pending",
    }

Error responses (JSON body {"error": "<message>"}):
    401 - missing/invalid/expired bearer token
    403 - caller is not a student
    405 - method other than GET
    500 - unexpected server error
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# --- make Lecture2Revision's (unmodified) adaptive-engine math importable --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_L2R_DIR = _REPO_ROOT / "Lecture2Revision"
if str(_L2R_DIR) not in sys.path:
    sys.path.insert(0, str(_L2R_DIR))

from adaptive_engine import calculate_topic_priority  # noqa: E402
from config import SM2_DEFAULT_EASE_FACTOR  # noqa: E402

# --- shared backend helpers (built by another agent in parallel) ----------
# Imported lazily inside handle_adaptive_schedule() rather than at module
# level: api/_lib/db.py and api/_lib/auth.py are being built concurrently by
# another agent and may not exist yet. A deferred import also lets tests
# inject fake versions via sys.modules without needing the real files
# present at collection time, and lets apply_risk_boost (a pure function)
# be imported and tested in isolation regardless of that work landing.


class _ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def apply_risk_boost(final_priority: float, risk_probability: float) -> float:
    """Blends ScholarTrack's risk_probability (0.0-1.0) into an
    already-computed calculate_topic_priority score, using the same 70/30
    ratio calculate_topic_priority itself uses for its rule-based/LSTM blend,
    so the two boosts compose consistently instead of using a second,
    arbitrary weighting.

        final_priority = final_priority * 0.7 + (risk_probability * 100) * 0.3

    Clamped to [0, 100] and rounded to 1 decimal place, matching
    calculate_topic_priority's own output convention.
    """
    boosted = (final_priority * 0.7) + ((risk_probability * 100.0) * 0.3)
    return round(min(100.0, max(0.0, boosted)), 1)


def _fetch_flagged_subjects(cursor, student_subject_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Returns {student_subject_id: {"checkpoint_index", "risk_probability"}}
    for the subset of the given subjects whose MOST RECENT risk_predictions
    row (highest checkpoint_index) has is_flagged = true."""
    if not student_subject_ids:
        return {}

    cursor.execute(
        """
        select distinct on (student_subject_id)
            student_subject_id, checkpoint_index, risk_probability, is_flagged
        from risk_predictions
        where student_subject_id = any(%s)
        order by student_subject_id, checkpoint_index desc
        """,
        (student_subject_ids,),
    )
    rows = cursor.fetchall()

    flagged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            subj_id, checkpoint_index, risk_probability, is_flagged = (
                row["student_subject_id"],
                row["checkpoint_index"],
                row["risk_probability"],
                row["is_flagged"],
            )
        else:
            subj_id, checkpoint_index, risk_probability, is_flagged = row

        if is_flagged:
            flagged[subj_id] = {
                "checkpoint_index": checkpoint_index,
                "risk_probability": risk_probability,
            }
    return flagged


def _gather_topic_inputs(cursor, student_id: str, student_subject_id: str) -> Dict[str, Any]:
    """Pulls calculate_topic_priority's raw inputs for one subject, aggregated
    across all of that subject's lectures. Falls back to the original
    generate_adaptive_schedule's own documented neutral defaults for a
    brand-new topic when there's no quiz/flashcard history yet."""
    now_utc = datetime.now(timezone.utc)

    # Most recent quiz attempt across any lecture under this subject.
    cursor.execute(
        """
        select qa.score_pct, qa.incorrect_count, qa.completed_at
        from quiz_attempts qa
        join lectures l on l.id = qa.lecture_id
        where qa.student_id = %s and l.student_subject_id = %s
        order by qa.completed_at desc
        limit 1
        """,
        (student_id, student_subject_id),
    )
    attempt = cursor.fetchone()

    if attempt:
        if isinstance(attempt, dict):
            recent_score = attempt["score_pct"]
            recent_incorrect = attempt["incorrect_count"]
            completed_at = attempt["completed_at"]
        else:
            recent_score, recent_incorrect, completed_at = attempt
        days_since_last_review = max(0.0, (now_utc - completed_at).total_seconds() / 86400.0)
    else:
        # Original's own documented neutral defaults for a topic with no
        # quiz attempts yet (adaptive_engine.py generate_adaptive_schedule).
        recent_score = 50.0
        recent_incorrect = 0
        days_since_last_review = 2.0

    # Flashcards needing practice across any lecture under this subject.
    cursor.execute(
        """
        select count(*) as cnt
        from flashcards f
        join lectures l on l.id = f.lecture_id
        where l.student_subject_id = %s and f.needs_practice = true
        """,
        (student_subject_id,),
    )
    cnt_row = cursor.fetchone()
    cards_needing_practice = cnt_row["cnt"] if isinstance(cnt_row, dict) else cnt_row[0]

    # Latest existing revision_schedules record for this subject.
    cursor.execute(
        """
        select scheduled_date, sm2_interval_days, sm2_ease_factor, sm2_repetitions
        from revision_schedules
        where student_id = %s and student_subject_id = %s
        order by created_at desc
        limit 1
        """,
        (student_id, student_subject_id),
    )
    sched = cursor.fetchone()

    if sched:
        if isinstance(sched, dict):
            scheduled_date, sm2_interval, sm2_ease, sm2_reps = (
                sched["scheduled_date"],
                sched["sm2_interval_days"],
                sched["sm2_ease_factor"],
                sched["sm2_repetitions"],
            )
        else:
            scheduled_date, sm2_interval, sm2_ease, sm2_reps = sched
        sched_dt = datetime(
            scheduled_date.year, scheduled_date.month, scheduled_date.day, tzinfo=timezone.utc
        )
        days_overdue = max(0.0, (now_utc - sched_dt).total_seconds() / 86400.0)
    else:
        days_overdue = 0.0
        sm2_interval = 1.0
        sm2_ease = SM2_DEFAULT_EASE_FACTOR
        sm2_reps = 0

    return {
        "recent_accuracy": recent_score,
        "recent_incorrect_count": recent_incorrect,
        "days_since_last_review": days_since_last_review,
        "days_overdue": days_overdue,
        "cards_needing_practice": cards_needing_practice,
        "sm2_interval": sm2_interval,
        "sm2_ease": sm2_ease,
        "sm2_reps": sm2_reps,
    }


def handle_adaptive_schedule(authorization_header: Optional[str]) -> List[Dict[str, Any]]:
    """Core request handler, factored out of the HTTP plumbing so it can be
    unit-tested directly without going through BaseHTTPRequestHandler."""
    from api._lib.auth import AuthError, get_current_profile
    from api._lib.db import get_db_connection

    try:
        profile = get_current_profile(authorization_header)
    except AuthError as exc:
        raise _ApiError(401, str(exc) or "invalid or missing authorization token")

    if profile.get("role") != "student" or not profile.get("student_id"):
        raise _ApiError(403, "only students have an adaptive revision schedule")

    student_id = profile["student_id"]
    subjects = profile.get("student_subjects", [])
    if not subjects:
        return []

    subject_by_id = {s["id"]: s for s in subjects}
    now_utc = datetime.now(timezone.utc)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        flagged = _fetch_flagged_subjects(cursor, list(subject_by_id.keys()))

        plan: List[Dict[str, Any]] = []
        for student_subject_id, risk_row in flagged.items():
            subject = subject_by_id[student_subject_id]
            risk_probability = risk_row["risk_probability"]

            inputs = _gather_topic_inputs(cursor, student_id, student_subject_id)

            priority_score, priority_reason = calculate_topic_priority(
                recent_accuracy=inputs["recent_accuracy"],
                recent_incorrect_count=inputs["recent_incorrect_count"],
                days_since_last_review=inputs["days_since_last_review"],
                days_overdue=inputs["days_overdue"],
                cards_needing_practice=inputs["cards_needing_practice"],
                lstm_predicted_score=None,
            )

            boosted_priority = apply_risk_boost(priority_score, risk_probability)
            boosted_reason = (
                f"{priority_reason} | ScholarTrack flags this subject at "
                f"{risk_probability:.0%} risk"
            )

            recent_score = inputs["recent_accuracy"]
            if recent_score >= 80.0:
                suggested_interval = max(2.0, inputs["sm2_interval"] * 1.5)
            elif recent_score >= 60.0:
                suggested_interval = max(1.0, inputs["sm2_interval"])
            else:
                suggested_interval = 1.0

            scheduled_date = (now_utc + timedelta(days=suggested_interval)).date()

            cursor.execute(
                """
                insert into revision_schedules (
                    student_id, student_subject_id, topic, priority_score,
                    priority_reason, scheduled_date, status,
                    sm2_interval_days, sm2_ease_factor, sm2_repetitions,
                    lstm_predicted_score
                ) values (%s, %s, %s, %s, %s, %s, 'pending', %s, %s, %s, %s)
                """,
                (
                    student_id,
                    student_subject_id,
                    subject["subject_name"],
                    boosted_priority,
                    boosted_reason,
                    scheduled_date,
                    suggested_interval,
                    inputs["sm2_ease"],
                    inputs["sm2_reps"],
                    None,
                ),
            )

            plan.append(
                {
                    "subject_slug": subject["subject_slug"],
                    "subject_name": subject["subject_name"],
                    "student_subject_id": student_subject_id,
                    "priority_score": boosted_priority,
                    "priority_reason": boosted_reason,
                    "scheduled_date": scheduled_date.strftime("%Y-%m-%d"),
                    "risk_probability": risk_probability,
                    "recent_score": recent_score,
                    "days_overdue": round(inputs["days_overdue"], 1),
                    "cards_needing_practice": inputs["cards_needing_practice"],
                    "lstm_predicted_score": None,
                    "status": "pending",
                }
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    plan.sort(key=lambda x: x["priority_score"], reverse=True)
    return plan


class handler(BaseHTTPRequestHandler):  # noqa: N801 -- Vercel Python runtime convention
    """Vercel Python Serverless Function entrypoint for GET /api/rebound/adaptive_schedule."""

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        try:
            # path/query are irrelevant here (the schedule is entirely scoped
            # by the caller's own auth'd profile), but parse defensively so
            # trailing query strings never 500.
            urlparse(self.path)
            auth_header = self.headers.get("Authorization")
            result = handle_adaptive_schedule(auth_header)
            self._send_json(200, result)
        except _ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive catch-all
            self._send_json(500, {"error": f"internal server error: {exc}"})

    def do_POST(self) -> None:  # noqa: N802
        self._send_json(405, {"error": "method not allowed, use GET"})
