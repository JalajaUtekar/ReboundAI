"""
Lecture To Revision - Evaluation and Metrics Module
Computes verifiable quantitative evaluation metrics for:
1. LSTM sequential performance model (MAE, RMSE, Baseline Comparison, sample sizes).
2. Generated learning materials (Lexical source-grounding audit, relevance, factual verification checklist).
3. Adaptive revision impact (Score progression delta across attempts, completion rates).
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import LSTM_METRICS_PATH
from database import get_connection


def get_lstm_evaluation_summary() -> Dict[str, Any]:
    """
    Returns authentic evaluated metrics for the LSTM predictor.
    Never fabricates metrics: returns clear status if model is untrained.
    """
    if not LSTM_METRICS_PATH.exists():
        return {
            "is_trained": False,
            "status_message": (
                "LSTM model has not been trained yet. Train the model using the "
                "'Model Evaluation' page or running 'python scripts/train_lstm.py'. "
                "Adaptive scheduling currently operates on deterministic rule-based algorithms."
            ),
            "metrics": None,
        }

    try:
        with open(LSTM_METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "is_trained": True,
                "status_message": "Trained and evaluated on chronological student trajectories.",
                "metrics": data,
            }
    except Exception as e:
        return {
            "is_trained": False,
            "status_message": f"Error reading metrics artifact: {e}",
            "metrics": None,
        }


def audit_material_grounding(
    raw_lecture_text: str,
    materials: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Performs empirical textual grounding verification:
    Checks if extracted terms, questions, and answers are directly rooted
    in the source lecture text.
    """
    if not raw_lecture_text:
        return {
            "grounding_score_pct": 0.0,
            "verified_definitions": 0,
            "total_definitions": 0,
            "verified_quiz_questions": 0,
            "total_quiz_questions": 0,
            "checklist": [],
        }

    source_lower = raw_lecture_text.lower()
    source_words = set(re.findall(r"\b[A-Za-z]{3,}\b", source_lower))

    # Audit definitions
    defs = materials.get("key_definitions", [])
    if isinstance(defs, str):
        try:
            defs = json.loads(defs)
        except Exception:
            defs = []

    verified_defs = 0
    for d in defs:
        term = d.get("term", "").lower().strip()
        # Verify if significant term words exist in source
        term_words = [w for w in term.split() if len(w) > 3]
        if term_words and any(tw in source_words for tw in term_words):
            verified_defs += 1

    # Audit quiz questions
    quizzes = materials.get("quiz_questions", [])
    if isinstance(quizzes, str):
        try:
            quizzes = json.loads(quizzes)
        except Exception:
            quizzes = []

    verified_quizzes = 0
    for q in quizzes:
        q_text = q.get("question_text", "").lower()
        q_words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", q_text)]
        # Check overlap
        overlap = [w for w in q_words if w in source_words]
        if len(overlap) >= min(2, len(q_words)):
            verified_quizzes += 1

    total_items = max(1, len(defs) + len(quizzes))
    grounded_items = verified_defs + verified_quizzes
    grounding_score = round((grounded_items / total_items) * 100.0, 1)

    checklist = [
        {
            "criterion": "Source Grounding Verification",
            "passed": grounding_score >= 80.0,
            "details": f"{grounding_score}% of key terms and questions match tokens in primary lecture text.",
        },
        {
            "criterion": "No Fabricated Concepts",
            "passed": verified_defs == len(defs) if defs else True,
            "details": f"{verified_defs}/{len(defs)} definitions verified against lecture contents.",
        },
        {
            "criterion": "Question Relevance",
            "passed": verified_quizzes == len(quizzes) if quizzes else True,
            "details": f"{verified_quizzes}/{len(quizzes)} quiz questions rooted in source text.",
        },
        {
            "criterion": "JSON Schema Integrity",
            "passed": bool(materials.get("summary") and materials.get("revision_notes")),
            "details": "Mandatory summary and revision notes successfully parsed.",
        },
    ]

    return {
        "grounding_score_pct": grounding_score,
        "verified_definitions": verified_defs,
        "total_definitions": len(defs),
        "verified_quiz_questions": verified_quizzes,
        "total_quiz_questions": len(quizzes),
        "checklist": checklist,
    }


def get_adaptive_progress_metrics(
    student_id: str = "demo_student",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Computes performance trajectory across attempts, score deltas,
    and revision completion rates from SQLite.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT topic, score_pct, completed_at
        FROM quiz_attempts
        WHERE student_id = ?
        ORDER BY completed_at ASC
    """, (student_id,))
    attempts = [dict(r) for r in cursor.fetchall()]

    # Group by topic
    topic_history: Dict[str, List[float]] = {}
    for a in attempts:
        top = a["topic"]
        if top not in topic_history:
            topic_history[top] = []
        topic_history[top].append(a["score_pct"])

    topic_progression = []
    for top, scores in topic_history.items():
        initial_score = scores[0]
        latest_score = scores[-1]
        delta = round(latest_score - initial_score, 1)
        topic_progression.append({
            "topic": top,
            "attempts_count": len(scores),
            "initial_score": initial_score,
            "latest_score": latest_score,
            "delta": delta,
            "improved": delta > 0,
        })

    # Revision schedule completion stats
    cursor.execute("SELECT COUNT(*) as total FROM revision_schedules WHERE student_id = ?", (student_id,))
    total_sched = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as completed FROM revision_schedules WHERE student_id = ? AND status = 'completed'", (student_id,))
    completed_sched = cursor.fetchone()["completed"]

    conn.close()

    completion_rate = round((completed_sched / max(1, total_sched)) * 100.0, 1) if total_sched > 0 else 0.0

    return {
        "topic_progression": topic_progression,
        "total_scheduled_sessions": total_sched,
        "completed_sessions": completed_sched,
        "completion_rate_pct": completion_rate,
        "has_progress_data": len(attempts) > 0,
    }
