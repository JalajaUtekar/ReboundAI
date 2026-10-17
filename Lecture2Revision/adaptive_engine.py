"""
Lecture To Revision - Adaptive Revision Engine
Calculates topic priority and spaced repetition schedules by synthesizing:
1. Recent quiz accuracy
2. Consecutive mistakes
3. Time elapsed and overdue status
4. Flashcard recall quality
5. LSTM sequential score predictions (when trained model is available)
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import SM2_DEFAULT_EASE_FACTOR, SM2_MIN_EASE_FACTOR
from database import get_connection


def calculate_topic_priority(
    recent_accuracy: float,
    recent_incorrect_count: int,
    days_since_last_review: float,
    days_overdue: float,
    cards_needing_practice: int,
    lstm_predicted_score: Optional[float] = None,
    weight_accuracy: float = 0.40,
    weight_mistakes: float = 0.20,
    weight_overdue: float = 0.25,
    weight_flashcards: float = 0.15,
) -> Tuple[float, str]:
    """
    Computes a transparent priority score (0 to 100) and human-readable explanation.
    Higher score indicates higher urgency for revision.
    """
    # 1. Accuracy component (0 to 100, where 0% accuracy gives 100 urgency)
    acc_urgency = max(0.0, min(100.0, 100.0 - recent_accuracy))

    # 2. Mistakes component (capped at 4 mistakes = 100)
    mistakes_urgency = min(100.0, recent_incorrect_count * 25.0)

    # 3. Overdue component (each overdue day adds 30 urgency, capped at 100)
    overdue_urgency = min(100.0, max(0.0, days_overdue * 30.0))

    # 4. Flashcards component (each card needing practice adds 30 urgency)
    fc_urgency = min(100.0, cards_needing_practice * 30.0)

    # Base rule-based score
    base_priority = (
        (acc_urgency * weight_accuracy) +
        (mistakes_urgency * weight_mistakes) +
        (overdue_urgency * weight_overdue) +
        (fc_urgency * weight_flashcards)
    )

    reasons = []

    if recent_accuracy < 60.0:
        reasons.append(f"Low recent quiz accuracy ({recent_accuracy:.1f}%)")
    elif recent_accuracy < 75.0:
        reasons.append(f"Moderate accuracy ({recent_accuracy:.1f}%) with room for mastery")
    else:
        reasons.append(f"High historical accuracy ({recent_accuracy:.1f}%)")

    if recent_incorrect_count > 0:
        reasons.append(f"{recent_incorrect_count} recent incorrect quiz answer(s)")

    if days_overdue > 0:
        reasons.append(f"Review is {days_overdue:.1f} day(s) overdue")
    elif days_since_last_review >= 4.0:
        reasons.append(f"No revision in {days_since_last_review:.0f} days")

    if cards_needing_practice > 0:
        reasons.append(f"{cards_needing_practice} flashcard(s) marked 'Needs Practice'")

    # Incorporate LSTM prediction if provided
    final_priority = base_priority
    if lstm_predicted_score is not None:
        lstm_urgency = max(0.0, min(100.0, 100.0 - lstm_predicted_score))
        # Blend 70% rule-based + 30% LSTM prediction
        final_priority = (base_priority * 0.70) + (lstm_urgency * 0.30)
        reasons.append(f"LSTM model predicts next quiz score of {lstm_predicted_score:.1f}%")

    final_priority = round(min(100.0, max(0.0, final_priority)), 1)
    explanation = " | ".join(reasons) if reasons else "Routine spaced repetition schedule."

    return final_priority, explanation


def generate_adaptive_schedule(
    student_id: str = "demo_student",
    db_path: Optional[Path] = None,
    lstm_predictor: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Analyzes student quiz attempts, flashcards, and existing schedules in SQLite
    to generate or refresh an adaptive revision plan.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Query distinct topics with lecture references
    cursor.execute("""
        SELECT DISTINCT l.topic, l.id as lecture_id, l.title as lecture_title
        FROM lectures l
    """)
    lectures_by_topic = {r["topic"]: dict(r) for r in cursor.fetchall()}

    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")

    plan = []

    for topic, l_meta in lectures_by_topic.items():
        lecture_id = l_meta["lecture_id"]

        # Fetch quiz attempts for this topic
        cursor.execute("""
            SELECT score_pct, correct_count, incorrect_count, completed_at
            FROM quiz_attempts
            WHERE student_id = ? AND topic = ?
            ORDER BY completed_at DESC
        """, (student_id, topic))
        attempts = [dict(r) for r in cursor.fetchall()]

        if attempts:
            recent_score = attempts[0]["score_pct"]
            recent_incorrect = attempts[0]["incorrect_count"]
            last_dt = datetime.fromisoformat(attempts[0]["completed_at"].replace("Z", "+00:00"))
            days_since_last_review = max(0.0, (now_utc - last_dt).total_seconds() / 86400.0)
        else:
            # Topic uploaded but no quiz attempted yet
            recent_score = 50.0  # neutral starting point
            recent_incorrect = 0
            days_since_last_review = 2.0

        # Fetch flashcards needing practice for this topic
        cursor.execute("""
            SELECT COUNT(*) as cnt
            FROM flashcards
            WHERE lecture_id = ? AND needs_practice = 1
        """, (lecture_id,))
        cards_needing_practice = cursor.fetchone()["cnt"]

        # Check existing revision schedule record
        cursor.execute("""
            SELECT * FROM revision_schedules
            WHERE student_id = ? AND topic = ?
            ORDER BY id DESC LIMIT 1
        """, (student_id, topic))
        sched = cursor.fetchone()

        if sched:
            sched_date_str = sched["scheduled_date"]
            sched_dt = datetime.strptime(sched_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_overdue = max(0.0, (now_utc - sched_dt).total_seconds() / 86400.0)
            sm2_interval = sched["sm2_interval_days"]
            sm2_ease = sched["sm2_ease_factor"]
            sm2_reps = sched["sm2_repetitions"]
        else:
            days_overdue = 0.0
            sm2_interval = 1.0
            sm2_ease = SM2_DEFAULT_EASE_FACTOR
            sm2_reps = 0

        # LSTM inference if available
        lstm_pred = None
        if lstm_predictor is not None and hasattr(lstm_predictor, "predict_next_score"):
            try:
                lstm_pred = lstm_predictor.predict_next_score(student_id, topic, db_path=db_path)
            except Exception:
                lstm_pred = None

        # Priority calculation
        priority_score, priority_reason = calculate_topic_priority(
            recent_accuracy=recent_score,
            recent_incorrect_count=recent_incorrect,
            days_since_last_review=days_since_last_review,
            days_overdue=days_overdue,
            cards_needing_practice=cards_needing_practice,
            lstm_predicted_score=lstm_pred,
        )

        # Spaced repetition calculation for next scheduled date
        # Lower scores shorten the interval; higher scores increase interval
        if recent_score >= 80.0:
            suggested_interval = max(2.0, sm2_interval * 1.5)
        elif recent_score >= 60.0:
            suggested_interval = max(1.0, sm2_interval)
        else:
            suggested_interval = 1.0  # review tomorrow or today

        recommended_dt = now_utc + timedelta(days=suggested_interval)
        recommended_date_str = recommended_dt.strftime("%Y-%m-%d")

        # Upsert into revision_schedules table
        cursor.execute("""
            INSERT INTO revision_schedules (
                student_id, topic, lecture_id, priority_score, priority_reason,
                scheduled_date, status, sm2_interval_days, sm2_ease_factor,
                sm2_repetitions, lstm_predicted_score, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
        """, (
            student_id,
            topic,
            lecture_id,
            priority_score,
            priority_reason,
            recommended_date_str,
            suggested_interval,
            sm2_ease,
            sm2_reps,
            lstm_pred,
            now_utc.isoformat(),
            now_utc.isoformat(),
        ))

        plan.append({
            "topic": topic,
            "lecture_title": l_meta["lecture_title"],
            "priority_score": priority_score,
            "priority_reason": priority_reason,
            "scheduled_date": recommended_date_str,
            "recent_score": recent_score,
            "days_overdue": round(days_overdue, 1),
            "cards_needing_practice": cards_needing_practice,
            "lstm_predicted_score": round(lstm_pred, 1) if lstm_pred is not None else None,
            "status": "pending",
        })

    conn.commit()
    conn.close()

    # Sort descending by priority score (highest urgency first)
    plan.sort(key=lambda x: x["priority_score"], reverse=True)
    return plan
