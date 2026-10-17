"""
Lecture To Revision - Quiz Generator & Evaluator Module
Manages quiz question formatting, response validation, score calculation,
and review analysis.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple


def generate_attempt_uuid() -> str:
    """Generates a unique idempotency token for a quiz session."""
    return str(uuid.uuid4())


def evaluate_quiz_answers(
    questions: List[Dict[str, Any]],
    user_answers: Dict[int, str],
    time_spent_seconds: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluates submitted user answers against questions.

    Args:
        questions: List of question dictionaries from DB.
        user_answers: Dict mapping question_id -> chosen option ('A', 'B', 'C', 'D').
        time_spent_seconds: Elapsed quiz time.

    Returns:
        Dictionary containing score_pct, correct_count, incorrect_count,
        detailed response breakdown, and topic summary.
    """
    total = len(questions)
    if total == 0:
        return {
            "score_pct": 0.0,
            "total_questions": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "time_spent_seconds": time_spent_seconds,
            "responses": [],
            "topic_accuracy": {},
        }

    correct_count = 0
    responses = []
    topic_stats: Dict[str, Dict[str, int]] = {}

    for q in questions:
        q_id = q["id"]
        chosen = user_answers.get(q_id, "").strip().upper()
        correct = q["correct_option"].strip().upper()
        topic = q.get("topic", "General")

        is_correct = (chosen == correct) and bool(chosen)
        if is_correct:
            correct_count += 1

        if topic not in topic_stats:
            topic_stats[topic] = {"correct": 0, "total": 0}
        topic_stats[topic]["total"] += 1
        if is_correct:
            topic_stats[topic]["correct"] += 1

        # Determine option text for chosen & correct
        opt_key_chosen = f"option_{chosen.lower()}" if chosen in ("A", "B", "C", "D") else None
        opt_key_correct = f"option_{correct.lower()}" if correct in ("A", "B", "C", "D") else None

        responses.append({
            "question_id": q_id,
            "question_text": q["question_text"],
            "selected_option": chosen,
            "selected_text": q.get(opt_key_chosen, "No answer selected") if opt_key_chosen else "No answer selected",
            "correct_option": correct,
            "correct_text": q.get(opt_key_correct, ""),
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
            "difficulty": q.get("difficulty", "Medium"),
            "topic": topic,
        })

    incorrect_count = total - correct_count
    score_pct = round((correct_count / total) * 100.0, 1)

    topic_accuracy = {
        top: round((vals["correct"] / vals["total"]) * 100.0, 1)
        for top, vals in topic_stats.items()
    }

    return {
        "score_pct": score_pct,
        "total_questions": total,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "time_spent_seconds": time_spent_seconds,
        "responses": responses,
        "topic_accuracy": topic_accuracy,
    }
