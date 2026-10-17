"""
Lecture To Revision - Flashcard Service & SM-2 Spaced Repetition Computation
Handles flashcard review state transitions, spaced repetition calculations,
and revision note compilation for export.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from config import SM2_DEFAULT_EASE_FACTOR, SM2_MIN_EASE_FACTOR


def calculate_sm2_update(
    current_interval: float,
    current_ease: float,
    current_repetitions: int,
    rating: str,  # 'remembered' (pass) or 'needs_practice' (fail)
) -> Dict[str, Any]:
    """
    Computes updated SM-2 parameters based on recall rating.

    Standard SM-2 Quality Scale:
    - 'remembered': Quality rating 4 (good recall with little hesitation)
    - 'needs_practice': Quality rating 1 (incorrect or difficult recall)

    Formula:
    EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    Interval:
      Repetition 1: 1 day
      Repetition 2: 6 days
      Repetition n: I(n-1) * EF'
    If recall failed (q < 3):
      Repetitions reset to 0, interval resets to 1 day.
    """
    ease = max(SM2_MIN_EASE_FACTOR, current_ease if current_ease else SM2_DEFAULT_EASE_FACTOR)
    reps = current_repetitions if current_repetitions is not None else 0
    interval = current_interval if current_interval is not None else 1.0

    if rating == "remembered":
        q = 4  # Good recall
        new_reps = reps + 1
        if new_reps == 1:
            new_interval = 1.0
        elif new_reps == 2:
            new_interval = 6.0
        else:
            new_interval = round(interval * ease, 1)

        # Update Ease Factor
        new_ease = ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        new_ease = max(SM2_MIN_EASE_FACTOR, round(new_ease, 2))
    else:
        # Needs practice
        new_reps = 0
        new_interval = 1.0
        # Reduce ease factor slightly for difficult cards
        new_ease = max(SM2_MIN_EASE_FACTOR, round(ease - 0.2, 2))

    next_review_dt = datetime.now(timezone.utc) + timedelta(days=new_interval)
    next_review_date = next_review_dt.strftime("%Y-%m-%d")

    return {
        "interval_days": new_interval,
        "ease_factor": new_ease,
        "repetitions": new_reps,
        "next_review_date": next_review_date,
    }


def filter_flashcards(
    flashcards: List[Dict[str, Any]],
    search_query: str = "",
    topic_filter: str = "All",
    status_filter: str = "All",  # "All", "Needs Practice", "Due Today"
) -> List[Dict[str, Any]]:
    """Filters flashcard list based on user UI criteria."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results = []

    for fc in flashcards:
        # Topic filter
        if topic_filter != "All" and fc.get("topic") != topic_filter:
            continue

        # Status filter
        if status_filter == "Needs Practice" and not fc.get("needs_practice"):
            continue
        elif status_filter == "Due Today":
            next_date = fc.get("next_review_date")
            if next_date and next_date > today_str:
                continue

        # Search query
        if search_query:
            q_lower = search_query.lower()
            front = str(fc.get("front_question", "")).lower()
            back = str(fc.get("back_answer", "")).lower()
            if q_lower not in front and q_lower not in back:
                continue

        results.append(fc)

    return results


def compile_downloadable_notes(lecture_dict: Dict[str, Any], materials_dict: Dict[str, Any]) -> str:
    """Formats full lecture notes into clean, printable markdown for students."""
    title = lecture_dict.get("title", "Lecture Notes")
    subject = lecture_dict.get("subject", "Subject")
    topic = lecture_dict.get("topic", "Topic")
    date_str = lecture_dict.get("created_at", datetime.now().strftime("%Y-%m-%d"))

    md = [
        f"# {title}",
        f"**Subject:** {subject} | **Topic:** {topic} | **Processed Date:** {date_str[:10]}",
        "\n---\n",
        "## Executive Summary",
        materials_dict.get("summary", "No summary available."),
        "\n---\n",
        "## Key Definitions & Concepts",
    ]

    defs = materials_dict.get("key_definitions", [])
    if isinstance(defs, str):
        import json
        try:
            defs = json.loads(defs)
        except Exception:
            defs = []

    for d in defs:
        md.append(f"- **{d.get('term', 'Term')}:** {d.get('definition', '')}")
        if d.get("source_context"):
            md.append(f"  *Context:* _{d.get('source_context')}_")

    md.append("\n---\n")
    md.append("## Detailed Revision Notes")
    md.append(materials_dict.get("revision_notes", "No revision notes available."))

    md.append("\n---\n")
    md.append("## Exam-Oriented Practice Questions")
    exam_q = materials_dict.get("exam_questions", [])
    if isinstance(exam_q, str):
        import json
        try:
            exam_q = json.loads(exam_q)
        except Exception:
            exam_q = []

    for i, eq in enumerate(exam_q, 1):
        md.append(f"### Question {i} ({eq.get('difficulty', 'Medium')})")
        md.append(f"**Q:** {eq.get('question', '')}")
        md.append(f"**Model Answer:** {eq.get('suggested_answer', '')}\n")

    md.append("\n---\n")
    md.append(f"_Generated by Lecture To Revision (Backend: {materials_dict.get('backend_used', 'Local Engine')})_")

    return "\n".join(md)
