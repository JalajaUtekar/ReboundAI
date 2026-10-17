"""
Unit tests for Flashcard Service and SM-2 Spaced Repetition logic.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashcard_service import (
    calculate_sm2_update,
    filter_flashcards,
    compile_downloadable_notes,
)


def test_sm2_update_remembered_progression():
    # Repetition 0 -> 1: interval should become 1.0 day
    u1 = calculate_sm2_update(current_interval=1.0, current_ease=2.5, current_repetitions=0, rating="remembered")
    assert u1["repetitions"] == 1
    assert u1["interval_days"] == 1.0
    assert u1["ease_factor"] >= 2.5

    # Repetition 1 -> 2: interval should become 6.0 days
    u2 = calculate_sm2_update(current_interval=u1["interval_days"], current_ease=u1["ease_factor"], current_repetitions=1, rating="remembered")
    assert u2["repetitions"] == 2
    assert u2["interval_days"] == 6.0

    # Repetition 2 -> 3: interval should multiply by ease factor (6 * 2.5 = 15.0)
    u3 = calculate_sm2_update(current_interval=u2["interval_days"], current_ease=u2["ease_factor"], current_repetitions=2, rating="remembered")
    assert u3["repetitions"] == 3
    assert u3["interval_days"] >= 14.0


def test_sm2_update_needs_practice_reset():
    # When failed/forgotten, repetitions must reset to 0 and interval to 1.0 day
    u_fail = calculate_sm2_update(current_interval=15.0, current_ease=2.5, current_repetitions=3, rating="needs_practice")
    assert u_fail["repetitions"] == 0
    assert u_fail["interval_days"] == 1.0
    assert u_fail["ease_factor"] == 2.3  # penalized by 0.2


def test_filter_flashcards():
    cards = [
        {"id": 1, "topic": "TCP", "front_question": "What is SYN?", "back_answer": "Flag", "needs_practice": 1},
        {"id": 2, "topic": "UDP", "front_question": "What is DNS?", "back_answer": "Domain system", "needs_practice": 0},
    ]

    # Filter by topic
    res_topic = filter_flashcards(cards, topic_filter="TCP")
    assert len(res_topic) == 1
    assert res_topic[0]["topic"] == "TCP"

    # Filter by search
    res_search = filter_flashcards(cards, search_query="domain")
    assert len(res_search) == 1
    assert res_search[0]["id"] == 2

    # Filter by status
    res_status = filter_flashcards(cards, status_filter="Needs Practice")
    assert len(res_status) == 1
    assert res_status[0]["id"] == 1


def test_compile_downloadable_notes():
    lecture = {"title": "TCP Lecture", "subject": "Networking", "topic": "TCP"}
    materials = {
        "summary": "This is summary.",
        "revision_notes": "- Note point 1",
        "key_definitions": [{"term": "Socket", "definition": "IP + Port"}],
        "backend_used": "TestEngine",
    }
    md = compile_downloadable_notes(lecture, materials)
    assert "# TCP Lecture" in md
    assert "This is summary." in md
    assert "**Socket:** IP + Port" in md
    assert "TestEngine" in md
