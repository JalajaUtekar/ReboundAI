"""
Unit tests for Adaptive Engine priority calculation and SM-2 schedule updates.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_engine import calculate_topic_priority, generate_adaptive_schedule
from database import init_db, save_lecture, record_quiz_attempt


def test_priority_score_increases_with_low_accuracy_and_mistakes():
    # Low accuracy (40%), 3 mistakes, overdue by 2 days
    high_urgency_score, high_reason = calculate_topic_priority(
        recent_accuracy=40.0,
        recent_incorrect_count=3,
        days_since_last_review=3.0,
        days_overdue=2.0,
        cards_needing_practice=2,
    )

    # High accuracy (95%), 0 mistakes, not overdue
    low_urgency_score, low_reason = calculate_topic_priority(
        recent_accuracy=95.0,
        recent_incorrect_count=0,
        days_since_last_review=1.0,
        days_overdue=0.0,
        cards_needing_practice=0,
    )

    assert high_urgency_score > low_urgency_score
    assert high_urgency_score >= 60.0
    assert low_urgency_score <= 20.0
    assert "Low recent quiz accuracy" in high_reason
    assert "High historical accuracy" in low_reason


def test_adaptive_schedule_generation_from_sqlite(tmp_path):
    test_db = tmp_path / "test_adaptive.db"
    init_db(test_db)

    # Create 2 lectures: 1 where student performed poorly, 1 where student excelled
    lec1_id = save_lecture("Lec TCP", "Networks", "TCP", "tcp.pdf", "p1", "text", "text", 10, 1, db_path=test_db)
    lec2_id = save_lecture("Lec UDP", "Networks", "UDP", "udp.pdf", "p2", "text", "text", 10, 1, db_path=test_db)

    # Poor TCP attempt
    record_quiz_attempt(
        attempt_uuid="att-tcp",
        student_id="demo_student",
        lecture_id=lec1_id,
        topic="TCP",
        score_pct=35.0,
        total_questions=4,
        correct_count=1,
        incorrect_count=3,
        time_spent_seconds=80.0,
        responses=[],
        db_path=test_db,
    )

    # Strong UDP attempt
    record_quiz_attempt(
        attempt_uuid="att-udp",
        student_id="demo_student",
        lecture_id=lec2_id,
        topic="UDP",
        score_pct=95.0,
        total_questions=4,
        correct_count=4,
        incorrect_count=0,
        time_spent_seconds=40.0,
        responses=[],
        db_path=test_db,
    )

    plan = generate_adaptive_schedule("demo_student", db_path=test_db)

    assert len(plan) == 2
    # TCP should be ranked first due to higher urgency score
    assert plan[0]["topic"] == "TCP"
    assert plan[0]["priority_score"] > plan[1]["priority_score"]
    assert plan[1]["topic"] == "UDP"
