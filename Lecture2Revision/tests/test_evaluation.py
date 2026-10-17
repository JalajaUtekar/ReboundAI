"""
Unit tests for Evaluation module: grounding audits and progress metrics.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import audit_material_grounding, get_adaptive_progress_metrics
from database import init_db, save_lecture, record_quiz_attempt


def test_audit_material_grounding_high_overlap():
    raw_text = (
        "The Transmission Control Protocol ensures reliable in-order byte stream delivery. "
        "A socket is an IP address combined with a port number."
    )
    materials = {
        "summary": "TCP delivers reliable byte streams.",
        "revision_notes": "- Sockets connect IP and ports.",
        "key_definitions": [
            {"term": "Socket", "definition": "An IP and port combination."}
        ],
        "quiz_questions": [
            {"question_text": "What does Transmission Control Protocol ensure?", "correct_option": "A"}
        ]
    }

    res = audit_material_grounding(raw_text, materials)
    assert res["grounding_score_pct"] >= 75.0
    assert res["verified_definitions"] == 1
    assert res["verified_quiz_questions"] == 1


def test_get_adaptive_progress_metrics_deltas(tmp_path):
    test_db = tmp_path / "test_eval.db"
    init_db(test_db)

    lec_id = save_lecture("Networks", "CS", "TCP", "f.pdf", "p", "t", "t", 10, 1, db_path=test_db)

    # First attempt: 40%
    record_quiz_attempt(
        attempt_uuid="att-1",
        student_id="demo_student",
        lecture_id=lec_id,
        topic="TCP",
        score_pct=40.0,
        total_questions=5,
        correct_count=2,
        incorrect_count=3,
        time_spent_seconds=60.0,
        responses=[],
        db_path=test_db,
    )

    # Second attempt: 80% (improvement of +40%)
    record_quiz_attempt(
        attempt_uuid="att-2",
        student_id="demo_student",
        lecture_id=lec_id,
        topic="TCP",
        score_pct=80.0,
        total_questions=5,
        correct_count=4,
        incorrect_count=1,
        time_spent_seconds=50.0,
        responses=[],
        db_path=test_db,
    )

    prog = get_adaptive_progress_metrics("demo_student", db_path=test_db)
    assert prog["has_progress_data"] is True
    assert len(prog["topic_progression"]) == 1
    tcp_stat = prog["topic_progression"][0]
    assert tcp_stat["topic"] == "TCP"
    assert tcp_stat["initial_score"] == 40.0
    assert tcp_stat["latest_score"] == 80.0
    assert tcp_stat["delta"] == 40.0
    assert tcp_stat["improved"] is True
