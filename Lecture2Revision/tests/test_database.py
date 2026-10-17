"""
Unit tests for SQLite database schema, persistence, and idempotency.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database import (
    init_db,
    save_lecture,
    get_lecture_by_id,
    save_learning_materials,
    get_learning_materials_by_lecture,
    save_flashcards,
    get_flashcards,
    record_quiz_attempt,
    get_student_dashboard_stats,
)


def test_database_initialization_and_lecture_save(tmp_path):
    test_db = tmp_path / "test.db"
    init_db(test_db)

    lec_id = save_lecture(
        title="Test Lecture",
        subject="Computer Science",
        topic="Operating Systems",
        filename="os.pdf",
        file_path="/path/to/os.pdf",
        raw_text="Raw OS text",
        cleaned_text="Cleaned OS text",
        word_count=50,
        chunk_count=1,
        db_path=test_db,
    )
    assert lec_id > 0

    fetched = get_lecture_by_id(lec_id, db_path=test_db)
    assert fetched is not None
    assert fetched["title"] == "Test Lecture"
    assert fetched["topic"] == "Operating Systems"


def test_learning_materials_and_flashcards_persistence(tmp_path):
    test_db = tmp_path / "test.db"
    init_db(test_db)

    lec_id = save_lecture("L1", "S1", "T1", "f1.pdf", "p1", "raw", "clean", 10, 1, db_path=test_db)

    mat_id = save_learning_materials(
        lecture_id=lec_id,
        summary="Summary of L1",
        revision_notes="- Point 1",
        key_definitions='[{"term":"T","definition":"D"}]',
        exam_questions="[]",
        source_grounding_notes="Notes",
        backend_used="TestBackend",
        db_path=test_db,
    )
    assert mat_id > 0

    saved_mat = get_learning_materials_by_lecture(lec_id, db_path=test_db)
    assert saved_mat["summary"] == "Summary of L1"
    assert saved_mat["backend_used"] == "TestBackend"

    fc_count = save_flashcards(
        lecture_id=lec_id,
        topic="T1",
        flashcards=[{"front_question": "Q?", "back_answer": "A!", "difficulty": "Easy"}],
        db_path=test_db,
    )
    assert fc_count == 1

    cards = get_flashcards(lecture_id=lec_id, db_path=test_db)
    assert len(cards) == 1
    assert cards[0]["front_question"] == "Q?"


def test_quiz_attempt_idempotency_prevents_duplicate_insertion(tmp_path):
    test_db = tmp_path / "test.db"
    init_db(test_db)

    lec_id = save_lecture("L1", "S1", "T1", "f1.pdf", "p1", "raw", "clean", 10, 1, db_path=test_db)

    uuid_token = "unique-test-attempt-12345"

    # First attempt save
    att_id_1 = record_quiz_attempt(
        attempt_uuid=uuid_token,
        student_id="demo_student",
        lecture_id=lec_id,
        topic="T1",
        score_pct=75.0,
        total_questions=4,
        correct_count=3,
        incorrect_count=1,
        time_spent_seconds=60.0,
        responses=[],
        db_path=test_db,
    )
    assert att_id_1 is not None

    # Duplicate attempt save with SAME uuid (simulating page refresh)
    att_id_2 = record_quiz_attempt(
        attempt_uuid=uuid_token,
        student_id="demo_student",
        lecture_id=lec_id,
        topic="T1",
        score_pct=75.0,
        total_questions=4,
        correct_count=3,
        incorrect_count=1,
        time_spent_seconds=60.0,
        responses=[],
        db_path=test_db,
    )

    # Must return the existing record ID and NOT create a second duplicate row
    assert att_id_2 == att_id_1

    stats = get_student_dashboard_stats("demo_student", db_path=test_db)
    assert stats["total_quizzes"] == 1
    assert stats["avg_score"] == 75.0
