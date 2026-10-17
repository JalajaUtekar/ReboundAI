"""
Lecture To Revision - Database Management Module
Handles SQLite schema definition, parameterized queries, and persistence
for lectures, materials, flashcards, quizzes, attempts, and adaptive revision schedules.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from config import DB_PATH


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Creates a thread-safe connection with dictionary rows and foreign keys enabled."""
    target_path = db_path or DB_PATH
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initializes all required relational tables for Lecture To Revision."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Students / Profiles
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        created_at TEXT NOT NULL
    );
    """)

    # Default demo student
    cursor.execute("""
    INSERT OR IGNORE INTO students (student_id, name, email, created_at)
    VALUES ('demo_student', 'Alex Taylor (Demo Student)', 'alex.taylor@university.edu', datetime('now'));
    """)

    # 2. Lectures
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lectures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        subject TEXT NOT NULL,
        topic TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        raw_text TEXT,
        cleaned_text TEXT,
        word_count INTEGER DEFAULT 0,
        chunk_count INTEGER DEFAULT 1,
        created_at TEXT NOT NULL
    );
    """)

    # 3. Generated Learning Materials
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS learning_materials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lecture_id INTEGER NOT NULL,
        summary TEXT NOT NULL,
        revision_notes TEXT NOT NULL,
        key_definitions TEXT,
        exam_questions TEXT,
        source_grounding_notes TEXT,
        backend_used TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (lecture_id) REFERENCES lectures (id) ON DELETE CASCADE
    );
    """)

    # 4. Flashcards
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flashcards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lecture_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        front_question TEXT NOT NULL,
        back_answer TEXT NOT NULL,
        difficulty TEXT DEFAULT 'Medium',
        repetitions INTEGER DEFAULT 0,
        interval_days REAL DEFAULT 1.0,
        ease_factor REAL DEFAULT 2.5,
        last_reviewed_at TEXT,
        next_review_date TEXT,
        needs_practice INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (lecture_id) REFERENCES lectures (id) ON DELETE CASCADE
    );
    """)

    # 5. Flashcard Review Logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flashcard_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        flashcard_id INTEGER NOT NULL,
        student_id TEXT NOT NULL,
        rating TEXT NOT NULL, -- 'remembered' or 'needs_practice'
        reviewed_at TEXT NOT NULL,
        FOREIGN KEY (flashcard_id) REFERENCES flashcards (id) ON DELETE CASCADE,
        FOREIGN KEY (student_id) REFERENCES students (student_id)
    );
    """)

    # 6. Quiz Questions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quiz_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lecture_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        question_text TEXT NOT NULL,
        option_a TEXT NOT NULL,
        option_b TEXT NOT NULL,
        option_c TEXT NOT NULL,
        option_d TEXT NOT NULL,
        correct_option TEXT NOT NULL, -- 'A', 'B', 'C', 'D'
        explanation TEXT NOT NULL,
        difficulty TEXT DEFAULT 'Medium',
        created_at TEXT NOT NULL,
        FOREIGN KEY (lecture_id) REFERENCES lectures (id) ON DELETE CASCADE
    );
    """)

    # 7. Quiz Attempts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quiz_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_uuid TEXT UNIQUE, -- Idempotency token to prevent duplicate saves on refresh
        student_id TEXT NOT NULL,
        lecture_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        score_pct REAL NOT NULL,
        total_questions INTEGER NOT NULL,
        correct_count INTEGER NOT NULL,
        incorrect_count INTEGER NOT NULL,
        time_spent_seconds REAL DEFAULT 0.0,
        completed_at TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students (student_id),
        FOREIGN KEY (lecture_id) REFERENCES lectures (id) ON DELETE CASCADE
    );
    """)

    # 8. Quiz Question Responses
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quiz_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id INTEGER NOT NULL,
        question_id INTEGER NOT NULL,
        selected_option TEXT NOT NULL,
        is_correct INTEGER NOT NULL,
        response_time_seconds REAL DEFAULT 0.0,
        FOREIGN KEY (attempt_id) REFERENCES quiz_attempts (id) ON DELETE CASCADE,
        FOREIGN KEY (question_id) REFERENCES quiz_questions (id) ON DELETE CASCADE
    );
    """)

    # 9. Adaptive Revision Schedules
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS revision_schedules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        topic TEXT NOT NULL,
        lecture_id INTEGER,
        priority_score REAL NOT NULL,
        priority_reason TEXT NOT NULL,
        scheduled_date TEXT NOT NULL,
        last_revised_at TEXT,
        status TEXT DEFAULT 'pending', -- 'pending', 'completed'
        sm2_interval_days REAL DEFAULT 1.0,
        sm2_ease_factor REAL DEFAULT 2.5,
        sm2_repetitions INTEGER DEFAULT 0,
        lstm_predicted_score REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students (student_id),
        FOREIGN KEY (lecture_id) REFERENCES lectures (id) ON DELETE SET NULL
    );
    """)

    conn.commit()
    conn.close()


# ==============================================================================
# Helper Query Functions
# ==============================================================================

def save_lecture(
    title: str,
    subject: str,
    topic: str,
    filename: str,
    file_path: str,
    raw_text: str,
    cleaned_text: str,
    word_count: int,
    chunk_count: int,
    db_path: Optional[Path] = None,
) -> int:
    """Inserts a processed lecture record into SQLite."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO lectures (title, subject, topic, filename, file_path, raw_text, cleaned_text, word_count, chunk_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (title, subject, topic, filename, file_path, raw_text, cleaned_text, word_count, chunk_count, now))
    lecture_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return lecture_id


def save_learning_materials(
    lecture_id: int,
    summary: str,
    revision_notes: str,
    key_definitions: str,
    exam_questions: str,
    source_grounding_notes: str,
    backend_used: str,
    db_path: Optional[Path] = None,
) -> int:
    """Saves structured learning materials derived from a lecture."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO learning_materials (lecture_id, summary, revision_notes, key_definitions, exam_questions, source_grounding_notes, backend_used, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (lecture_id, summary, revision_notes, key_definitions, exam_questions, source_grounding_notes, backend_used, now))
    mat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return mat_id


def save_flashcards(lecture_id: int, topic: str, flashcards: List[Dict[str, Any]], db_path: Optional[Path] = None) -> int:
    """Inserts generated flashcards."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    for fc in flashcards:
        cursor.execute("""
            INSERT INTO flashcards (lecture_id, topic, front_question, back_answer, difficulty, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            lecture_id,
            topic,
            fc.get("front_question", fc.get("question", "")),
            fc.get("back_answer", fc.get("answer", "")),
            fc.get("difficulty", "Medium"),
            now,
        ))
        count += 1
    conn.commit()
    conn.close()
    return count


def save_quiz_questions(lecture_id: int, topic: str, questions: List[Dict[str, Any]], db_path: Optional[Path] = None) -> int:
    """Inserts generated quiz questions."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    for q in questions:
        cursor.execute("""
            INSERT INTO quiz_questions (lecture_id, topic, question_text, option_a, option_b, option_c, option_d, correct_option, explanation, difficulty, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            lecture_id,
            topic,
            q.get("question_text", q.get("question", "")),
            q.get("option_a", ""),
            q.get("option_b", ""),
            q.get("option_c", ""),
            q.get("option_d", ""),
            q.get("correct_option", "A").upper().strip(),
            q.get("explanation", ""),
            q.get("difficulty", "Medium"),
            now,
        ))
        count += 1
    conn.commit()
    conn.close()
    return count


def record_quiz_attempt(
    attempt_uuid: str,
    student_id: str,
    lecture_id: int,
    topic: str,
    score_pct: float,
    total_questions: int,
    correct_count: int,
    incorrect_count: int,
    time_spent_seconds: float,
    responses: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> Optional[int]:
    """Records a quiz attempt idempotently. Returns None if attempt_uuid was already recorded."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Check if duplicate UUID
    cursor.execute("SELECT id FROM quiz_attempts WHERE attempt_uuid = ?", (attempt_uuid,))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return existing["id"]

    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO quiz_attempts (attempt_uuid, student_id, lecture_id, topic, score_pct, total_questions, correct_count, incorrect_count, time_spent_seconds, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (attempt_uuid, student_id, lecture_id, topic, score_pct, total_questions, correct_count, incorrect_count, time_spent_seconds, now))
    attempt_id = cursor.lastrowid

    for r in responses:
        cursor.execute("""
            INSERT INTO quiz_responses (attempt_id, question_id, selected_option, is_correct, response_time_seconds)
            VALUES (?, ?, ?, ?, ?)
        """, (
            attempt_id,
            r["question_id"],
            r.get("selected_option", ""),
            1 if r.get("is_correct") else 0,
            r.get("response_time_seconds", 0.0),
        ))

    conn.commit()
    conn.close()
    return attempt_id


def get_all_lectures(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves all registered lectures."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM lectures ORDER BY id DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_lecture_by_id(lecture_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieves single lecture record."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM lectures WHERE id = ?", (lecture_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_learning_materials_by_lecture(lecture_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieves learning materials for a lecture."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM learning_materials WHERE lecture_id = ? ORDER BY id DESC LIMIT 1", (lecture_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_quiz_questions_by_lecture(lecture_id: int, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves quiz questions for a lecture."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM quiz_questions WHERE lecture_id = ? ORDER BY id ASC", (lecture_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_flashcards(lecture_id: Optional[int] = None, topic: Optional[str] = None, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves flashcards with optional filtering."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    query = "SELECT * FROM flashcards WHERE 1=1"
    params = []
    if lecture_id is not None:
        query += " AND lecture_id = ?"
        params.append(lecture_id)
    if topic:
        query += " AND topic = ?"
        params.append(topic)
    query += " ORDER BY id ASC"
    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def update_flashcard_review(
    flashcard_id: int,
    student_id: str,
    rating: str,  # 'remembered' or 'needs_practice'
    interval_days: float,
    ease_factor: float,
    repetitions: int,
    next_review_date: str,
    db_path: Optional[Path] = None,
) -> None:
    """Updates flashcard review status using SM-2 computed properties and logs review."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    needs_practice = 1 if rating == "needs_practice" else 0

    cursor.execute("""
        UPDATE flashcards
        SET interval_days = ?, ease_factor = ?, repetitions = ?,
            last_reviewed_at = ?, next_review_date = ?, needs_practice = ?
        WHERE id = ?
    """, (interval_days, ease_factor, repetitions, now, next_review_date, needs_practice, flashcard_id))

    cursor.execute("""
        INSERT INTO flashcard_reviews (flashcard_id, student_id, rating, reviewed_at)
        VALUES (?, ?, ?, ?)
    """, (flashcard_id, student_id, rating, now))

    conn.commit()
    conn.close()


def get_student_dashboard_stats(student_id: str = "demo_student", db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Calculates factual statistics directly from the database for the dashboard."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS total FROM lectures")
    lectures_count = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS total, AVG(score_pct) AS avg_score FROM quiz_attempts WHERE student_id = ?", (student_id,))
    row = cursor.fetchone()
    total_quizzes = row["total"] or 0
    avg_score = round(row["avg_score"], 1) if row["avg_score"] is not None else 0.0

    # Topics needing revision: topics where recent average score < 70% or marked needs_practice
    cursor.execute("""
        SELECT topic, AVG(score_pct) as avg_score, COUNT(*) as attempts
        FROM quiz_attempts
        WHERE student_id = ?
        GROUP BY topic
        HAVING avg_score < 70.0
        ORDER BY avg_score ASC
    """, (student_id,))
    weak_topics = [dict(r) for r in cursor.fetchall()]

    # Upcoming revision sessions from revision_schedules
    cursor.execute("""
        SELECT * FROM revision_schedules
        WHERE student_id = ? AND status = 'pending'
        ORDER BY scheduled_date ASC
        LIMIT 5
    """, (student_id,))
    upcoming_sessions = [dict(r) for r in cursor.fetchall()]

    # Recent activity
    cursor.execute("""
        SELECT 'quiz' as activity_type, topic as title, score_pct as score, completed_at as timestamp
        FROM quiz_attempts
        WHERE student_id = ?
        ORDER BY completed_at DESC
        LIMIT 5
    """, (student_id,))
    recent_activity = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return {
        "lectures_count": lectures_count,
        "total_quizzes": total_quizzes,
        "avg_score": avg_score,
        "weak_topics": weak_topics,
        "upcoming_sessions": upcoming_sessions,
        "recent_activity": recent_activity,
        "has_data": (lectures_count > 0 or total_quizzes > 0),
    }


def get_topic_performance(student_id: str = "demo_student", db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Returns chronological scores grouped by topic for progress tracking."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, topic, score_pct, correct_count, total_questions, time_spent_seconds, completed_at
        FROM quiz_attempts
        WHERE student_id = ?
        ORDER BY completed_at ASC
    """, (student_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_quiz_history(student_id: str = "demo_student", db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Returns list of all quiz attempts with lecture metadata."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT q.*, l.title as lecture_title
        FROM quiz_attempts q
        LEFT JOIN lectures l ON q.lecture_id = l.id
        WHERE q.student_id = ?
        ORDER BY q.completed_at DESC
    """, (student_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
