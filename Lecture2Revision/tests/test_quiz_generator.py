"""
Unit tests for Quiz Generator and answer evaluation.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quiz_generator import evaluate_quiz_answers, generate_attempt_uuid


def test_generate_attempt_uuid_uniqueness():
    u1 = generate_attempt_uuid()
    u2 = generate_attempt_uuid()
    assert u1 != u2
    assert len(u1) > 20


def test_evaluate_quiz_answers_all_correct():
    questions = [
        {
            "id": 1,
            "question_text": "What does TCP provide?",
            "option_a": "Reliable byte-stream",
            "option_b": "Unreliable datagrams",
            "option_c": "MAC addressing",
            "option_d": "Physical routing",
            "correct_option": "A",
            "explanation": "TCP ensures reliability.",
            "topic": "TCP",
            "difficulty": "Easy",
        },
        {
            "id": 2,
            "question_text": "What size is UDP header?",
            "option_a": "20 bytes",
            "option_b": "8 bytes",
            "option_c": "40 bytes",
            "option_d": "64 bytes",
            "correct_option": "B",
            "explanation": "UDP header is fixed 8 bytes.",
            "topic": "UDP",
            "difficulty": "Easy",
        },
    ]

    answers = {1: "A", 2: "B"}
    res = evaluate_quiz_answers(questions, answers, time_spent_seconds=45.0)

    assert res["score_pct"] == 100.0
    assert res["correct_count"] == 2
    assert res["incorrect_count"] == 0
    assert res["time_spent_seconds"] == 45.0
    assert res["topic_accuracy"]["TCP"] == 100.0
    assert res["topic_accuracy"]["UDP"] == 100.0


def test_evaluate_quiz_answers_partial_and_missing():
    questions = [
        {"id": 10, "question_text": "Q1", "option_a": "A", "option_b": "B", "correct_option": "A", "topic": "T1"},
        {"id": 20, "question_text": "Q2", "option_a": "A", "option_b": "B", "correct_option": "B", "topic": "T1"},
    ]

    # Student only answered Q1 wrong and left Q2 empty
    answers = {10: "B"}
    res = evaluate_quiz_answers(questions, answers, time_spent_seconds=20.0)

    assert res["score_pct"] == 0.0
    assert res["correct_count"] == 0
    assert res["incorrect_count"] == 2
    assert res["topic_accuracy"]["T1"] == 0.0
