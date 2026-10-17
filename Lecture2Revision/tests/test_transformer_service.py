"""
Unit tests for Transformer service JSON parsing, validation, and offline NLP extraction.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformer_service import (
    sanitize_and_parse_json,
    validate_learning_materials_dict,
    generate_offline_extractive_materials,
)


def test_sanitize_and_parse_json_with_code_fences():
    raw_markdown = """```json
    {
        "summary": "Sample executive summary.",
        "revision_notes": "- Note 1\\n- Note 2",
        "key_definitions": [
            {"term": "Socket", "definition": "An IP and port pair.", "source_context": "Section 1"}
        ],
        "exam_questions": [
            {"question": "What is TCP?", "suggested_answer": "Reliable byte-stream protocol.", "difficulty": "Easy"}
        ],
        "flashcards": [
            {"front_question": "What is rwnd?", "back_answer": "Receive Window for flow control.", "difficulty": "Medium"}
        ],
        "quiz_questions": [
            {
                "question_text": "What flag starts TCP?",
                "option_a": "SYN",
                "option_b": "ACK",
                "option_c": "FIN",
                "option_d": "RST",
                "correct_option": "A",
                "explanation": "SYN initiates the three-way handshake.",
                "difficulty": "Easy"
            }
        ]
    }
    ```"""

    data = sanitize_and_parse_json(raw_markdown)
    assert data["summary"] == "Sample executive summary."
    assert len(data["key_definitions"]) == 1
    assert data["key_definitions"][0]["term"] == "Socket"
    assert len(data["quiz_questions"]) == 1
    assert data["quiz_questions"][0]["correct_option"] == "A"


def test_validate_learning_materials_fallback():
    # Incomplete dictionary
    data = {
        "summary": "Only summary",
        "key_definitions": [{"term": "T", "definition": "D"}],
    }
    validated = validate_learning_materials_dict(data)
    assert validated["summary"] == "Only summary"
    assert validated["revision_notes"] == ""
    assert len(validated["key_definitions"]) == 1
    assert validated["quiz_questions"] == []


def test_offline_extractive_generation():
    text = (
        "Transmission Control Protocol is a reliable transport protocol. "
        "A socket is defined as an IP address combined with a port number. "
        "The three-way handshake establishes a synchronized bidirectional connection. "
        "Congestion control dynamically throttles the congestion window size. "
        "User Datagram Protocol provides best-effort connectionless delivery."
    )
    res = generate_offline_extractive_materials("Transport Layer", text)

    assert "summary" in res and len(res["summary"]) > 0
    assert "revision_notes" in res and len(res["revision_notes"]) > 0
    assert len(res["key_definitions"]) > 0
    assert len(res["flashcards"]) > 0
    assert len(res["quiz_questions"]) > 0

    # Ensure all quiz questions have valid option fields and valid correct_option
    for q in res["quiz_questions"]:
        assert q["correct_option"] in ("A", "B", "C", "D")
        assert "option_a" in q and "option_b" in q
