"""
Unit tests for Lecture Processor (PyMuPDF extraction, cleaning, and chunking).
"""

import pytest
import fitz
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lecture_processor import (
    clean_extracted_text,
    chunk_document,
    extract_text_from_pdf,
    NoExtractableTextError,
    CorruptPDFError,
)


def test_clean_extracted_text():
    raw = "This is a trans-\nformer lecture.\r\nPage 1 of 5\n\n\n\nNext section with   extra spaces."
    cleaned = clean_extracted_text(raw)

    assert "transformer lecture." in cleaned
    assert "Page 1 of 5" not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "extra spaces" in cleaned


def test_chunk_document_single_chunk():
    text = "Word " * 100
    chunks = chunk_document(text, chunk_size_words=200, overlap_words=20)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == 1
    assert chunks[0]["total_chunks"] == 1
    assert chunks[0]["word_count"] == 100


def test_chunk_document_multiple_chunks_with_overlap():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_document(text, chunk_size_words=200, overlap_words=50)

    assert len(chunks) > 1
    assert chunks[0]["total_chunks"] == len(chunks)
    assert chunks[0]["end_word"] == 200
    # Next chunk starts at 200 - 50 = 150
    assert chunks[1]["start_word"] == 150


def test_scanned_pdf_error_handling(tmp_path):
    # Create empty PDF with zero readable characters
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    with pytest.raises(NoExtractableTextError):
        extract_text_from_pdf(pdf_path)


def test_extract_text_from_valid_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Transmission Control Protocol provides reliable stream delivery across networks.")
    doc.save(str(pdf_path))
    doc.close()

    text, pages = extract_text_from_pdf(pdf_path)
    assert "Transmission Control Protocol" in text
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
