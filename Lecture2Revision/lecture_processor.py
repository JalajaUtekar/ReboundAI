"""
Lecture To Revision - Lecture Processor Module
Extracts text from lecture PDFs using PyMuPDF (fitz), sanitizes content,
handles scanned PDF detection, and segments long documents into coherent chunks.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import fitz  # PyMuPDF


class NoExtractableTextError(Exception):
    """Raised when uploaded PDF has no extractable text (e.g., scanned images)."""
    pass


class CorruptPDFError(Exception):
    """Raised when PDF file is invalid or cannot be opened."""
    pass


def extract_text_from_pdf(pdf_source: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts text from a PDF file path or binary byte stream using PyMuPDF.

    Args:
        pdf_source: File path (str/Path) or bytes / BytesIO buffer.

    Returns:
        Tuple of (full_cleaned_text, list_of_page_dictionaries)

    Raises:
        NoExtractableTextError: If document contains no extractable characters.
        CorruptPDFError: If PyMuPDF fails to parse the document.
    """
    try:
        if isinstance(pdf_source, (str, Path)):
            doc = fitz.open(str(pdf_source))
        elif hasattr(pdf_source, "read"):
            # Streamlit UploadedFile or file-like object
            pdf_bytes = pdf_source.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        elif isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            raise CorruptPDFError(f"Unsupported PDF source type: {type(pdf_source)}")
    except Exception as e:
        if isinstance(e, (NoExtractableTextError, CorruptPDFError)):
            raise
        raise CorruptPDFError(f"Failed to open PDF document: {str(e)}")

    page_records = []
    full_text_parts = []
    total_characters = 0

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            cleaned_page = clean_extracted_text(text)
            char_count = len(cleaned_page)
            total_characters += char_count

            page_records.append({
                "page_number": page_num + 1,
                "raw_text": text,
                "cleaned_text": cleaned_page,
                "char_count": char_count,
            })
            if cleaned_page:
                full_text_parts.append(cleaned_page)
    finally:
        doc.close()

    if total_characters < 30:
        raise NoExtractableTextError(
            "This PDF appears to be a scanned document or contains only graphics/images. "
            "PyMuPDF extracted fewer than 30 characters of readable text. "
            "Please provide a text-based digital lecture PDF or slides."
        )

    full_cleaned_text = "\n\n".join(full_text_parts)
    return full_cleaned_text, page_records


def clean_extracted_text(raw_text: str) -> str:
    """
    Sanitizes raw text extracted from PDF:
    - Normalizes unicode spaces and tabs
    - Removes isolated page numbers or standalone hyphenations
    - Collapses excessive newlines while preserving paragraph breaks
    """
    if not raw_text:
        return ""

    text = raw_text

    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Reconnect words split across lines by hyphens (e.g., "trans- / former" -> "transformer")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Remove standalone page numbers or headers like "Page 3 of 12"
    text = re.sub(r"(?i)\bpage\s+\d+(\s+of\s+\d+)?\b", "", text)

    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines to 2 newlines (paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_document(
    text: str,
    chunk_size_words: int = 800,
    overlap_words: int = 100,
) -> List[Dict[str, Any]]:
    """
    Splits document text into overlapping token/word chunks to prevent truncation
    in Transformer context windows while retaining semantic coherence.

    Args:
        text: Full document string.
        chunk_size_words: Maximum number of words per chunk.
        overlap_words: Overlapping words between adjacent chunks.

    Returns:
        List of chunk dictionaries with metadata (chunk_id, word_count, text).
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size_words:
        return [{
            "chunk_id": 1,
            "total_chunks": 1,
            "start_word": 0,
            "end_word": len(words),
            "word_count": len(words),
            "text": text,
        }]

    chunks = []
    step = max(1, chunk_size_words - overlap_words)
    chunk_id = 1

    for start_idx in range(0, len(words), step):
        end_idx = min(start_idx + chunk_size_words, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": chunk_id,
            "start_word": start_idx,
            "end_word": end_idx,
            "word_count": len(chunk_words),
            "text": chunk_text,
        })
        chunk_id += 1

        if end_idx >= len(words):
            break

    # Annotate total chunks count
    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks
