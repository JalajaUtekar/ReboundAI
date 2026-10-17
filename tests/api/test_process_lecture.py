"""
Tests for api/rebound/process_lecture.py.

Three tiers:
  1. Gating logic (_is_subject_flagged) against the real, live Supabase
     Postgres database, using known real student_subject_id values from the
     120-student seed data -- one that IS currently flagged at risk, one
     that is NOT.
  2. The PDF extraction + offline_extractive generation pipeline, against a
     small synthetic PDF built with PyMuPDF itself, confirming the produced
     revision kit has the expected well-formed shape.
  3. A check that process_lecture_request() degrades to a clear 500
     RequestError (rather than an ImportError at collection time) while
     api/_lib (db.py / auth.py) has not landed yet -- these are being built
     by a parallel workstream and may not exist when this suite runs.

Full HTTP-handler-level integration testing (do_POST with a real Supabase
Auth JWT) is intentionally NOT included here even though api/_lib/auth.py
and api/_lib/db.py have landed: minting a real JWT for a seeded student
would require either knowing their password (seeded as a random uuid4,
never recorded) or temporarily changing it via the admin API, which is a
real account-credential mutation against the live project and out of scope
for an automated test run. The gating/ownership/extraction/generation
logic below is exercised directly and thoroughly instead; a true end-to-end
POST is better done manually (or via a dedicated, explicitly-approved test
harness) against a demo account.
"""

import base64
import os

import fitz  # PyMuPDF
import psycopg2
import pytest
from dotenv import load_dotenv

from api.rebound.process_lecture import (
    RequestError,
    _decode_pdf_base64,
    _extract_and_chunk,
    _generate_revision_kit,
    _is_subject_flagged,
    _owns_student_subject,
    get_current_profile,
    get_db_connection,
    process_lecture_request,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Known real seed-data student_subject_id values (see docstring above).
# Confirmed live against the deployed Supabase database at the time this
# test was written: risk_predictions currently has exactly 12 rows, all
# is_flagged = true (~10-15% of 720 student_subjects, per spec section 4);
# every other student_subject has no risk_predictions row at all, i.e. not
# flagged. If the seed data is regenerated these may need refreshing.
FLAGGED_STUDENT_SUBJECT_ID = "53f7d342-909e-5353-b7a5-071efb58edef"  # CN, is_flagged=true
NOT_FLAGGED_STUDENT_SUBJECT_ID = "002ff16f-f4a6-52de-af67-05ffcb2dda73"  # no risk_predictions row


def _live_db_conn():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set; cannot reach the live Supabase database.")
    try:
        return psycopg2.connect(database_url)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not connect to the live Supabase database: {exc}")


class TestIsSubjectFlaggedLiveDB:
    def test_known_flagged_subject_returns_true(self):
        conn = _live_db_conn()
        try:
            assert _is_subject_flagged(FLAGGED_STUDENT_SUBJECT_ID, conn) is True
        finally:
            conn.close()

    def test_known_not_flagged_subject_returns_false(self):
        conn = _live_db_conn()
        try:
            assert _is_subject_flagged(NOT_FLAGGED_STUDENT_SUBJECT_ID, conn) is False
        finally:
            conn.close()

    def test_nonexistent_subject_returns_false(self):
        conn = _live_db_conn()
        try:
            assert _is_subject_flagged("00000000-0000-0000-0000-000000000000", conn) is False
        finally:
            conn.close()


class TestDecodePdfBase64:
    def test_rejects_empty_string(self):
        with pytest.raises(RequestError) as exc_info:
            _decode_pdf_base64("")
        assert exc_info.value.status == 400

    def test_rejects_non_base64(self):
        with pytest.raises(RequestError) as exc_info:
            _decode_pdf_base64("not-%-valid-base64-!!!")
        assert exc_info.value.status == 400

    def test_strips_data_url_prefix(self):
        raw = b"%PDF-1.4 minimal"
        encoded = base64.b64encode(raw).decode("ascii")
        decoded = _decode_pdf_base64(f"data:application/pdf;base64,{encoded}")
        assert decoded == raw

    def test_decodes_plain_base64(self):
        raw = b"hello world"
        encoded = base64.b64encode(raw).decode("ascii")
        assert _decode_pdf_base64(encoded) == raw


def _build_synthetic_lecture_pdf() -> bytes:
    """Builds a small real PDF with PyMuPDF containing enough lecture-like text."""
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Computer Networks Lecture: TCP Congestion Control\n\n"
        "Congestion control is defined as the mechanism that prevents a network from "
        "being overwhelmed by too much traffic. TCP uses congestion control to adjust "
        "the sending rate based on observed network conditions.\n\n"
        "Slow start is a phase where the congestion window grows exponentially until "
        "a threshold is reached. Congestion avoidance refers to the phase where the "
        "window grows linearly to probe for additional available bandwidth.\n\n"
        "Fast retransmit is a technique that allows a sender to retransmit a lost "
        "segment before the retransmission timer expires, triggered by duplicate "
        "acknowledgements. Fast recovery is a mechanism that avoids returning to slow "
        "start after a fast retransmit event.\n\n"
        "The congestion window, often called cwnd, represents the amount of data a "
        "sender is permitted to have in flight at any given time. Network congestion "
        "occurs when the offered load on the network exceeds its capacity.\n"
    )
    page.insert_text((50, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestExtractionAndGenerationPipeline:
    def test_extraction_produces_cleaned_text_and_chunks(self):
        pdf_bytes = _build_synthetic_lecture_pdf()
        extraction = _extract_and_chunk(pdf_bytes)
        assert extraction["word_count"] > 20
        assert extraction["chunk_count"] >= 1
        assert "congestion" in extraction["cleaned_text"].lower()

    def test_corrupt_pdf_raises_request_error_400(self):
        with pytest.raises(RequestError) as exc_info:
            _extract_and_chunk(b"this is not a pdf at all")
        assert exc_info.value.status == 400

    def test_offline_extractive_generation_shape(self):
        pdf_bytes = _build_synthetic_lecture_pdf()
        extraction = _extract_and_chunk(pdf_bytes)
        kit = _generate_revision_kit("TCP Congestion Control", extraction["cleaned_text"])

        assert isinstance(kit["summary"], str) and len(kit["summary"]) > 0
        assert isinstance(kit["revision_notes"], str) and len(kit["revision_notes"]) > 0
        assert isinstance(kit["key_definitions"], list) and len(kit["key_definitions"]) >= 1
        assert isinstance(kit["exam_questions"], list) and len(kit["exam_questions"]) >= 1
        assert isinstance(kit["flashcards"], list) and len(kit["flashcards"]) >= 3
        assert isinstance(kit["quiz_questions"], list) and len(kit["quiz_questions"]) >= 3
        assert "offline extractive" in kit["backend_used"].lower()

        for fc in kit["flashcards"]:
            assert fc["front_question"]
            assert fc["back_answer"]
        for qq in kit["quiz_questions"]:
            assert qq["question_text"]
            assert qq["correct_option"] in ("A", "B", "C", "D")
            for opt in ("option_a", "option_b", "option_c", "option_d"):
                assert qq[opt]


@pytest.mark.skipif(
    get_current_profile is not None and get_db_connection is not None,
    reason="api/_lib has landed; this checks the pre-landing degradation path only.",
)
class TestDegradesCleanlyBeforeApiLibLands:
    def test_process_lecture_request_returns_500_not_import_error(self):
        with pytest.raises(RequestError) as exc_info:
            process_lecture_request("Bearer faketoken", {
                "student_subject_id": FLAGGED_STUDENT_SUBJECT_ID,
                "title": "Test Lecture",
                "pdf_base64": base64.b64encode(b"%PDF-1.4").decode("ascii"),
            })
        assert exc_info.value.status == 500


class TestOwnershipCheck:
    """Exercises _owns_student_subject against the real profile shape get_current_profile returns."""

    def test_owns_when_id_present_in_students_own_subjects(self):
        profile = {
            "user_id": "u1",
            "role": "student",
            "department": "CE",
            "student_id": "s1",
            "student_subjects": [{"id": FLAGGED_STUDENT_SUBJECT_ID, "subject_slug": "CN"}],
        }
        assert _owns_student_subject(FLAGGED_STUDENT_SUBJECT_ID, profile) is True

    def test_does_not_own_someone_elses_subject(self):
        profile = {
            "user_id": "u1",
            "role": "student",
            "department": "CE",
            "student_id": "s1",
            "student_subjects": [{"id": FLAGGED_STUDENT_SUBJECT_ID, "subject_slug": "CN"}],
        }
        assert _owns_student_subject(NOT_FLAGGED_STUDENT_SUBJECT_ID, profile) is False

    def test_non_student_profile_owns_nothing(self):
        profile = {
            "user_id": "u2",
            "role": "professor",
            "department": "CE",
            "student_id": None,
            "student_subjects": [],
        }
        assert _owns_student_subject(FLAGGED_STUDENT_SUBJECT_ID, profile) is False


def test_missing_auth_header_returns_401():
    with pytest.raises(RequestError) as exc_info:
        process_lecture_request(None, {
            "student_subject_id": FLAGGED_STUDENT_SUBJECT_ID,
            "title": "Should Be Rejected",
            "pdf_base64": base64.b64encode(b"%PDF-1.4").decode("ascii"),
        })
    assert exc_info.value.status == 401
