"""Tests for api/predict_risk.py's request-handling logic
(`handle_predict_risk`) against the REAL live Supabase database.

Only the GoTrue network call inside `get_current_profile` is stubbed (see
tests/api/test_auth.py for why); everything else -- the DB-backed
ownership check, feature computation, model inference, and the
risk_predictions UPSERT -- runs for real.
"""

from __future__ import annotations

import pytest

from api._lib.db import get_db_connection
from api.predict_risk import _ApiError, handle_predict_risk


class _FakeGoTrueResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _fetch_real_flagged_student_subject():
    """A real (profile_id, student_subject_id) pair for a currently
    flagged student-subject, looked up live."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "select s.profile_id, ss.id from public.student_subjects ss "
            "join public.students s on s.id = ss.student_id "
            "join public.risk_predictions rp on rp.student_subject_id = ss.id "
            "where rp.is_flagged limit 1"
        )
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()
    if row is None:
        pytest.skip("no flagged student_subject found in the live database")
    return row[0], row[1]


def _fetch_other_students_subject_id(exclude_profile_id: str):
    """A student_subject_id belonging to a DIFFERENT student, for the
    cross-student 403 test."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "select ss.id from public.student_subjects ss "
            "join public.students s on s.id = ss.student_id "
            "where s.profile_id != %s limit 1",
            (exclude_profile_id,),
        )
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()
    if row is None:
        pytest.skip("expected at least two students in the live database")
    return row[0]


@pytest.fixture(scope="module")
def real_flagged_pair():
    return _fetch_real_flagged_student_subject()


def test_missing_student_subject_id_is_400(monkeypatch, real_flagged_pair):
    profile_id, _ = real_flagged_pair

    def fake_get(url, headers=None, timeout=None):
        return _FakeGoTrueResponse(200, {"id": profile_id})

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(_ApiError) as exc_info:
        handle_predict_risk("Bearer fake", {})
    assert exc_info.value.status == 400


def test_other_students_subject_is_403(monkeypatch, real_flagged_pair):
    profile_id, _ = real_flagged_pair
    other_subject_id = _fetch_other_students_subject_id(profile_id)

    def fake_get(url, headers=None, timeout=None):
        return _FakeGoTrueResponse(200, {"id": profile_id})

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(_ApiError) as exc_info:
        handle_predict_risk("Bearer fake", {"student_subject_id": other_subject_id})
    assert exc_info.value.status == 403


def test_predict_risk_end_to_end_upserts_real_row(monkeypatch, real_flagged_pair):
    profile_id, student_subject_id = real_flagged_pair

    def fake_get(url, headers=None, timeout=None):
        return _FakeGoTrueResponse(200, {"id": profile_id})

    monkeypatch.setattr("requests.get", fake_get)

    result = handle_predict_risk("Bearer fake", {"student_subject_id": student_subject_id})

    assert result["student_subject_id"] == student_subject_id
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert isinstance(result["is_flagged"], bool)
    assert isinstance(result["is_failing"], bool)
    assert result["model_version"] == "logreg_v1"
    assert result["human_readable_summary"]
    assert 2 <= result["checkpoint_index"] <= 7

    # Calling it again for the SAME checkpoint must UPDATE the same row,
    # not insert a second one (see api/predict_risk.py's upsert comment).
    result_again = handle_predict_risk("Bearer fake", {"student_subject_id": student_subject_id})
    assert result_again["id"] == result["id"]
    assert result_again["checkpoint_index"] == result["checkpoint_index"]

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "select count(*) from public.risk_predictions "
            "where student_subject_id = %s and checkpoint_index = %s",
            (student_subject_id, result["checkpoint_index"]),
        )
        count = cur.fetchone()[0]
        cur.close()
    finally:
        conn.close()
    assert count == 1
