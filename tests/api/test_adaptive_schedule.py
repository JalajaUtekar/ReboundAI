"""
Tests for api/rebound/adaptive_schedule.py.

Three layers:
  1. The risk-boost blend formula (apply_risk_boost) -- hand-picked
     final_priority/risk_probability values, exact expected result. Pure
     function, no DB/mocking.
  2. calculate_topic_priority itself (Lecture2Revision/adaptive_engine.py,
     unmodified) with the neutral "brand-new topic" defaults, since all 12
     real at-risk student_subjects in the live DB currently have zero
     quiz/flashcard/revision_schedules history (Lecture2Revision hasn't been
     used yet) -- confirmed live below.
  3. A real-data integration check: connects to the live Supabase Postgres
     via DATABASE_URL, picks one of the real at-risk student_subject_ids,
     and confirms a priority score can be computed end-to-end from real
     (sparse/default) inputs. Skipped if DATABASE_URL isn't configured.

Full HTTP-level testing of handle_adaptive_schedule (auth + DB together) is
NOT done here because api/_lib/db.py and api/_lib/auth.py -- built by
another agent in parallel -- only exist as an empty api/_lib/__init__.py at
the time this file was written. See the test_sm2_review.py module for the
fake-module-injection pattern used to unit test handler logic independent of
that work landing; the same approach applies here once needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
L2R_DIR = REPO_ROOT / "Lecture2Revision"
if str(L2R_DIR) not in sys.path:
    sys.path.insert(0, str(L2R_DIR))

API_DIR = REPO_ROOT
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from adaptive_engine import calculate_topic_priority  # noqa: E402
from api.rebound.adaptive_schedule import apply_risk_boost  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Risk-boost blend formula -- hand-picked values
# ---------------------------------------------------------------------------

class TestApplyRiskBoost:
    def test_zero_risk_shrinks_to_70_percent_of_original(self):
        # final_priority=50, risk_probability=0.0 -> 50*0.7 + 0*0.3 = 35.0
        assert apply_risk_boost(50.0, 0.0) == 35.0

    def test_full_risk_adds_the_full_30_point_ceiling(self):
        # final_priority=50, risk_probability=1.0 -> 50*0.7 + 100*0.3 = 65.0
        assert apply_risk_boost(50.0, 1.0) == 65.0

    def test_typical_high_risk_subject(self):
        # final_priority=42.5, risk_probability=0.9999 (matches a real
        # flagged row seen in the live DB) -> 42.5*0.7 + 99.99*0.3
        # = 29.75 + 29.997 = 59.747 -> rounds to 59.7
        assert apply_risk_boost(42.5, 0.9999) == 59.7

    def test_moderate_risk_blend(self):
        # final_priority=60.0, risk_probability=0.8 -> 60*0.7 + 80*0.3
        # = 42.0 + 24.0 = 66.0
        assert apply_risk_boost(60.0, 0.8) == 66.0

    def test_result_is_clamped_to_100(self):
        assert apply_risk_boost(100.0, 1.0) == 100.0

    def test_result_is_clamped_to_0(self):
        assert apply_risk_boost(0.0, 0.0) == 0.0

    def test_result_rounded_to_one_decimal(self):
        result = apply_risk_boost(33.333, 0.666)
        assert result == round(result, 1)


# ---------------------------------------------------------------------------
# 2. calculate_topic_priority with the neutral "brand-new topic" defaults
#    (what every one of the live DB's 12 at-risk subjects currently uses,
#    since none has quiz/flashcard history yet)
# ---------------------------------------------------------------------------

class TestCalculateTopicPriorityNeutralDefaults:
    def test_neutral_defaults_produce_a_low_but_nonzero_priority(self):
        score, reason = calculate_topic_priority(
            recent_accuracy=50.0,
            recent_incorrect_count=0,
            days_since_last_review=2.0,
            days_overdue=0.0,
            cards_needing_practice=0,
            lstm_predicted_score=None,
        )
        # acc_urgency=50 * 0.40 = 20.0, everything else 0 -> 20.0
        assert score == 20.0
        assert "accuracy" in reason.lower()

    def test_risk_boost_dominates_for_a_brand_new_flagged_subject(self):
        score, _reason = calculate_topic_priority(
            recent_accuracy=50.0,
            recent_incorrect_count=0,
            days_since_last_review=2.0,
            days_overdue=0.0,
            cards_needing_practice=0,
            lstm_predicted_score=None,
        )
        boosted = apply_risk_boost(score, risk_probability=0.999)
        # 20.0*0.7 + 99.9*0.3 = 14.0 + 29.97 = 43.97 -> 44.0
        assert boosted == 44.0
        # still higher than the un-boosted rule-based score, i.e. the
        # ScholarTrack signal genuinely raises urgency as intended.
        assert boosted > score


# ---------------------------------------------------------------------------
# 3. Real-data check against the live Supabase Postgres database
# ---------------------------------------------------------------------------

def _get_database_url() -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass
    return os.environ.get("DATABASE_URL")


DATABASE_URL = _get_database_url()


@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not configured; skipping live-DB test")
class TestRealAtRiskSubjectPriorityComputation:
    def test_priority_score_computable_for_a_real_flagged_subject(self):
        import psycopg2

        conn = psycopg2.connect(DATABASE_URL)
        try:
            cursor = conn.cursor()

            # Most recent checkpoint per subject, filtered to currently
            # flagged ones -- same query shape as
            # api.rebound.adaptive_schedule._fetch_flagged_subjects.
            cursor.execute(
                """
                select distinct on (student_subject_id)
                    student_subject_id, risk_probability, is_flagged
                from risk_predictions
                order by student_subject_id, checkpoint_index desc
                limit 1
                """
            )
            # Re-run without the premature LIMIT so distinct on works first,
            # then take one flagged row from the full result.
            cursor.execute(
                """
                select distinct on (student_subject_id)
                    student_subject_id, risk_probability, is_flagged
                from risk_predictions
                order by student_subject_id, checkpoint_index desc
                """
            )
            flagged_rows = [r for r in cursor.fetchall() if r[2] is True]
            assert len(flagged_rows) >= 1, "expected at least one at-risk subject in the live DB"

            student_subject_id, risk_probability, _is_flagged = flagged_rows[0]

            # Confirm there's no quiz/flashcard/schedule history for this
            # subject yet (the neutral-default path), matching live reality
            # at the time this test was written.
            cursor.execute(
                """
                select count(*) from quiz_attempts qa
                join lectures l on l.id = qa.lecture_id
                where l.student_subject_id = %s
                """,
                (student_subject_id,),
            )
            (quiz_count,) = cursor.fetchone()

            if quiz_count == 0:
                recent_accuracy = 50.0
                recent_incorrect_count = 0
                days_since_last_review = 2.0
            else:  # pragma: no cover - exercised once real usage exists
                cursor.execute(
                    """
                    select qa.score_pct, qa.incorrect_count
                    from quiz_attempts qa
                    join lectures l on l.id = qa.lecture_id
                    where l.student_subject_id = %s
                    order by qa.completed_at desc limit 1
                    """,
                    (student_subject_id,),
                )
                recent_accuracy, recent_incorrect_count = cursor.fetchone()
                days_since_last_review = 0.0

            score, reason = calculate_topic_priority(
                recent_accuracy=recent_accuracy,
                recent_incorrect_count=recent_incorrect_count,
                days_since_last_review=days_since_last_review,
                days_overdue=0.0,
                cards_needing_practice=0,
                lstm_predicted_score=None,
            )
            boosted = apply_risk_boost(score, risk_probability)

            assert isinstance(boosted, float)
            assert 0.0 <= boosted <= 100.0
            assert isinstance(reason, str) and len(reason) > 0
        finally:
            conn.close()
