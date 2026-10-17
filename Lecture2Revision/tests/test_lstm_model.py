"""
Unit tests for LSTM sequence tensor construction, feature shapes, and graceful fallbacks.
"""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lstm_model import LSTMStudentPredictor, FEATURE_NAMES, TENSORFLOW_AVAILABLE


def test_create_sequences_from_records_tensor_shapes():
    predictor = LSTMStudentPredictor(sequence_length=4)

    # Create 10 sequential chronological records
    records = []
    for i in range(10):
        records.append({
            "score_pct": 50.0 + (i * 4),
            "total_questions": 10,
            "correct_count": 5 + (i // 2),
            "incorrect_count": 5 - (i // 2),
            "time_spent_seconds": 60.0 - i,
            "interval_days": 1.0,
        })

    X, y, baselines = predictor.create_sequences_from_records(records)

    # 10 records with sequence length 4 -> 10 - 4 = 6 sequences
    assert X.shape == (6, 4, len(FEATURE_NAMES))
    assert y.shape == (6,)
    assert baselines.shape == (6,)

    # Verify first target corresponds to record 4 score
    assert y[0] == records[4]["score_pct"]
    # Verify baseline is the last item of the input sequence (record 3 score)
    assert baselines[0] == records[3]["score_pct"]


def test_create_sequences_insufficient_records():
    predictor = LSTMStudentPredictor(sequence_length=5)
    records = [{"score_pct": 80.0, "total_questions": 10, "correct_count": 8, "incorrect_count": 2}]

    X, y, baselines = predictor.create_sequences_from_records(records)
    assert len(X) == 0
    assert len(y) == 0


def test_predictor_graceful_fallback():
    predictor = LSTMStudentPredictor()
    # Predict without model trained should return None, not raise exception
    res = predictor.predict_next_score("nonexistent_student")
    assert res is None
