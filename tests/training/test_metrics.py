"""Tests for scripts/training/metrics.py."""

from __future__ import annotations

import pytest

from scripts.training.metrics import evaluate_predictions


class TestEvaluatePredictions:
    def test_perfect_separation_gives_recall_and_precision_1(self):
        y_true = [False] * 10 + [True] * 10
        y_proba = [0.1] * 10 + [0.9] * 10

        result = evaluate_predictions(y_true, y_proba, target_recall=0.90)

        assert result["recall"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)
        assert result["target_recall_achieved"] is True
        assert result["pr_auc"] == pytest.approx(1.0)
        # confusion_matrix = [[tn, fp], [fn, tp]]
        assert result["confusion_matrix"] == [[10, 0], [0, 10]]

    def test_hand_computed_confusion_matrix_at_known_threshold(self):
        # 5 negatives, 5 positives. Proba values chosen so that at
        # threshold 0.5, exactly 1 positive is missed (FN) and all
        # negatives are correctly rejected.
        y_true = [False, False, False, False, False, True, True, True, True, True]
        y_proba = [0.05, 0.10, 0.15, 0.20, 0.25, 0.60, 0.65, 0.70, 0.75, 0.40]
        # At threshold 0.5: predictions = [0,0,0,0,0,1,1,1,1,0]
        # TP=4 (idx 5-8), FN=1 (idx 9, proba 0.40), TN=5, FP=0
        # recall = 4/5 = 0.8, precision = 4/4 = 1.0

        # Ask for target_recall=0.8 - threshold 0.5 should qualify exactly.
        result = evaluate_predictions(y_true, y_proba, target_recall=0.8)

        assert result["target_recall_achieved"] is True
        assert result["recall"] >= 0.8
        # Since threshold 0.5 gives recall exactly 0.8 with precision 1.0,
        # and higher thresholds would drop recall below 0.8 (losing another
        # TP), the chosen threshold's precision should be 1.0.
        assert result["precision"] == pytest.approx(1.0)

    def test_falls_back_when_target_recall_unreachable(self):
        # Worst-case separation: probabilities don't correlate with labels
        # at all, so recall=1.0 is only achievable at the very lowest
        # threshold, which also lets in lots of false positives. Construct
        # a case where NO threshold in [0.10, 0.90] achieves recall=0.99
        # (impossible exactly, but let's force an unreachable target like
        # 1.5 which can never be satisfied).
        y_true = [False] * 5 + [True] * 5
        y_proba = [0.3, 0.4, 0.5, 0.6, 0.7, 0.35, 0.45, 0.55, 0.65, 0.75]

        result = evaluate_predictions(y_true, y_proba, target_recall=1.5)

        assert result["target_recall_achieved"] is False
        # Still returns a usable result dict.
        assert "threshold" in result
        assert 0.10 <= result["threshold"] <= 0.90

    def test_does_not_include_bare_accuracy_key(self):
        y_true = [False, True]
        y_proba = [0.2, 0.8]
        result = evaluate_predictions(y_true, y_proba)
        assert "accuracy" not in result

    def test_returns_expected_keys(self):
        y_true = [False, True, False, True]
        y_proba = [0.2, 0.8, 0.3, 0.6]
        result = evaluate_predictions(y_true, y_proba)
        expected_keys = {
            "threshold",
            "recall",
            "precision",
            "f1",
            "pr_auc",
            "confusion_matrix",
            "target_recall_achieved",
        }
        assert set(result.keys()) == expected_keys

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            evaluate_predictions([], [])

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            evaluate_predictions([True, False], [0.5])
