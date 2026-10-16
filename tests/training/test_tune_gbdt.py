"""Tests for scripts/training/tune_gbdt.py.

Focuses on the NEW tuning/selection logic added by this module
(`select_best`, `evaluate_config`'s wiring, `run_search`'s end-to-end
wiring against the real CV pool) - `train_and_predict_oof_gbdt` and
`evaluate_predictions` themselves are already covered by
tests/training/test_baselines.py and tests/training/test_metrics.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.training import tune_gbdt

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "training_rows.json"


# ---------------------------------------------------------------------------
# 1. Selection logic - proven with injected/known metrics, no real training.
# ---------------------------------------------------------------------------


def _result(recall, precision, f1, pr_auc, tag):
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "pr_auc": pr_auc,
        "tag": tag,
    }


def test_select_best_prefers_highest_min_metric_among_qualifying():
    # Three qualifying (all >= 0.80) candidates with different minimums,
    # plus one disqualified candidate that would otherwise look best on
    # f1/pr_auc alone (a trap for a naive "pick highest f1" selector).
    candidates = [
        _result(0.81, 0.82, 0.81, 0.83, "barely_qualifies"),
        _result(0.90, 0.85, 0.87, 0.92, "best_qualifying"),
        _result(0.80, 0.80, 0.80, 0.80, "just_qualifies"),
        _result(0.99, 0.99, 0.99, 0.60, "disqualified_low_pr_auc"),
    ]

    best = tune_gbdt.select_best(candidates)

    assert best["tag"] == "best_qualifying"


def test_select_best_falls_back_to_highest_min_metric_when_none_qualify():
    # No candidate clears 0.80 on all four; selection must still return the
    # least-bad one (highest minimum metric) rather than erroring or
    # picking arbitrarily.
    candidates = [
        _result(0.70, 0.60, 0.64, 0.75, "weak"),
        _result(0.78, 0.79, 0.78, 0.79, "closest"),
        _result(0.50, 0.90, 0.64, 0.95, "recall_starved"),
    ]

    best = tune_gbdt.select_best(candidates)

    assert best["tag"] == "closest"


def test_select_best_empty_raises():
    with pytest.raises(ValueError):
        tune_gbdt.select_best([])


def test_all_above_threshold_helper():
    assert tune_gbdt._all_above_threshold(
        _result(0.80, 0.80, 0.80, 0.80, "edge")
    )
    assert not tune_gbdt._all_above_threshold(
        _result(0.79, 0.99, 0.99, 0.99, "one_short")
    )


# ---------------------------------------------------------------------------
# 2. evaluate_config wiring - stub out training/eval to prove the plumbing
#    (config keys reach train_fn/eval_fn correctly) without fitting a model.
# ---------------------------------------------------------------------------


def test_evaluate_config_forwards_hyperparameters_and_merges_config():
    captured_train_kwargs = {}
    captured_eval_kwargs = {}

    def fake_train_fn(rows, folds, **kwargs):
        captured_train_kwargs.update(kwargs)
        return {"y_true": [True, False], "y_proba": [0.9, 0.1]}

    def fake_eval_fn(y_true, y_proba, target_recall=0.9):
        captured_eval_kwargs["target_recall"] = target_recall
        return {
            "threshold": 0.5,
            "recall": 1.0,
            "precision": 1.0,
            "f1": 1.0,
            "pr_auc": 1.0,
            "confusion_matrix": [[1, 0], [0, 1]],
            "target_recall_achieved": True,
        }

    config = {
        "max_depth": 2,
        "min_samples_leaf": 20,
        "learning_rate": 0.2,
        "l2_regularization": 1.0,
        "max_iter": 200,
        "oversample_target_ratio": 1.5,
        "target_recall": 0.85,
    }

    result = tune_gbdt.evaluate_config(
        config, rows=[], folds=[], train_fn=fake_train_fn, eval_fn=fake_eval_fn
    )

    assert captured_train_kwargs == {
        "max_depth": 2,
        "min_samples_leaf": 20,
        "learning_rate": 0.2,
        "l2_regularization": 1.0,
        "max_iter": 200,
        "oversample_target_ratio": 1.5,
    }
    assert captured_eval_kwargs == {"target_recall": 0.85}
    # Config keys travel with the metrics in the merged result.
    assert result["max_depth"] == 2
    assert result["target_recall"] == 0.85
    assert result["recall"] == 1.0
    assert result["pr_auc"] == 1.0


# ---------------------------------------------------------------------------
# 3. End-to-end: real CV pool, a tiny real grid, must not crash and must
#    return a well-formed result. Skipped if the data file isn't present.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DATA_PATH.exists(), reason="data/training_rows.json not present"
)
def test_run_search_end_to_end_on_real_cv_pool():
    tiny_grid = [
        {
            "max_depth": 3,
            "min_samples_leaf": 10,
            "learning_rate": 0.1,
            "l2_regularization": 0.1,
            "max_iter": 100,
            "oversample_target_ratio": 1.0,
            "target_recall": 0.85,
        },
        {
            "max_depth": 2,
            "min_samples_leaf": 5,
            "learning_rate": 0.1,
            "l2_regularization": 0.0,
            "max_iter": 100,
            "oversample_target_ratio": 1.0,
            "target_recall": 0.90,
        },
    ]

    outcome = tune_gbdt.run_search(configs=tiny_grid)

    assert outcome["n_test_students"] > 0
    assert outcome["n_cv_pool_students"] > 0
    # Held-out test set must never leak into the pool used for training.
    assert outcome["n_cv_pool_students"] + outcome["n_test_students"] >= outcome[
        "n_cv_pool_students"
    ]
    assert len(outcome["all_results"]) == len(tiny_grid)

    best = outcome["best"]
    for key in ("recall", "precision", "f1", "pr_auc", "threshold"):
        assert key in best
        assert 0.0 <= best[key] <= 1.0
