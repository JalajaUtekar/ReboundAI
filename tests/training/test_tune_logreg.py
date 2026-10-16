"""
Tests for scripts/training/tune_logreg.py.

Per the task instructions, these focus on OUR new tuning/selection logic:
    1. The real end-to-end tuning search runs against the actual CV pool
       (data/training_rows.json, split via held_out_test_split) without
       crashing and returns well-formed result dicts - skipped if the data
       file isn't present.
    2. `select_best_config`'s selection logic actually picks the
       best-by-criterion config out of a set of fake/injected candidates,
       proving the selection logic itself rather than just "something ran".
    3. `load_cv_pool_rows` never includes any held-out test student's rows
       in what it returns (the core safety property this whole exercise
       exists to protect).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.training import tune_logreg

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "training_rows.json"


# ---------------------------------------------------------------------------
# 1. Selection logic - fake/injected candidates, no training involved.
# ---------------------------------------------------------------------------


def test_select_best_config_picks_highest_minimum_metric():
    """A config where all four metrics clear 0.80 should beat one that is
    higher on average but fails one metric badly."""
    candidates = [
        {
            "C": 1.0,
            "oversample_target_ratio": 1.0,
            "target_recall": 0.90,
            "recall": 0.95,
            "precision": 0.60,  # drags the min way down
            "f1": 0.73,
            "pr_auc": 0.95,
        },
        {
            "C": 0.1,
            "oversample_target_ratio": 0.5,
            "target_recall": 0.80,
            "recall": 0.83,
            "precision": 0.85,
            "f1": 0.84,
            "pr_auc": 0.93,  # every metric clears 0.80
        },
        {
            "C": 10.0,
            "oversample_target_ratio": 2.0,
            "target_recall": 0.85,
            "recall": 0.70,  # fails the recall bar
            "precision": 0.90,
            "f1": 0.79,
            "pr_auc": 0.88,
        },
    ]

    best = tune_logreg.select_best_config(candidates)

    assert best["C"] == 0.1
    assert best["oversample_target_ratio"] == 0.5
    assert best["target_recall"] == 0.80


def test_select_best_config_breaks_ties_with_metric_sum():
    """Two candidates tied on the minimum metric should be broken by the
    sum of all four metrics (prefer the one that's also better elsewhere)."""
    low_min_but_better_elsewhere = {
        "C": 1.0,
        "oversample_target_ratio": 1.0,
        "target_recall": 0.85,
        "recall": 0.81,
        "precision": 0.95,
        "f1": 0.87,
        "pr_auc": 0.95,
    }
    same_min_but_worse_elsewhere = {
        "C": 5.0,
        "oversample_target_ratio": 1.5,
        "target_recall": 0.80,
        "recall": 0.81,
        "precision": 0.82,
        "f1": 0.81,
        "pr_auc": 0.83,
    }

    best = tune_logreg.select_best_config(
        [low_min_but_better_elsewhere, same_min_but_worse_elsewhere]
    )

    assert best["C"] == 1.0


def test_select_best_config_raises_on_empty_list():
    with pytest.raises(ValueError):
        tune_logreg.select_best_config([])


# ---------------------------------------------------------------------------
# 2. run_logreg_tuning wiring - stub out training/evaluation so this test
#    proves the grid-iteration logic, not sklearn behavior.
# ---------------------------------------------------------------------------


def test_run_logreg_tuning_covers_full_grid(monkeypatch):
    calls = []

    def _fake_train_and_predict_oof_logreg(rows, folds, C=1.0, oversample_target_ratio=1.0, feature_names=None):
        calls.append((C, oversample_target_ratio))
        return {"y_true": [True, False], "y_proba": [0.9, 0.1]}

    def _fake_evaluate_predictions(y_true, y_proba, target_recall=0.90):
        return {
            "threshold": 0.5,
            "recall": 1.0,
            "precision": 1.0,
            "f1": 1.0,
            "pr_auc": 1.0,
            "confusion_matrix": [[1, 0], [0, 1]],
            "target_recall_achieved": True,
        }

    def _fake_grouped_student_kfold(student_ids, student_at_risk, n_splits=5, seed=42):
        return [(student_ids, student_ids)]

    monkeypatch.setattr(
        tune_logreg, "train_and_predict_oof_logreg", _fake_train_and_predict_oof_logreg
    )
    monkeypatch.setattr(tune_logreg, "evaluate_predictions", _fake_evaluate_predictions)
    monkeypatch.setattr(tune_logreg, "grouped_student_kfold", _fake_grouped_student_kfold)

    student_at_risk = {f"S{i}": (i < 6) for i in range(10)}
    student_ids = list(student_at_risk.keys())

    c_grid = [0.1, 1.0]
    ratio_grid = [0.5, 1.0]
    recall_grid = [0.80, 0.90]

    results = tune_logreg.run_logreg_tuning(
        cv_pool_rows=[],
        cv_pool_ids=student_ids,
        pool_student_at_risk=student_at_risk,
        c_grid=c_grid,
        oversample_ratio_grid=ratio_grid,
        target_recall_grid=recall_grid,
    )

    # One training call per (C, ratio) pair, not per target_recall (reuses
    # the same OOF predictions across the target_recall sweep).
    assert len(calls) == len(c_grid) * len(ratio_grid)
    assert set(calls) == {(0.1, 0.5), (0.1, 1.0), (1.0, 0.5), (1.0, 1.0)}

    # One result row per (C, ratio, target_recall) combination.
    assert len(results) == len(c_grid) * len(ratio_grid) * len(recall_grid)
    for r in results:
        assert r["C"] in c_grid
        assert r["oversample_target_ratio"] in ratio_grid
        assert r["target_recall"] in recall_grid
        assert r["recall"] == 1.0  # from the fake evaluate_predictions


# ---------------------------------------------------------------------------
# 3. Real data integration - skipped if data isn't present.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DATA_PATH.exists(), reason="data/training_rows.json not present")
def test_load_cv_pool_rows_excludes_held_out_test_students():
    import json

    from scripts.training.cv import held_out_test_split

    all_rows = json.loads(DATA_PATH.read_text())
    student_at_risk = {}
    for row in all_rows:
        sid = row["student_id"]
        student_at_risk[sid] = student_at_risk.get(sid, False) or bool(row["at_risk"])
    student_ids = list(student_at_risk.keys())
    _cv_pool_ids, test_ids = held_out_test_split(student_ids, student_at_risk)
    test_id_set = set(test_ids)

    cv_pool_rows, cv_pool_ids, pool_student_at_risk = tune_logreg.load_cv_pool_rows()

    assert test_id_set.isdisjoint(set(cv_pool_ids))
    assert all(row["student_id"] not in test_id_set for row in cv_pool_rows)
    assert test_id_set.isdisjoint(set(pool_student_at_risk.keys()))


@pytest.mark.skipif(not DATA_PATH.exists(), reason="data/training_rows.json not present")
def test_run_logreg_tuning_end_to_end_on_real_pool_small_grid():
    """Smoke test: a tiny real grid runs against the real CV pool without
    crashing and returns well-formed result dicts."""
    cv_pool_rows, cv_pool_ids, pool_student_at_risk = tune_logreg.load_cv_pool_rows()

    results = tune_logreg.run_logreg_tuning(
        cv_pool_rows,
        cv_pool_ids,
        pool_student_at_risk,
        c_grid=[0.1],
        oversample_ratio_grid=[1.0],
        target_recall_grid=[0.80],
    )

    assert len(results) == 1
    r = results[0]
    for key in (
        "C",
        "oversample_target_ratio",
        "target_recall",
        "recall",
        "precision",
        "f1",
        "pr_auc",
        "threshold",
    ):
        assert key in r
    for metric_key in ("recall", "precision", "f1", "pr_auc"):
        assert 0.0 <= r[metric_key] <= 1.0

    best = tune_logreg.select_best_config(results)
    assert best == r
