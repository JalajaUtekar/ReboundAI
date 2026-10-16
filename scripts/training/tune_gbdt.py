"""
ScholarTrack Gradient Boosting hyperparameter tuning.

Searches over `HistGradientBoostingClassifier` hyperparameters,
`oversample_training_rows`'s `target_ratio`, and `evaluate_predictions`'s
`target_recall` floor, using ONLY the CV pool returned by
`scripts.training.cv.held_out_test_split` (the `cv_pool_ids` half - the
`test_ids` half is reserved for a single final evaluation run elsewhere
and must never be touched here).

All model fitting and metric computation happens via grouped, student-level
CV folds (`grouped_student_kfold` on the pool), exactly like
`baselines.train_and_predict_oof_gbdt` / `metrics.evaluate_predictions` are
used elsewhere in this project - this module adds a search loop around
those two functions, it does not reimplement them.

Selection criterion: among configurations whose CV metrics (recall,
precision, f1, pr_auc) are ALL >= `MIN_METRIC_THRESHOLD` (0.80), prefer the
one with the highest minimum metric (the "weakest link" is as strong as
possible). If no configuration clears all four thresholds, report the
configuration with the highest minimum-metric anyway (best honest attempt).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.training.baselines import train_and_predict_oof_gbdt
from scripts.training.cv import grouped_student_kfold, held_out_test_split
from scripts.training.metrics import evaluate_predictions

MIN_METRIC_THRESHOLD = 0.80

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "training_rows.json"

_METRIC_KEYS = ("recall", "precision", "f1", "pr_auc")


def _min_metric(metrics: Dict[str, float]) -> float:
    return min(metrics[key] for key in _METRIC_KEYS)


def _all_above_threshold(metrics: Dict[str, float], threshold: float = MIN_METRIC_THRESHOLD) -> bool:
    return all(metrics[key] >= threshold for key in _METRIC_KEYS)


def default_candidate_configs() -> List[dict]:
    """The grid of GBDT configurations this module searches by default.

    Each dict holds: `max_depth`, `min_samples_leaf`, `learning_rate`,
    `l2_regularization`, `max_iter` (forwarded to
    `train_and_predict_oof_gbdt`), `oversample_target_ratio` (forwarded as
    `oversample_training_rows`'s `target_ratio`), and `target_recall`
    (forwarded to `evaluate_predictions`).

    Kept deliberately modest in size (rather than a full cartesian product
    of every lever times every other lever) so a full run finishes in a
    reasonable time on this dataset - informed by the priority ordering in
    the task brief (target_recall and depth/leaf-size first, then
    learning_rate/l2/max_iter/oversample ratio as refinements around the
    best region found).
    """
    configs: List[dict] = []

    # Stage 1: vary target_recall x max_depth x min_samples_leaf around the
    # documented baseline (max_depth=3, min_samples_leaf=5, target_recall=0.90).
    for target_recall in (0.80, 0.85, 0.90):
        for max_depth in (2, 3, 4, None):
            for min_samples_leaf in (5, 10, 20):
                configs.append(
                    {
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "learning_rate": 0.1,
                        "l2_regularization": 0.0,
                        "max_iter": 100,
                        "oversample_target_ratio": 1.0,
                        "target_recall": target_recall,
                    }
                )

    # Stage 2: vary learning_rate x l2_regularization x max_iter at a few
    # promising target_recall/depth combinations.
    for target_recall in (0.80, 0.85):
        for learning_rate in (0.05, 0.1, 0.2):
            for l2_regularization in (0.0, 0.1, 1.0):
                for max_iter in (100, 200):
                    configs.append(
                        {
                            "max_depth": 3,
                            "min_samples_leaf": 10,
                            "learning_rate": learning_rate,
                            "l2_regularization": l2_regularization,
                            "max_iter": max_iter,
                            "oversample_target_ratio": 1.0,
                            "target_recall": target_recall,
                        }
                    )

    # Stage 3: vary oversample target_ratio at a couple of target_recall
    # floors.
    for target_recall in (0.80, 0.85):
        for oversample_target_ratio in (0.5, 1.0, 1.5, 2.0):
            configs.append(
                {
                    "max_depth": 3,
                    "min_samples_leaf": 10,
                    "learning_rate": 0.1,
                    "l2_regularization": 0.1,
                    "max_iter": 100,
                    "oversample_target_ratio": oversample_target_ratio,
                    "target_recall": target_recall,
                }
            )

    return configs


def evaluate_config(
    config: dict,
    rows: List[dict],
    folds: List[Tuple[List[str], List[str]]],
    train_fn=train_and_predict_oof_gbdt,
    eval_fn=evaluate_predictions,
) -> Dict[str, float]:
    """Train+predict OOF under `folds` with `config`'s hyperparameters,
    then evaluate at `config["target_recall"]`.

    `train_fn`/`eval_fn` are injectable seams purely for testing the
    selection logic below without actually fitting a GBDT model.

    Returns the dict from `evaluate_predictions`, plus the config's own
    keys merged in (so the full configuration travels with its result).
    """
    oof = train_fn(
        rows,
        folds,
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        learning_rate=config["learning_rate"],
        l2_regularization=config["l2_regularization"],
        max_iter=config["max_iter"],
        oversample_target_ratio=config["oversample_target_ratio"],
    )
    metrics = eval_fn(
        oof["y_true"], oof["y_proba"], target_recall=config["target_recall"]
    )
    return {**config, **metrics}


def select_best(results: List[dict]) -> dict:
    """Pick the single best result from a list of `evaluate_config`-shaped
    dicts (each must carry recall/precision/f1/pr_auc keys).

    Preference order:
        1. Among results where ALL FOUR of recall/precision/f1/pr_auc are
           >= MIN_METRIC_THRESHOLD, the one with the highest minimum metric
           ("all four clearly above 80%, not just barely" per the task
           brief).
        2. If none qualifies, the result with the highest minimum metric
           overall (best honest attempt, reported as a miss).

    Raises ValueError on an empty `results` list.
    """
    if not results:
        raise ValueError("select_best requires at least one result")

    qualifying = [r for r in results if _all_above_threshold(r)]
    pool = qualifying if qualifying else results
    return max(pool, key=_min_metric)


def run_search(
    configs: Optional[List[dict]] = None,
    data_path: Path = DATA_PATH,
    n_splits: int = 5,
    test_size: float = 0.2,
    held_out_seed: int = 7,
    cv_seed: int = 42,
) -> dict:
    """End-to-end tuning run: load data, carve the held-out test split,
    run grouped CV on the `cv_pool_ids` half only, search `configs`
    (defaulting to `default_candidate_configs()`), and return the best
    config's full result plus bookkeeping (`all_results`, pool sizes).

    NEVER touches `test_ids` - only `cv_pool_ids` feeds `rows`/`folds`.
    """
    rows = json.loads(Path(data_path).read_text())

    student_at_risk: Dict[str, bool] = {}
    for row in rows:
        sid = row["student_id"]
        student_at_risk[sid] = student_at_risk.get(sid, False) or bool(row["at_risk"])

    student_ids = list(student_at_risk.keys())
    cv_pool_ids, test_ids = held_out_test_split(
        student_ids, student_at_risk, test_size=test_size, seed=held_out_seed
    )

    pool_id_set = set(cv_pool_ids)
    pool_rows = [row for row in rows if row["student_id"] in pool_id_set]

    pool_at_risk_count = sum(
        1 for sid in cv_pool_ids if student_at_risk.get(sid, False)
    )
    effective_n_splits = n_splits if pool_at_risk_count >= n_splits else max(
        2, pool_at_risk_count
    )

    folds = grouped_student_kfold(
        cv_pool_ids, student_at_risk, n_splits=effective_n_splits, seed=cv_seed
    )

    search_configs = configs if configs is not None else default_candidate_configs()

    all_results = [
        evaluate_config(config, pool_rows, folds) for config in search_configs
    ]
    best = select_best(all_results)

    return {
        "best": best,
        "all_results": all_results,
        "n_cv_pool_students": len(cv_pool_ids),
        "n_test_students": len(test_ids),
        "n_splits_used": effective_n_splits,
    }


def _format_report(outcome: dict) -> str:
    best = outcome["best"]
    all_clear = _all_above_threshold(best)
    lines = [
        f"CV pool students: {outcome['n_cv_pool_students']} "
        f"(held-out test students, untouched: {outcome['n_test_students']})",
        f"Folds used: {outcome['n_splits_used']}",
        f"Configs searched: {len(outcome['all_results'])}",
        "",
        "All four metrics >= 0.80: " + ("YES" if all_clear else "NO (best attempt reported)"),
        "",
        "Best configuration:",
        f"  max_depth={best['max_depth']}, min_samples_leaf={best['min_samples_leaf']}, "
        f"learning_rate={best['learning_rate']}, l2_regularization={best['l2_regularization']}, "
        f"max_iter={best['max_iter']}, oversample_target_ratio={best['oversample_target_ratio']}, "
        f"target_recall={best['target_recall']}",
        f"  threshold={best['threshold']:.2f}",
        f"  recall={best['recall']:.4f}  precision={best['precision']:.4f}  "
        f"f1={best['f1']:.4f}  pr_auc={best['pr_auc']:.4f}",
        f"  target_recall_achieved={best['target_recall_achieved']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    outcome = run_search()
    print(_format_report(outcome))
