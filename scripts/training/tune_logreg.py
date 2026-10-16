"""
Hyperparameter tuning for the Logistic Regression baseline - spec section 5
point 8 follow-up: push recall, precision, F1, and PR-AUC all above 0.80
simultaneously, tuning ONLY against the CV pool (never the held-out test
set carved off by `scripts.training.cv.held_out_test_split`).

Levers searched (see module docstring on `train_and_predict_oof_logreg` and
`evaluate_predictions` for why each matters):
    - `target_recall` passed to `evaluate_predictions` - the recall floor
      the threshold sweep is required to clear. A lower floor lets the
      sweep pick a higher threshold, which tends to raise precision/F1.
    - `C` (inverse regularization strength) passed to LogisticRegression.
    - `oversample_target_ratio` passed through to
      `cv.oversample_training_rows` - how aggressively at-risk
      student-subject trajectories are oversampled at train time.

`run_logreg_tuning` trains/evaluates one OOF run per (C, oversample_ratio)
combination using `grouped_student_kfold` over `cv_pool_ids` only, then -
for each trained run - evaluates it at every candidate `target_recall`
(re-evaluating thresholds is cheap: it's just a sweep over the same OOF
y_proba/y_true, no retraining). `select_best_config` picks the single best
(C, oversample_ratio, target_recall) combination by maximizing the minimum
of the four metrics (recall, precision, f1, pr_auc) - "all four clearly
above 80%" is best captured by the worst of the four, not their average,
since an average can hide one metric failing the bar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from scripts.training.baselines import train_and_predict_oof_logreg
from scripts.training.cv import grouped_student_kfold, held_out_test_split
from scripts.training.metrics import evaluate_predictions

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "training_rows.json"

# Search grid. Kept modest in size since each (C, ratio) pair triggers a
# full 5-fold retrain; target_recall is swept "for free" on top of each.
DEFAULT_C_GRID: List[float] = [0.01, 0.1, 1.0, 10.0]
DEFAULT_OVERSAMPLE_RATIO_GRID: List[float] = [0.5, 1.0, 1.5, 2.0]
DEFAULT_TARGET_RECALL_GRID: List[float] = [0.80, 0.85, 0.90]


def load_cv_pool_rows(data_path: Path = DATA_PATH) -> tuple:
    """Load training_rows.json, carve off the held-out test set via
    `held_out_test_split`, and return (cv_pool_rows, cv_pool_ids,
    student_at_risk_for_pool) - rows restricted to `cv_pool_ids` only.

    This is the ONLY place test_ids is allowed to be computed in this
    module; it is discarded immediately after filtering and never used
    again (never loaded into a model, never scored).
    """
    rows = json.loads(data_path.read_text())

    student_at_risk: Dict[str, bool] = {}
    for row in rows:
        sid = row["student_id"]
        student_at_risk[sid] = student_at_risk.get(sid, False) or bool(row["at_risk"])

    student_ids = list(student_at_risk.keys())
    cv_pool_ids, _test_ids = held_out_test_split(student_ids, student_at_risk)
    # _test_ids deliberately unused beyond this point - see module docstring.

    pool_id_set = set(cv_pool_ids)
    cv_pool_rows = [row for row in rows if row["student_id"] in pool_id_set]
    pool_student_at_risk = {
        sid: at_risk for sid, at_risk in student_at_risk.items() if sid in pool_id_set
    }
    return cv_pool_rows, cv_pool_ids, pool_student_at_risk


def _n_splits_for_pool(pool_student_at_risk: Dict[str, bool], requested: int = 5) -> int:
    """Fall back to fewer folds if the pool doesn't have enough at-risk
    students for `requested`-fold stratified splitting (matching the
    discipline documented in cv.py's grouped_student_kfold docstring)."""
    n_at_risk = sum(1 for v in pool_student_at_risk.values() if v)
    return min(requested, n_at_risk) if n_at_risk > 0 else requested


def run_logreg_tuning(
    cv_pool_rows: List[dict],
    cv_pool_ids: List[str],
    pool_student_at_risk: Dict[str, bool],
    c_grid: Optional[List[float]] = None,
    oversample_ratio_grid: Optional[List[float]] = None,
    target_recall_grid: Optional[List[float]] = None,
    n_splits: int = 5,
    seed: int = 42,
) -> List[dict]:
    """Grid-search C x oversample_target_ratio x target_recall on the CV
    pool only. Returns a flat list of result dicts, one per combination,
    each holding {C, oversample_target_ratio, target_recall, recall,
    precision, f1, pr_auc, threshold, target_recall_achieved}.
    """
    c_grid = c_grid if c_grid is not None else DEFAULT_C_GRID
    oversample_ratio_grid = (
        oversample_ratio_grid if oversample_ratio_grid is not None else DEFAULT_OVERSAMPLE_RATIO_GRID
    )
    target_recall_grid = (
        target_recall_grid if target_recall_grid is not None else DEFAULT_TARGET_RECALL_GRID
    )

    effective_n_splits = _n_splits_for_pool(pool_student_at_risk, n_splits)
    folds = grouped_student_kfold(
        cv_pool_ids, pool_student_at_risk, n_splits=effective_n_splits, seed=seed
    )

    results: List[dict] = []
    for C in c_grid:
        for oversample_ratio in oversample_ratio_grid:
            oof = train_and_predict_oof_logreg(
                cv_pool_rows,
                folds,
                C=C,
                oversample_target_ratio=oversample_ratio,
            )
            for target_recall in target_recall_grid:
                metrics = evaluate_predictions(
                    oof["y_true"], oof["y_proba"], target_recall=target_recall
                )
                results.append(
                    {
                        "C": C,
                        "oversample_target_ratio": oversample_ratio,
                        "target_recall": target_recall,
                        "n_splits": effective_n_splits,
                        **metrics,
                    }
                )
    return results


def select_best_config(results: List[dict]) -> dict:
    """Pick the single best config by maximizing the minimum of the four
    headline metrics (recall, precision, f1, pr_auc) - this directly
    captures "all four clearly above 80%", since the worst-performing
    metric is what determines whether the 80% bar is cleared across the
    board. Ties are broken by the sum of the four metrics (prefer configs
    that are also higher on the other three, not just tied on the worst
    one).

    Raises ValueError on an empty `results` list.
    """
    if not results:
        raise ValueError("select_best_config requires at least one result")

    def _score(r: dict) -> tuple:
        four = (r["recall"], r["precision"], r["f1"], r["pr_auc"])
        return (min(four), sum(four))

    return max(results, key=_score)


def main() -> None:
    cv_pool_rows, cv_pool_ids, pool_student_at_risk = load_cv_pool_rows()
    n_at_risk = sum(1 for v in pool_student_at_risk.values() if v)
    print(
        f"CV pool: {len(cv_pool_ids)} students "
        f"({n_at_risk} at-risk, {len(cv_pool_ids) - n_at_risk} not-at-risk), "
        f"{len(cv_pool_rows)} rows"
    )

    results = run_logreg_tuning(cv_pool_rows, cv_pool_ids, pool_student_at_risk)
    results_sorted = sorted(
        results,
        key=lambda r: (min(r["recall"], r["precision"], r["f1"], r["pr_auc"])),
        reverse=True,
    )

    print(f"\nTop 10 of {len(results)} configs (by min of the four metrics):")
    for r in results_sorted[:10]:
        print(
            f"  C={r['C']:<6} ratio={r['oversample_target_ratio']:<4} "
            f"target_recall={r['target_recall']:<4} -> "
            f"recall={r['recall']:.3f} precision={r['precision']:.3f} "
            f"f1={r['f1']:.3f} pr_auc={r['pr_auc']:.3f} "
            f"threshold={r['threshold']:.2f}"
        )

    best = select_best_config(results)
    print("\nBest config:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
