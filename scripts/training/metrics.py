"""
ScholarTrack evaluation metrics - spec section 5, point 10.

"Metrics reported: recall, precision, F1, PR-AUC, and a confusion matrix -
never bare accuracy, since always predicting 'not at risk' already scores
85-95% on this class balance and would be a meaningless win."

`evaluate_predictions` sweeps candidate decision thresholds and picks the
highest threshold that still clears a target recall floor (spec section 5
point 3: "Cutoff threshold is tuned per the CV protocol against a target
recall (>=88-92%), not left at the default 0.5"), so recall is defended
first and precision is maximized only subject to that floor.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

# Threshold sweep: 0.10 to 0.90 inclusive, step 0.02 (41 candidate
# thresholds). Fine enough to find a precise recall/precision trade-off
# point without being a meaningful runtime cost at this dataset size.
_THRESHOLD_MIN = 0.10
_THRESHOLD_MAX = 0.90
_THRESHOLD_STEP = 0.02


def _candidate_thresholds() -> np.ndarray:
    # Add a half-step epsilon so the inclusive upper bound (0.90) survives
    # floating point arange truncation.
    return np.arange(_THRESHOLD_MIN, _THRESHOLD_MAX + _THRESHOLD_STEP / 2, _THRESHOLD_STEP)


def _recall_precision_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return recall, precision, f1


def evaluate_predictions(
    y_true: List[bool],
    y_proba: List[float],
    target_recall: float = 0.90,
) -> dict:
    """Sweep thresholds in [0.10, 0.90] (step 0.02) and pick the HIGHEST
    threshold whose recall still clears `target_recall`, maximizing
    precision subject to that recall floor.

    If no threshold in the sweep achieves target_recall, falls back to the
    threshold that gets closest to it (ties broken by higher recall, since
    missing the floor should err toward catching more at-risk students),
    and sets `target_recall_achieved` to False so callers can tell the
    difference.

    Returns a dict with: threshold, recall, precision, f1, pr_auc,
    confusion_matrix ([[tn, fp], [fn, tp]]), target_recall_achieved.

    Bare accuracy is deliberately NOT included in the returned dict - per
    spec section 5 point 10, it's a meaningless headline metric on this
    class balance (always predicting "not at risk" scores 85-95%).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    if len(y_true_arr) != len(y_proba_arr):
        raise ValueError("y_true and y_proba must be the same length")
    if len(y_true_arr) == 0:
        raise ValueError("evaluate_predictions requires at least one example")

    pr_auc = float(average_precision_score(y_true_arr, y_proba_arr))

    best_qualifying: dict | None = None  # highest-threshold result meeting target_recall
    closest_fallback: dict | None = None  # best result when none qualifies
    closest_gap = None

    for threshold in _candidate_thresholds():
        y_pred = (y_proba_arr >= threshold).astype(int)
        recall, precision, f1 = _recall_precision_f1(y_true_arr, y_pred)

        candidate = {
            "threshold": float(threshold),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

        if recall >= target_recall:
            # Among qualifying thresholds, prefer the highest (best precision
            # tends to increase with threshold), so keep overwriting as the
            # sweep walks upward.
            if best_qualifying is None or threshold > best_qualifying["threshold"]:
                best_qualifying = candidate

        gap = abs(recall - target_recall)
        if closest_gap is None or gap < closest_gap or (
            gap == closest_gap and recall > closest_fallback["recall"]
        ):
            closest_gap = gap
            closest_fallback = candidate

    if best_qualifying is not None:
        chosen = best_qualifying
        target_recall_achieved = True
    else:
        chosen = closest_fallback
        target_recall_achieved = False

    y_pred_final = (y_proba_arr >= chosen["threshold"]).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred_final, labels=[0, 1])

    return {
        "threshold": chosen["threshold"],
        "recall": chosen["recall"],
        "precision": chosen["precision"],
        "f1": chosen["f1"],
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "target_recall_achieved": target_recall_achieved,
    }
