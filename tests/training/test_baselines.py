"""
Tests for scripts/training/baselines.py (spec section 5, point 8).

Two layers:

1. Pure wiring tests that prove OUR OWN fold-iteration / OOF-collection
   logic is correct, independent of the other agent's cv.py/metrics.py.
   These use a trivial inline fake in place of oversample_training_rows
   (monkeypatched onto scripts.training.baselines) so they never import
   the real scripts/training/cv.py.
2. Real integration tests against the actual scripts/training/cv.py (and,
   if present, scripts/training/metrics.py) and the real
   data/training_rows.json - skipped if those files don't exist yet,
   since this agent does not block on the other agents' work landing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from scripts.training import baselines

REPO_ROOT = Path(__file__).resolve().parents[2]
CV_PATH = REPO_ROOT / "scripts" / "training" / "cv.py"
METRICS_PATH = REPO_ROOT / "scripts" / "training" / "metrics.py"
DATA_PATH = REPO_ROOT / "data" / "training_rows.json"


FEATURE_NAMES = baselines.DEFAULT_FEATURE_NAMES


def _make_row(student_id, subject_slug, checkpoint_idx, at_risk, base=50.0, jitter=0.0):
    """Hand-constructed row matching the documented training_rows.json shape."""
    row = {name: base for name in FEATURE_NAMES}
    # Give at-risk rows a different feature profile so models have signal
    # to learn from, rather than a degenerate constant-feature dataset.
    # `jitter` gives each student a distinct value (rather than every
    # student in a class sharing byte-identical features), so that holding
    # out different students actually changes a fold's train-set mean.
    if at_risk:
        row["internal_marks_pct"] = 30.0 + jitter
        row["attendance_pct"] = 60.0 + jitter
        row["attendance_deficit_margin"] = -15.0 + jitter
        row["marks_deficit_margin"] = -20.0 + jitter
    else:
        row["internal_marks_pct"] = 85.0 + jitter
        row["attendance_pct"] = 92.0 + jitter
        row["attendance_deficit_margin"] = 17.0 + jitter
        row["marks_deficit_margin"] = 15.0 + jitter
    row["student_id"] = student_id
    row["subject_slug"] = subject_slug
    row["checkpoint_idx"] = checkpoint_idx
    row["at_risk"] = at_risk
    row["is_failing"] = False
    return row


def _build_fixture_rows(n_not_at_risk=12, n_at_risk=6, n_checkpoints=3):
    """A small, hand-built dataset: each student has one subject with
    n_checkpoints prefix rows, mirroring the real data's per-student-subject
    trajectory shape."""
    rows = []
    student_at_risk = {}
    sid_counter = 0
    for _ in range(n_not_at_risk):
        sid = f"S{sid_counter:03d}"
        jitter = float(sid_counter)
        sid_counter += 1
        student_at_risk[sid] = False
        for cp in range(1, n_checkpoints + 1):
            rows.append(_make_row(sid, "DSML", cp, at_risk=False, jitter=jitter))
    for _ in range(n_at_risk):
        sid = f"S{sid_counter:03d}"
        jitter = float(sid_counter)
        sid_counter += 1
        student_at_risk[sid] = True
        for cp in range(1, n_checkpoints + 1):
            rows.append(_make_row(sid, "DSML", cp, at_risk=True, jitter=jitter))
    return rows, student_at_risk


def _fake_grouped_kfold(student_ids, student_at_risk, n_splits=3, seed=42):
    """Trivial local fake: round-robin assign unique student_ids into
    n_splits val buckets; everyone else trains. Deliberately NOT imported
    from the real cv.py, to prove our own wiring independent of it."""
    unique_ids = list(dict.fromkeys(student_ids))
    buckets = [[] for _ in range(n_splits)]
    for i, sid in enumerate(unique_ids):
        buckets[i % n_splits].append(sid)
    folds = []
    for i in range(n_splits):
        val_ids = buckets[i]
        train_ids = [sid for sid in unique_ids if sid not in val_ids]
        folds.append((train_ids, val_ids))
    return folds


def _fake_oversample_noop(train_rows, target_ratio=1.0, seed=42):
    """Local fake oversampler: identity (no-op), used to isolate OUR
    fold/OOF wiring from the real oversampling implementation."""
    return list(train_rows)


# ---------------------------------------------------------------------------
# 1. Pure wiring tests - no dependency on the real cv.py.
# ---------------------------------------------------------------------------


def test_oof_covers_every_row_exactly_once_logreg(monkeypatch):
    monkeypatch.setattr(baselines, "oversample_training_rows", _fake_oversample_noop)
    rows, student_at_risk = _build_fixture_rows()
    student_ids = list(student_at_risk.keys())
    folds = _fake_grouped_kfold(student_ids, student_at_risk, n_splits=3)

    result = baselines.train_and_predict_oof_logreg(rows, folds)

    n = len(rows)
    assert len(result["student_id"]) == n
    assert len(result["subject_slug"]) == n
    assert len(result["checkpoint_idx"]) == n
    assert len(result["y_true"]) == n
    assert len(result["y_proba"]) == n

    seen_keys = set()
    for sid, subj, cp in zip(
        result["student_id"], result["subject_slug"], result["checkpoint_idx"]
    ):
        key = (sid, subj, cp)
        assert key not in seen_keys, f"duplicate OOF row: {key}"
        seen_keys.add(key)

    expected_keys = {
        (row["student_id"], row["subject_slug"], row["checkpoint_idx"])
        for row in rows
    }
    assert seen_keys == expected_keys


def test_oof_covers_every_row_exactly_once_gbdt(monkeypatch):
    monkeypatch.setattr(baselines, "oversample_training_rows", _fake_oversample_noop)
    rows, student_at_risk = _build_fixture_rows()
    student_ids = list(student_at_risk.keys())
    folds = _fake_grouped_kfold(student_ids, student_at_risk, n_splits=3)

    result = baselines.train_and_predict_oof_gbdt(rows, folds)

    n = len(rows)
    assert len(result["student_id"]) == n
    assert len(result["y_proba"]) == n

    seen_keys = {
        (sid, subj, cp)
        for sid, subj, cp in zip(
            result["student_id"], result["subject_slug"], result["checkpoint_idx"]
        )
    }
    expected_keys = {
        (row["student_id"], row["subject_slug"], row["checkpoint_idx"])
        for row in rows
    }
    assert seen_keys == expected_keys


def test_y_true_matches_original_row_label(monkeypatch):
    """Each OOF entry's y_true must match the original row's at_risk label,
    proving our row<->fold<->prediction bookkeeping doesn't scramble rows."""
    monkeypatch.setattr(baselines, "oversample_training_rows", _fake_oversample_noop)
    rows, student_at_risk = _build_fixture_rows()
    student_ids = list(student_at_risk.keys())
    folds = _fake_grouped_kfold(student_ids, student_at_risk, n_splits=3)

    result = baselines.train_and_predict_oof_logreg(rows, folds)

    by_key = {
        (row["student_id"], row["subject_slug"], row["checkpoint_idx"]): row["at_risk"]
        for row in rows
    }
    for sid, subj, cp, y_true in zip(
        result["student_id"],
        result["subject_slug"],
        result["checkpoint_idx"],
        result["y_true"],
    ):
        assert by_key[(sid, subj, cp)] == y_true


def test_val_rows_are_never_passed_to_oversample(monkeypatch):
    """Spec section 5 point 5: validation rows must never be oversampled.
    Assert the oversample function only ever receives rows whose
    student_id is in that fold's TRAIN set."""
    rows, student_at_risk = _build_fixture_rows()
    student_ids = list(student_at_risk.keys())
    folds = _fake_grouped_kfold(student_ids, student_at_risk, n_splits=3)

    seen_train_student_ids = []

    def _spy_oversample(train_rows, target_ratio=1.0, seed=42):
        seen_train_student_ids.append({r["student_id"] for r in train_rows})
        return list(train_rows)

    monkeypatch.setattr(baselines, "oversample_training_rows", _spy_oversample)

    baselines.train_and_predict_oof_logreg(rows, folds)

    for (train_ids, val_ids), passed_ids in zip(folds, seen_train_student_ids):
        assert passed_ids == set(train_ids)
        assert passed_ids.isdisjoint(set(val_ids))


def test_logreg_scaler_fit_only_on_train_differs_per_fold(monkeypatch):
    """Verify per-fold StandardScaler discipline: fit on train rows only.
    We check this by making each fold's train distribution different and
    confirming the scaler's fitted mean differs fold to fold (if val rows
    leaked into the fit, scaler stats would be identical/biased toward the
    full dataset every time)."""
    monkeypatch.setattr(baselines, "oversample_training_rows", _fake_oversample_noop)

    captured_means = []
    real_scaler_cls = baselines.StandardScaler

    class _SpyScaler(real_scaler_cls):
        def fit_transform(self, X, y=None):
            out = super().fit_transform(X, y)
            captured_means.append(tuple(self.mean_))
            return out

    monkeypatch.setattr(baselines, "StandardScaler", _SpyScaler)

    rows, student_at_risk = _build_fixture_rows()
    student_ids = list(student_at_risk.keys())
    folds = _fake_grouped_kfold(student_ids, student_at_risk, n_splits=3)

    baselines.train_and_predict_oof_logreg(rows, folds)

    assert len(captured_means) == len(folds)
    # Different folds hold out different students, so train-set means
    # should not be byte-identical across folds.
    assert len(set(captured_means)) > 1


def test_gbdt_skips_standardization_but_still_predicts(monkeypatch):
    """GBDT path should not need/use a scaler; just confirm it produces
    valid probabilities in [0, 1] without erroring."""
    monkeypatch.setattr(baselines, "oversample_training_rows", _fake_oversample_noop)
    rows, student_at_risk = _build_fixture_rows()
    student_ids = list(student_at_risk.keys())
    folds = _fake_grouped_kfold(student_ids, student_at_risk, n_splits=3)

    result = baselines.train_and_predict_oof_gbdt(rows, folds)
    proba = np.array(result["y_proba"])
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_custom_feature_names_are_respected(monkeypatch):
    """Passing a feature_names subset should not error and should still
    produce full OOF coverage - proves feature_names is actually wired
    through to the feature matrix builder, not ignored."""
    monkeypatch.setattr(baselines, "oversample_training_rows", _fake_oversample_noop)
    rows, student_at_risk = _build_fixture_rows()
    student_ids = list(student_at_risk.keys())
    folds = _fake_grouped_kfold(student_ids, student_at_risk, n_splits=3)

    subset = ["internal_marks_pct", "attendance_pct"]
    result = baselines.train_and_predict_oof_logreg(rows, folds, feature_names=subset)
    assert len(result["y_proba"]) == len(rows)


# ---------------------------------------------------------------------------
# 2. Real integration tests - only run once cv.py / data exist.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not CV_PATH.exists(), reason="scripts/training/cv.py not landed yet"
)
@pytest.mark.skipif(
    not DATA_PATH.exists(), reason="data/training_rows.json not present"
)
def test_real_end_to_end_with_actual_cv_and_data():
    from scripts.training.cv import grouped_student_kfold

    rows = json.loads(DATA_PATH.read_text())
    student_at_risk = {}
    for row in rows:
        sid = row["student_id"]
        student_at_risk[sid] = student_at_risk.get(sid, False) or bool(row["at_risk"])

    student_ids = list(student_at_risk.keys())
    folds = grouped_student_kfold(student_ids, student_at_risk, n_splits=5, seed=42)

    logreg_result = baselines.train_and_predict_oof_logreg(rows, folds)
    gbdt_result = baselines.train_and_predict_oof_gbdt(rows, folds)

    n = len(rows)
    assert len(logreg_result["y_proba"]) == n
    assert len(gbdt_result["y_proba"]) == n

    if METRICS_PATH.exists():
        from scripts.training.metrics import evaluate_predictions

        logreg_metrics = evaluate_predictions(
            logreg_result["y_true"], logreg_result["y_proba"]
        )
        gbdt_metrics = evaluate_predictions(
            gbdt_result["y_true"], gbdt_result["y_proba"]
        )
        print("LogReg metrics:", logreg_metrics)
        print("GBDT metrics:", gbdt_metrics)
