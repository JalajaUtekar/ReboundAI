"""CNN sequence model for Rebound AI's ScholarTrack risk prediction (Step 3).

Unlike the flat-feature LogReg/GBDT baselines (which train on
`data/training_rows.json`, one flattened scalar vector per as-of checkpoint),
this module builds a genuine per-checkpoint SEQUENCE directly from
`data/synthetic_students.json`, per spec section 5's architecture paragraph:
a Conv1D stack over "the checkpoints available so far -- not the whole
completed term."

Sequence shape and feature composition
---------------------------------------
For each student-subject and each valid as-of prefix (`checkpoint_idx` 2..7,
matching Step 2's convention -- checkpoint 1 alone is too little signal), we
build a fixed-size `(6, 5)` array:

  - 6 timesteps: one row per checkpoint in `CHECKPOINTS_IN_TIME_ORDER`
    (max 7 checkpoints exist, but a valid prefix with `checkpoint_idx` up to
    7 only ever has up to 6 *preceding-or-current* rows worth populating
    relative to a 6-row buffer -- see note below). Rows
    `0 .. checkpoint_idx - 1` hold REAL per-checkpoint values in chronological
    order; remaining rows (up to index 5) are zero-padded. (For
    checkpoint_idx=7 all 6 timesteps... see IMPORTANT note below.)
  - 5 features per timestep, each computed using ONLY that checkpoint's own
    event and earlier ones -- never later checkpoints, never ETE:
      0. marks_pct_at_this_checkpoint: this checkpoint's own score as a pct
         of its own max_marks (own-event pct, NOT a cumulative aggregate --
         this is what makes each row a genuinely distinct timestep rather
         than a repeated running total).
      1. cumulative_attendance_pct_as_of_this_checkpoint: mean attendance_pct
         over every attendance week up to and including this checkpoint's
         calendar week.
      2. on_time_submission_ratio_so_far: fraction of submissions up to and
         including this checkpoint (0 if this checkpoint or all so-far
         checkpoints are unit tests and no submission has occurred yet) that
         were submitted on or before their due day.
      3. avg_days_late_so_far: mean of max(0, submitted_day - due_day) over
         submissions up to and including this checkpoint.
      4. is_unit_test_flag: 1.0 if this checkpoint is a unit test, 0.0 if a
         submission.

IMPORTANT -- the 6-row buffer vs. 7 checkpoints: `CHECKPOINTS_IN_TIME_ORDER`
has 7 entries, but the spec fixes the sequence length at 6 timesteps. We
therefore cap the buffer at 6 rows and, for `checkpoint_idx=7` (the only case
where 7 real checkpoints would need to fit in 6 slots), keep the 6 MOST
RECENT checkpoints (i.e. drop the oldest one, checkpoint index 0) rather than
truncating the most recent (most informative) one. For checkpoint_idx <= 6
this never triggers and all real checkpoints are kept, zero-padded at the
tail. This is documented here because it is the one place the "rows
0..checkpoint_idx-1 are real" description needs qualification.

No-leakage discipline mirrors `scripts/feature_engineering/extraction.py`:
every feature at row i is computed strictly from checkpoints
`CHECKPOINTS_IN_TIME_ORDER[:i+1]` (bounded by index, not by scanning the
whole record), and `subject["outcome"]` is read only for the label, never as
a feature input. A sequence built for a given `checkpoint_idx` is identical
whether or not the source record contains data for later checkpoints -- see
`tests/training/test_cnn_model.py::test_no_leakage_truncated_record_matches`.

Expected row count: 120 students x 6 subjects x 6 valid prefixes
(checkpoint_idx 2..7) = 4,320 -- the same as Step 2's flat extraction, since
both iterate the identical (student, subject, prefix) space; only the
per-row payload shape differs (a (6,5) sequence here vs. a flat 10-feature
vector there).
"""

from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from scripts.data_generation.generator import CHECKPOINTS_IN_TIME_ORDER

MIN_PREFIX_LEN = 2
MAX_PREFIX_LEN = len(CHECKPOINTS_IN_TIME_ORDER)  # 7
SEQ_LEN = 6
NUM_FEATURES = 5


def _checkpoint_week(subject: Dict, kind: str, idx: int) -> int:
    """The calendar week a (kind, idx) checkpoint falls in, read from the
    subject's own records (not module-level constants) so this stays
    correct on a truncated/mid-term copy of the record."""
    if kind == "unit_test":
        return subject["unit_tests"][idx]["week"]
    return subject["submissions"][idx]["due_day"] // 7


def _own_pct(subject: Dict, kind: str, idx: int) -> float:
    """A checkpoint's own score as a percentage of its own max marks (not cumulative)."""
    record = subject["unit_tests"][idx] if kind == "unit_test" else subject["submissions"][idx]
    if record["max_marks"] <= 0:
        return 0.0
    return record["score"] / record["max_marks"] * 100.0


def _cumulative_attendance_pct(subject: Dict, through_week: int) -> float:
    weeks = [w for w in subject["attendance_weeks"] if w["week"] <= through_week]
    if not weeks:
        return 0.0
    return sum(w["attendance_pct"] for w in weeks) / len(weeks)


def _checkpoint_row(subject: Dict, prefix_checkpoints: List[Tuple[str, int]]) -> List[float]:
    """Builds the 5-feature row for the LAST checkpoint in `prefix_checkpoints`,
    using only that checkpoint and the ones before it in the list (all of which
    are earlier in time, by construction of `CHECKPOINTS_IN_TIME_ORDER`)."""
    kind, idx = prefix_checkpoints[-1]

    marks_pct_at_this_checkpoint = _own_pct(subject, kind, idx)

    week = _checkpoint_week(subject, kind, idx)
    cumulative_attendance_pct = _cumulative_attendance_pct(subject, week)

    submissions_so_far = [(k, i) for (k, i) in prefix_checkpoints if k == "submission"]
    if submissions_so_far:
        sub_records = [subject["submissions"][i] for (_, i) in submissions_so_far]
        on_time_submission_ratio = sum(
            1 for s in sub_records if s["submitted_day"] <= s["due_day"]
        ) / len(sub_records)
        avg_days_late = sum(
            max(0, s["submitted_day"] - s["due_day"]) for s in sub_records
        ) / len(sub_records)
    else:
        on_time_submission_ratio = 0.0
        avg_days_late = 0.0

    is_unit_test_flag = 1.0 if kind == "unit_test" else 0.0

    return [
        marks_pct_at_this_checkpoint,
        cumulative_attendance_pct,
        on_time_submission_ratio,
        avg_days_late,
        is_unit_test_flag,
    ]


def build_sequence(subject: Dict, checkpoint_idx: int) -> np.ndarray:
    """Builds the `(SEQ_LEN, NUM_FEATURES)` = `(6, 5)` zero-padded sequence for one
    student-subject as of `checkpoint_idx` (1-indexed count of checkpoints elapsed,
    2..7). Only reads `subject["unit_tests"]`, `subject["submissions"]`, and
    `subject["attendance_weeks"]` entries within the prefix -- never `outcome`.
    """
    if not (MIN_PREFIX_LEN <= checkpoint_idx <= MAX_PREFIX_LEN):
        raise ValueError(f"checkpoint_idx must be between {MIN_PREFIX_LEN} and {MAX_PREFIX_LEN}, got {checkpoint_idx}")

    all_prefix_checkpoints = CHECKPOINTS_IN_TIME_ORDER[:checkpoint_idx]
    # Cap to the SEQ_LEN most recent checkpoints (see module docstring note).
    kept_checkpoints = all_prefix_checkpoints[-SEQ_LEN:]

    seq = np.zeros((SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    for row_idx, _ in enumerate(kept_checkpoints):
        prefix_so_far = kept_checkpoints[: row_idx + 1]
        seq[row_idx, :] = _checkpoint_row(subject, prefix_so_far)
    return seq


def build_sequences(dataset: List[Dict]) -> Dict[str, np.ndarray]:
    """Builds the full sequence dataset for a generated cohort (see module docstring).

    Returns a dict with:
      - "X": float32 array, shape (N, 6, 5)
      - "y": float32 array, shape (N,) -- subject["outcome"]["at_risk"] as 0./1.
      - "student_id": list[str], length N
      - "subject_slug": list[str], length N
      - "checkpoint_idx": list[int], length N
    """
    X, y, student_ids, subject_slugs, checkpoint_idxs = [], [], [], [], []
    for student in dataset:
        student_id = student["student_id"]
        for subject in student["subjects"]:
            outcome = subject["outcome"]
            for checkpoint_idx in range(MIN_PREFIX_LEN, MAX_PREFIX_LEN + 1):
                X.append(build_sequence(subject, checkpoint_idx))
                y.append(1.0 if outcome["at_risk"] else 0.0)
                student_ids.append(student_id)
                subject_slugs.append(subject["subject_slug"])
                checkpoint_idxs.append(checkpoint_idx)

    return {
        "X": np.stack(X, axis=0).astype(np.float32) if X else np.zeros((0, SEQ_LEN, NUM_FEATURES), dtype=np.float32),
        "y": np.array(y, dtype=np.float32),
        "student_id": student_ids,
        "subject_slug": subject_slugs,
        "checkpoint_idx": checkpoint_idxs,
    }


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


def build_model(input_shape: Tuple[int, int] = (SEQ_LEN, NUM_FEATURES)) -> keras.Model:
    """Builds the course-required CNN architecture (CE24MDM305 CNN unit, per spec):

        Conv1D(16) -> BatchNormalization -> ReLU -> MaxPooling1D
        -> Conv1D(32) -> GlobalAveragePooling1D
        -> Dense(16) -> Dropout(0.3) -> Dense(1, sigmoid)
    """
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(16, kernel_size=3, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = keras.layers.Conv1D(32, kernel_size=3, padding="same")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------


def binary_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Binary focal loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = y_pred if y_true == 1 else (1 - y_pred). Returns a Keras-compatible
    loss function `loss_fn(y_true, y_pred) -> scalar` (mean over the batch).
    """

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        focal = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal)

    return loss_fn


# ---------------------------------------------------------------------------
# Group-aware oversampling (sequence-shaped, independent of baselines.py)
# ---------------------------------------------------------------------------


def _oversample_sequences(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, jitter_std: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Basic oversampling for the minority (at_risk=1) class: duplicates each
    at-risk sequence, adding small Gaussian jitter (relative to each feature's
    own scale) only to the nonzero (real, non-padded) entries, until the
    positive class count matches the negative class count. Kept deliberately
    simple -- this is a per-fold, sequence-shaped analogue of the row-shaped
    `oversample_training_rows` used for the flat-feature baselines, which
    can't be reused here because of the different (N, 6, 5) data shape.
    """
    pos_mask = y == 1.0
    neg_mask = y == 0.0
    n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
    if n_pos == 0 or n_pos >= n_neg:
        return X, y

    pos_X = X[pos_mask]
    pos_y = y[pos_mask]
    deficit = n_neg - n_pos

    extra_idx = rng.integers(0, n_pos, size=deficit)
    extra_X = pos_X[extra_idx].copy()
    nonzero_mask = extra_X != 0.0
    noise = rng.normal(loc=0.0, scale=jitter_std, size=extra_X.shape) * np.abs(extra_X)
    extra_X[nonzero_mask] += noise[nonzero_mask]
    extra_y = pos_y[extra_idx]

    X_out = np.concatenate([X, extra_X], axis=0)
    y_out = np.concatenate([y, extra_y], axis=0)
    return X_out, y_out


# ---------------------------------------------------------------------------
# Training / OOF prediction
# ---------------------------------------------------------------------------


def train_and_predict_oof(
    dataset: List[Dict],
    folds: List[Tuple[List[str], List[str]]],
    loss: str = "weighted_bce",
    epochs: int = 30,
    batch_size: int = 16,
    seed: int = 0,
) -> Dict:
    """Trains a fresh CNN per fold and collects out-of-fold predictions.

    `folds` is `list[(train_student_ids, val_student_ids)]`, matching the shape
    `cv.grouped_student_kfold` produces (accepted structurally here, not imported).

    For each fold: builds sequences once, filters rows to that fold's train/val
    student IDs, applies group-aware oversampling to the TRAIN split only, builds
    and compiles a brand-new model (no weight reuse across folds), trains, and
    predicts probabilities on the val split. `loss="weighted_bce"` uses standard
    binary_crossentropy plus `class_weight` in `.fit()`; `loss="focal"` uses
    `binary_focal_loss()` with no `class_weight` (alpha already handles imbalance).

    Returns {"student_id", "subject_slug", "checkpoint_idx", "y_true", "y_proba"},
    each a list/array aligned across folds' concatenated val predictions.
    """
    if loss not in ("weighted_bce", "focal"):
        raise ValueError(f"loss must be 'weighted_bce' or 'focal', got {loss!r}")

    built = build_sequences(dataset)
    X_all, y_all = built["X"], built["y"]
    student_id_all = np.array(built["student_id"])
    subject_slug_all = np.array(built["subject_slug"])
    checkpoint_idx_all = np.array(built["checkpoint_idx"])

    out_student_id, out_subject_slug, out_checkpoint_idx = [], [], []
    out_y_true, out_y_proba = [], []

    for fold_i, (train_ids, val_ids) in enumerate(folds):
        train_mask = np.isin(student_id_all, train_ids)
        val_mask = np.isin(student_id_all, val_ids)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]

        rng = np.random.default_rng(seed + fold_i)
        X_train, y_train = _oversample_sequences(X_train, y_train, rng)

        tf.keras.utils.set_random_seed(seed + fold_i)
        model = build_model()

        fit_kwargs = {}
        if loss == "focal":
            compiled_loss = binary_focal_loss(alpha=0.25, gamma=2.0)
        else:
            compiled_loss = "binary_crossentropy"
            n_pos = float(y_train.sum())
            n_neg = float(len(y_train) - n_pos)
            if n_pos > 0 and n_neg > 0:
                total = n_pos + n_neg
                fit_kwargs["class_weight"] = {
                    0: total / (2.0 * n_neg),
                    1: total / (2.0 * n_pos),
                }

        model.compile(optimizer="adam", loss=compiled_loss, metrics=["accuracy"])

        if len(X_train) > 0:
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                **fit_kwargs,
            )

        if len(X_val) > 0:
            y_proba = model.predict(X_val, verbose=0).reshape(-1)
        else:
            y_proba = np.zeros((0,), dtype=np.float32)

        out_student_id.extend(student_id_all[val_mask].tolist())
        out_subject_slug.extend(subject_slug_all[val_mask].tolist())
        out_checkpoint_idx.extend(checkpoint_idx_all[val_mask].tolist())
        out_y_true.extend(y_val.tolist())
        out_y_proba.extend(y_proba.tolist())

    return {
        "student_id": out_student_id,
        "subject_slug": out_subject_slug,
        "checkpoint_idx": out_checkpoint_idx,
        "y_true": np.array(out_y_true, dtype=np.float32),
        "y_proba": np.array(out_y_proba, dtype=np.float32),
    }
