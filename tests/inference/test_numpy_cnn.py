"""Tests for scripts/inference/numpy_cnn.py.

NOT the served model (Logistic Regression is -- see
models/benchmark_report.md); this is the documented, lower-priority CNN
NumPy serving path for the Conv1D architecture in
`scripts/training/cnn_model.py::build_model`.

Trains a real (cheap -- few epochs, no CV/tuning) CNN on real sequence
data built from `data/synthetic_students.json` via
`scripts/training/cnn_model.py::build_sequences`, to get a real set of
trained weights (not random-init weights) to check parity against.

Tolerance: `atol=1e-4`, looser than the LogReg path's `atol=1e-8`. This
forward pass re-implements Conv1D as explicit sliding-window dot
products, which sums the same numbers in different order than
TensorFlow/Keras's own (vectorized, oneDNN-accelerated) convolution
kernel. Floating-point addition is not associative, so a different
summation order over ~6 convolution windows x 2 conv layers x batch-norm
division can accumulate rounding differences on the order of 1e-5 to
1e-6 even though the computation is mathematically identical -- hence
the looser (but still strict) tolerance. See `numpy_cnn.py`'s module
docstring for the same note.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    TENSORFLOW_AVAILABLE = False

from scripts.inference.numpy_cnn import NumPyCNNModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "synthetic_students.json"

pytestmark = pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")


@pytest.fixture(scope="module")
def trained_cnn():
    if not DATA_PATH.exists():
        pytest.skip(f"{DATA_PATH} not present")
    import json

    from scripts.training.cnn_model import build_model, build_sequences

    dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    built = build_sequences(dataset)
    X, y = built["X"], built["y"]

    tf.keras.utils.set_random_seed(0)
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Cheap, real training run -- just enough epochs to get real
    # (non-trivial) trained weights, no CV/tuning per the task's scope.
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model, X


class TestParity:
    def test_parity_on_real_trained_weights(self, trained_cnn):
        model, X = trained_cnn
        numpy_model = NumPyCNNModel.from_keras_model(model)

        numpy_proba = numpy_model.predict_proba(X)
        keras_proba = model.predict(X, verbose=0).reshape(-1)

        max_abs_diff = float(np.max(np.abs(numpy_proba - keras_proba)))
        print(f"\n[cnn parity] max abs diff: {max_abs_diff:.3e}")

        assert np.allclose(numpy_proba, keras_proba, atol=1e-4)

    def test_parity_on_subset_batch(self, trained_cnn):
        """Parity should hold for a smaller batch too (not just the full
        dataset), to catch any batch-size-dependent bug."""
        model, X = trained_cnn
        numpy_model = NumPyCNNModel.from_keras_model(model)

        subset = X[:16]
        numpy_proba = numpy_model.predict_proba(subset)
        keras_proba = model.predict(subset, verbose=0).reshape(-1)
        assert np.allclose(numpy_proba, keras_proba, atol=1e-4)

    def test_single_sequence_input(self, trained_cnn):
        model, X = trained_cnn
        numpy_model = NumPyCNNModel.from_keras_model(model)
        single = X[0]  # shape (6, 5), no batch dim
        proba = numpy_model.predict_proba(single)
        assert proba.shape == (1,)

        keras_proba = model.predict(X[0:1], verbose=0).reshape(-1)
        assert np.allclose(proba, keras_proba, atol=1e-4)


class TestSaveLoadRoundTrip:
    def test_round_trip(self, trained_cnn, tmp_path):
        model, X = trained_cnn
        numpy_model = NumPyCNNModel.from_keras_model(model)
        path = tmp_path / "cnn.npz"
        numpy_model.save(path)
        loaded = NumPyCNNModel.load(path)

        assert np.allclose(loaded.predict_proba(X), numpy_model.predict_proba(X), atol=1e-12)


class TestSanity:
    def test_shape_and_probability_range(self, trained_cnn):
        model, X = trained_cnn
        numpy_model = NumPyCNNModel.from_keras_model(model)
        proba = numpy_model.predict_proba(X)
        assert proba.shape == (X.shape[0],)
        assert np.all((proba >= 0.0) & (proba <= 1.0))
