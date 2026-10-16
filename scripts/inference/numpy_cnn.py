"""Pure-NumPy inference path for ScholarTrack's Conv1D model (Step 5a) --
documented, NOT the production model. Per `models/benchmark_report.md`,
the CNN loses to Logistic Regression on the held-out test (precision
collapses); `numpy_logreg.py` is the served path. This module exists to
satisfy spec section 5's "all three model types need a NumPy serving
path" requirement.

Mirrors `Lecture2Revision/lstm_model.py`'s `NumPyLSTMModel` pattern (train
with Keras, export `model.get_weights()`, reimplement the forward pass by
hand) for `scripts/training/cnn_model.py::build_model`'s architecture:

    Conv1D(16, k=3, same) -> BatchNormalization -> ReLU -> MaxPooling1D(2, same)
    -> Conv1D(32, k=3, same) -> GlobalAveragePooling1D
    -> Dense(16, relu) -> Dropout(0.3) -> Dense(1, sigmoid)

`model.get_weights()` returns, in order:
    [0] conv1d kernel,  shape (3, 5, 16)
    [1] conv1d bias,    shape (16,)
    [2] batchnorm gamma (scale),        shape (16,)
    [3] batchnorm beta (offset),        shape (16,)
    [4] batchnorm moving_mean,          shape (16,)
    [5] batchnorm moving_variance,      shape (16,)
    [6] conv1d_1 kernel, shape (3, 16, 32)
    [7] conv1d_1 bias,   shape (32,)
    [8] dense kernel,    shape (32, 16)
    [9] dense bias,      shape (16,)
    [10] dense_1 kernel, shape (16, 1)
    [11] dense_1 bias,   shape (1,)

Dropout is a no-op at inference time (it's only active during
`model.fit`), so it's simply skipped in the forward pass below.

Numerical parity: unlike `numpy_logreg.py` (exact dot-product + sigmoid,
atol=1e-8), this forward pass involves a convolution implemented as
explicit sliding-window dot products, re-ordering the same floating-point
sums Keras/TensorFlow computes via its own (likely vectorized/fused, and
possibly oneDNN-accelerated) conv kernels. Different summation order for
the same mathematical sum is not bitwise-identical in floating point, so
`tests/inference/test_numpy_cnn.py` uses a looser `atol=1e-4` -- still
tight enough to catch a real implementation bug, loose enough to tolerate
legitimate floating-point reordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


class NumPyCNNModel:
    """Plain-NumPy forward pass for a trained Keras Conv1D ScholarTrack model."""

    def __init__(
        self,
        conv1_kernel: np.ndarray,
        conv1_bias: np.ndarray,
        bn_gamma: np.ndarray,
        bn_beta: np.ndarray,
        bn_mean: np.ndarray,
        bn_var: np.ndarray,
        conv2_kernel: np.ndarray,
        conv2_bias: np.ndarray,
        dense1_kernel: np.ndarray,
        dense1_bias: np.ndarray,
        dense2_kernel: np.ndarray,
        dense2_bias: np.ndarray,
        bn_epsilon: float = 1e-3,
    ) -> None:
        self.conv1_kernel = np.asarray(conv1_kernel, dtype=float)  # (3, 5, 16)
        self.conv1_bias = np.asarray(conv1_bias, dtype=float)  # (16,)
        self.bn_gamma = np.asarray(bn_gamma, dtype=float)
        self.bn_beta = np.asarray(bn_beta, dtype=float)
        self.bn_mean = np.asarray(bn_mean, dtype=float)
        self.bn_var = np.asarray(bn_var, dtype=float)
        self.conv2_kernel = np.asarray(conv2_kernel, dtype=float)  # (3, 16, 32)
        self.conv2_bias = np.asarray(conv2_bias, dtype=float)  # (32,)
        self.dense1_kernel = np.asarray(dense1_kernel, dtype=float)  # (32, 16)
        self.dense1_bias = np.asarray(dense1_bias, dtype=float)
        self.dense2_kernel = np.asarray(dense2_kernel, dtype=float)  # (16, 1)
        self.dense2_bias = np.asarray(dense2_bias, dtype=float)
        self.bn_epsilon = float(bn_epsilon)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_keras_model(cls, model, bn_epsilon: Optional[float] = None) -> "NumPyCNNModel":
        """Extracts weights via `model.get_weights()` (order documented in
        the module docstring). `bn_epsilon` defaults to the trained
        model's own BatchNormalization layer epsilon when available
        (falls back to Keras's default 1e-3 otherwise)."""
        weights = model.get_weights()
        if bn_epsilon is None:
            bn_epsilon = 1e-3
            for layer in getattr(model, "layers", []):
                if "batch_normalization" in layer.__class__.__name__.lower() or hasattr(layer, "epsilon"):
                    try:
                        bn_epsilon = float(layer.epsilon)
                        break
                    except (AttributeError, TypeError):
                        continue
        return cls(
            conv1_kernel=weights[0],
            conv1_bias=weights[1],
            bn_gamma=weights[2],
            bn_beta=weights[3],
            bn_mean=weights[4],
            bn_var=weights[5],
            conv2_kernel=weights[6],
            conv2_bias=weights[7],
            dense1_kernel=weights[8],
            dense1_bias=weights[9],
            dense2_kernel=weights[10],
            dense2_bias=weights[11],
            bn_epsilon=bn_epsilon,
        )

    # -- layer primitives -------------------------------------------------

    @staticmethod
    def _conv1d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """x: (batch, T, C_in); kernel: (k, C_in, C_out); bias: (C_out,).
        "same" padding, stride 1. Returns (batch, T, C_out)."""
        batch, T, c_in = x.shape
        k, k_c_in, c_out = kernel.shape
        assert k_c_in == c_in
        pad_total = k - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_padded = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")

        out = np.zeros((batch, T, c_out), dtype=float)
        for t in range(T):
            window = x_padded[:, t : t + k, :]  # (batch, k, c_in)
            # Sliding-window dot product: sum over k and c_in.
            out[:, t, :] = np.tensordot(window, kernel, axes=([1, 2], [0, 1])) + bias
        return out

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _batchnorm(self, x: np.ndarray) -> np.ndarray:
        """Inference-mode BatchNormalization affine transform, using the
        trained moving mean/variance (not batch statistics)."""
        return self.bn_gamma * (x - self.bn_mean) / np.sqrt(self.bn_var + self.bn_epsilon) + self.bn_beta

    @staticmethod
    def _maxpool1d_same(x: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
        """x: (batch, T, C). "same" padding (pads with -inf so padded
        positions never win the max)."""
        batch, T, C = x.shape
        out_len = int(np.ceil(T / stride))
        pad_total = max(0, (out_len - 1) * stride + pool_size - T)
        pad_right = pad_total
        if pad_right > 0:
            x = np.pad(x, ((0, 0), (0, pad_right), (0, 0)), mode="constant", constant_values=-np.inf)

        out = np.zeros((batch, out_len, C), dtype=float)
        for t in range(out_len):
            start = t * stride
            out[:, t, :] = np.max(x[:, start : start + pool_size, :], axis=1)
        return out

    @staticmethod
    def _global_avg_pool1d(x: np.ndarray) -> np.ndarray:
        """x: (batch, T, C) -> (batch, C)."""
        return np.mean(x, axis=1)

    @staticmethod
    def _dense(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return x @ kernel + bias

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    # -- inference ----------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (batch, 6, 5) sequence input (see `scripts/training/cnn_model.py
        ::build_sequence`). Returns P(at_risk=True), shape (batch,)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        x = self._conv1d_same(X, self.conv1_kernel, self.conv1_bias)
        x = self._batchnorm(x)
        x = self._relu(x)
        x = self._maxpool1d_same(x, pool_size=2, stride=2)
        x = self._conv1d_same(x, self.conv2_kernel, self.conv2_bias)
        x = self._global_avg_pool1d(x)
        x = self._relu(self._dense(x, self.dense1_kernel, self.dense1_bias))
        # Dropout skipped: it is a no-op at inference time.
        x = self._dense(x, self.dense2_kernel, self.dense2_bias)
        return self._sigmoid(x).reshape(-1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(X) >= float(threshold)

    # -- persistence ----------------------------------------------------

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            conv1_kernel=self.conv1_kernel,
            conv1_bias=self.conv1_bias,
            bn_gamma=self.bn_gamma,
            bn_beta=self.bn_beta,
            bn_mean=self.bn_mean,
            bn_var=self.bn_var,
            conv2_kernel=self.conv2_kernel,
            conv2_bias=self.conv2_bias,
            dense1_kernel=self.dense1_kernel,
            dense1_bias=self.dense1_bias,
            dense2_kernel=self.dense2_kernel,
            dense2_bias=self.dense2_bias,
            bn_epsilon=np.array([self.bn_epsilon]),
        )

    @classmethod
    def load(cls, path) -> "NumPyCNNModel":
        path = Path(path)
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        with np.load(path, allow_pickle=False) as data:
            return cls(
                conv1_kernel=data["conv1_kernel"],
                conv1_bias=data["conv1_bias"],
                bn_gamma=data["bn_gamma"],
                bn_beta=data["bn_beta"],
                bn_mean=data["bn_mean"],
                bn_var=data["bn_var"],
                conv2_kernel=data["conv2_kernel"],
                conv2_bias=data["conv2_bias"],
                dense1_kernel=data["dense1_kernel"],
                dense1_bias=data["dense1_bias"],
                dense2_kernel=data["dense2_kernel"],
                dense2_bias=data["dense2_bias"],
                bn_epsilon=float(data["bn_epsilon"][0]),
            )
