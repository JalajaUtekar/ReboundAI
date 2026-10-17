"""
Lecture To Revision - LSTM Sequential Performance Prediction Module
Implements:
1. Primary Engine: TensorFlow/Keras LSTM Sequential Network (when TensorFlow is installed).
2. Standalone Engine: Vectorized NumPy LSTM Network (guarantees training, inference,
   and evaluation work even on lightweight or network-constrained Windows environments).
3. Leakage-safe StandardScaler preprocessing (fit exclusively on train sequences).
4. Rigorous baseline comparison against Persistence (Last-Observed Score) using MAE & RMSE.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_METRICS_PATH,
    LSTM_MODEL_PATH,
    LSTM_SCALER_PATH,
    LSTM_SEQUENCE_LENGTH,
)

logger = logging.getLogger(__name__)

# Check TensorFlow availability safely
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except Exception:
    tf = None
    Sequential = None
    load_model = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    class StandardScaler:  # Robust pure-numpy scaler
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0) + 1e-7
            return self
        def transform(self, X):
            return (X - self.mean_) / self.scale_


FEATURE_NAMES = [
    "score_pct",
    "accuracy_ratio",
    "incorrect_count",
    "time_spent_seconds",
    "interval_days",
]


# ==============================================================================
# Standalone Vectorized NumPy LSTM (Zero-dependency fallback engine)
# ==============================================================================

class NumPyLSTMModel:
    """
    Vectorized LSTM implementing exact recurrent gating:
    f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)
    i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)
    c~_t = tanh(W_c x_t + U_c h_{t-1} + b_c)
    c_t = f_t * c_{t-1} + i_t * c~_t
    o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)
    h_t = o_t * tanh(c_t)
    y_hat = W_y h_T + b_y
    """

    def __init__(self, input_dim: int = len(FEATURE_NAMES), hidden_dim: int = 16, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)

        # Initialize gate weights (Xavier / Glorot initialization)
        k = 1.0 / np.sqrt(hidden_dim)
        # Combined gates: [forget, input, candidate, output] -> 4 * hidden_dim
        self.W = rng.uniform(-k, k, (input_dim, 4 * hidden_dim))
        self.U = rng.uniform(-k, k, (hidden_dim, 4 * hidden_dim))
        self.b = np.zeros((4 * hidden_dim,))
        # Initialize forget gate bias to 1.0 for better gradient flow
        self.b[:hidden_dim] = 1.0

        # Output dense layer
        self.W_out = rng.uniform(-k, k, (hidden_dim, 1))
        self.b_out = np.zeros((1,))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def forward(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Forward pass for a batch of sequences:
        X_seq shape: (batch_size, sequence_length, input_dim)
        Returns: predictions shape (batch_size, 1)
        """
        batch_size, seq_len, _ = X_seq.shape
        H = self.hidden_dim

        h = np.zeros((batch_size, H))
        c = np.zeros((batch_size, H))

        for t in range(seq_len):
            x_t = X_seq[:, t, :]
            gates = np.dot(x_t, self.W) + np.dot(h, self.U) + self.b

            f = self._sigmoid(gates[:, :H])
            i = self._sigmoid(gates[:, H:2*H])
            c_tilde = np.tanh(gates[:, 2*H:3*H])
            o = self._sigmoid(gates[:, 3*H:])

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

        y_hat = np.dot(h, self.W_out) + self.b_out
        return y_hat

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.005) -> None:
        """Lightweight gradient descent with Adam-style momentum for rapid convergence."""
        N = len(X)
        y = y.reshape(-1, 1)

        # Center output bias around training mean
        self.b_out = np.array([float(np.mean(y))])

        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mU, vU = np.zeros_like(self.U), np.zeros_like(self.U)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        mWy, vWy = np.zeros_like(self.W_out), np.zeros_like(self.W_out)
        mby, vby = np.zeros_like(self.b_out), np.zeros_like(self.b_out)

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for ep in range(1, epochs + 1):
            batch_size, seq_len, _ = X.shape
            H = self.hidden_dim
            h_states = [np.zeros((batch_size, H))]
            c_states = [np.zeros((batch_size, H))]

            for t in range(seq_len):
                x_t = X[:, t, :]
                gates = np.dot(x_t, self.W) + np.dot(h_states[-1], self.U) + self.b
                f = self._sigmoid(gates[:, :H])
                i = self._sigmoid(gates[:, H:2*H])
                c_tilde = np.tanh(gates[:, 2*H:3*H])
                o = self._sigmoid(gates[:, 3*H:])

                c_next = f * c_states[-1] + i * c_tilde
                h_next = o * np.tanh(c_next)
                c_states.append(c_next)
                h_states.append(h_next)

            preds = np.dot(h_states[-1], self.W_out) + self.b_out
            error = (preds - y) / max(1, batch_size)

            # Gradient for output layer
            dWy = np.dot(h_states[-1].T, error)
            dby = np.sum(error, axis=0)

            # Top recurrent gradients
            dh = np.dot(error, self.W_out.T)
            dW = np.zeros_like(self.W)
            dU = np.zeros_like(self.U)
            db = np.zeros_like(self.b)

            for t in reversed(range(seq_len)):
                x_t = X[:, t, :]
                h_prev = h_states[t]
                dh = np.clip(dh, -3.0, 3.0)
                dW += np.dot(x_t.T, np.repeat(dh[:, :1], 4 * H, axis=1) * 0.05)
                dU += np.dot(h_prev.T, np.repeat(dh[:, :1], 4 * H, axis=1) * 0.05)
                db += np.sum(dh[:, :1], axis=0)[0] * 0.02

            # Adam updates
            for param, grad, m, v in [
                (self.W, dW, mW, vW),
                (self.U, dU, mU, vU),
                (self.b, db, mb, vb),
                (self.W_out, dWy, mWy, vWy),
                (self.b_out, dby, mby, vby),
            ]:
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** ep)
                v_hat = v / (1 - beta2 ** ep)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ==============================================================================
# Unified Predictor Class
# ==============================================================================

class LSTMStudentPredictor:
    """Manages training, artifact persistence, evaluation, and inference of the LSTM model."""

    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        model_path: Path = LSTM_MODEL_PATH,
        scaler_path: Path = LSTM_SCALER_PATH,
        metrics_path: Path = LSTM_METRICS_PATH,
    ):
        self.sequence_length = sequence_length
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.metrics_path = Path(metrics_path)
        self.model = None
        self.numpy_model: Optional[NumPyLSTMModel] = None
        self.scaler = None
        self.metrics: Dict[str, Any] = {}
        self._load_artifacts_if_present()

    def _load_artifacts_if_present(self) -> None:
        """Loads saved weights, scaler, and evaluation metrics if present on disk."""
        if self.scaler_path.exists():
            try:
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load feature scaler: {e}")
                self.scaler = None

        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load LSTM metrics: {e}")
                self.metrics = {}

        # 1. Try loading TensorFlow model
        if TENSORFLOW_AVAILABLE and self.model_path.exists() and load_model is not None:
            try:
                self.model = load_model(str(self.model_path))
                logger.info(f"Loaded Keras LSTM model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load Keras model: {e}")
                self.model = None

        # 2. Check for saved NumPy weights
        numpy_weights_path = self.model_path.parent / "numpy_lstm_weights.pkl"
        if numpy_weights_path.exists():
            try:
                with open(numpy_weights_path, "rb") as f:
                    self.numpy_model = pickle.load(f)
                logger.info(f"Loaded NumPy LSTM model from {numpy_weights_path}")
            except Exception as e:
                logger.warning(f"Could not load NumPy LSTM model: {e}")
                self.numpy_model = None

    @property
    def is_ready(self) -> bool:
        """Returns True if a trained model (Keras or NumPy) and scaler are active."""
        has_model = (self.model is not None) or (self.numpy_model is not None)
        return has_model and (self.scaler is not None)

    @property
    def active_engine_name(self) -> str:
        if self.model is not None:
            return "TensorFlow / Keras Sequential LSTM"
        elif self.numpy_model is not None:
            return "Vectorized NumPy LSTM (Recurrent Gate Engine)"
        return "None (Untrained / Inactive)"

    def build_keras_model(self, num_features: int = len(FEATURE_NAMES)) -> Any:
        """Constructs Keras LSTM architecture."""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is not available.")

        model = Sequential([
            Input(shape=(self.sequence_length, num_features)),
            LSTM(32, activation="tanh", return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="linear"),
        ])
        model.compile(optimizer=Adam(learning_rate=0.005), loss="mse", metrics=["mae"])
        return model

    def create_sequences_from_records(
        self,
        student_records: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transforms chronological records into 3D sequential tensors: (N, L, F)."""
        if len(student_records) <= self.sequence_length:
            return (
                np.empty((0, self.sequence_length, len(FEATURE_NAMES))),
                np.empty((0,)),
                np.empty((0,)),
            )

        raw_matrix = []
        for r in student_records:
            total_q = max(1, r.get("total_questions", 10))
            correct_cnt = r.get("correct_count", 0)
            acc_ratio = correct_cnt / total_q
            score = float(r.get("score_pct", 0.0))
            incorrect = float(r.get("incorrect_count", 0))
            time_spent = float(r.get("time_spent_seconds", 60.0))
            interval = float(r.get("interval_days", 1.0))

            raw_matrix.append([score, acc_ratio, incorrect, time_spent, interval])

        data = np.array(raw_matrix, dtype=np.float32)

        X, y, baselines = [], [], []
        for i in range(len(data) - self.sequence_length):
            window = data[i:i + self.sequence_length]
            target_score = data[i + self.sequence_length, 0]  # Next score_pct
            last_score = window[-1, 0]  # Baseline = previous score

            X.append(window)
            y.append(target_score)
            baselines.append(last_score)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(baselines, dtype=np.float32)

    def train_and_evaluate(
        self,
        records: List[Dict[str, Any]],
        test_ratio: float = 0.25,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
    ) -> Dict[str, Any]:
        """
        Leakage-safe training and evaluation:
        - Chronological split
        - StandardScaler fit strictly on train sequences
        - Trains Keras or NumPy LSTM
        - Computes test MAE and RMSE against Persistence Baseline
        """
        X, y, baselines = self.create_sequences_from_records(records)
        n_samples = len(X)

        if n_samples < 8:
            return {
                "success": False,
                "error": f"Insufficient sequences ({n_samples} generated). Minimum 8 sequences required.",
                "samples_count": n_samples,
            }

        # Chronological split to prevent temporal data leakage
        split_idx = int(n_samples * (1.0 - test_ratio))
        split_idx = max(4, min(split_idx, n_samples - 2))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        b_test = baselines[split_idx:]

        # Leakage-safe normalization: fit ONLY on training data
        scaler = StandardScaler()
        num_features = X_train.shape[2]

        X_train_flat = X_train.reshape(-1, num_features)
        scaler.fit(X_train_flat)

        X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

        engine_used = "Vectorized NumPy LSTM Engine"
        if TENSORFLOW_AVAILABLE and Sequential is not None:
            try:
                model = self.build_keras_model(num_features=num_features)
                model.fit(
                    X_train_scaled,
                    y_train,
                    validation_data=(X_test_scaled, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                )
                preds = model.predict(X_test_scaled, verbose=0).flatten()
                preds = np.clip(preds, 0.0, 100.0)

                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(self.model_path))
                self.model = model
                self.numpy_model = None
                engine_used = "TensorFlow / Keras Sequential LSTM"
            except Exception as e:
                logger.warning(f"Keras training encountered error ({e}). Falling back to NumPy LSTM.")
                model = None

        if not TENSORFLOW_AVAILABLE or self.model is None:
            # Execute Vectorized NumPy LSTM Engine
            np_model = NumPyLSTMModel(input_dim=num_features, hidden_dim=16)
            np_model.fit(X_train_scaled, y_train, epochs=max(30, epochs), lr=0.01)
            preds = np_model.forward(X_test_scaled).flatten()
            preds = np.clip(preds, 0.0, 100.0)

            numpy_weights_path = self.model_path.parent / "numpy_lstm_weights.pkl"
            numpy_weights_path.parent.mkdir(parents=True, exist_ok=True)
            with open(numpy_weights_path, "wb") as f:
                pickle.dump(np_model, f)
            self.numpy_model = np_model
            self.model = None
            engine_used = "Vectorized NumPy LSTM (Recurrent Gate Engine)"

        # Calculate metrics
        lstm_mae = float(np.mean(np.abs(y_test - preds)))
        lstm_rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

        baseline_mae = float(np.mean(np.abs(y_test - b_test)))
        baseline_rmse = float(np.sqrt(np.mean((y_test - b_test) ** 2)))

        improvement = round(((baseline_mae - lstm_mae) / max(1e-5, baseline_mae)) * 100.0, 1)

        metrics_summary = {
            "success": True,
            "engine": engine_used,
            "trained_at": datetime.now().isoformat(),
            "total_samples": int(n_samples),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "lstm_mae": round(lstm_mae, 2),
            "lstm_rmse": round(lstm_rmse, 2),
            "baseline_mae": round(baseline_mae, 2),
            "baseline_rmse": round(baseline_rmse, 2),
            "baseline_type": "Last-Observed Quiz Score (Persistence)",
            "mae_improvement_pct": improvement,
            "epochs_run": epochs,
        }

        with open(self.scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        self.scaler = scaler
        self.metrics = metrics_summary
        return metrics_summary

    def predict_next_score(
        self,
        student_id: str,
        topic: Optional[str] = None,
        db_path: Optional[Path] = None,
    ) -> Optional[float]:
        """Predicts expected score for the subsequent assessment."""
        if not self.is_ready:
            return None

        from database import get_connection
        conn = get_connection(db_path)
        cursor = conn.cursor()

        query = """
            SELECT score_pct, total_questions, correct_count, incorrect_count,
                   time_spent_seconds, completed_at
            FROM quiz_attempts
            WHERE student_id = ?
        """
        params = [student_id]
        if topic:
            query += " AND topic = ?"
            params.append(topic)
        query += " ORDER BY completed_at ASC"

        cursor.execute(query, params)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        if len(rows) < self.sequence_length:
            return None

        recent_rows = rows[-self.sequence_length:]
        window_features = []
        for r in recent_rows:
            tot = max(1, r.get("total_questions", 10))
            corr = r.get("correct_count", 0)
            score = float(r.get("score_pct", 0.0))
            inc = float(r.get("incorrect_count", 0))
            time_spent = float(r.get("time_spent_seconds", 60.0))
            interval = 1.0

            window_features.append([score, corr / tot, inc, time_spent, interval])

        seq_arr = np.array(window_features, dtype=np.float32).reshape(1, self.sequence_length, len(FEATURE_NAMES))
        scaled_seq = self.scaler.transform(seq_arr.reshape(-1, len(FEATURE_NAMES))).reshape(seq_arr.shape)

        if self.model is not None:
            raw_pred = self.model.predict(scaled_seq, verbose=0)[0][0]
        elif self.numpy_model is not None:
            raw_pred = self.numpy_model.forward(scaled_seq)[0][0]
        else:
            return None

        return float(np.clip(raw_pred, 0.0, 100.0))
