"""Lightweight NumPy/plain-Python inference paths for ScholarTrack's models.

Per `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` section 5's
architecture paragraph, and mirroring `Lecture2Revision/lstm_model.py`'s
`NumPyLSTMModel` pattern: train with a framework (scikit-learn / Keras),
export the learned weights, and reimplement the forward pass in plain
NumPy so production serving never needs TensorFlow / scikit-learn /
LightGBM runtimes installed.

`numpy_logreg.py` is the priority module -- Logistic Regression is the
model that actually ships (see `models/benchmark_report.md`). `numpy_gbdt.py`
and `numpy_cnn.py` exist to satisfy the spec's "all three model types need a
serving path" requirement but are documented, lower-priority, and not
served in production.
"""
