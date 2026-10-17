"""
Lecture To Revision - LSTM Training and Evaluation Script
Executes the sequential student performance pipeline:
1. Ingests chronological quiz attempts (from database or synthetic demo dataset).
2. Builds sliding temporal sequences.
3. Splits chronologically into training and test partitions (leakage-safe).
4. Fits StandardScaler strictly on training samples.
5. Trains TensorFlow/Keras LSTM architecture.
6. Evaluates MAE and RMSE against Last-Value persistence baseline.
7. Saves model weights and metrics artifacts.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, DB_PATH, LSTM_EPOCHS, LSTM_BATCH_SIZE
from database import get_connection
from lstm_model import LSTMStudentPredictor, TENSORFLOW_AVAILABLE


def load_training_data(use_synthetic: bool = False) -> list:
    """Loads records from SQLite or the labeled synthetic dataset."""
    if use_synthetic:
        synthetic_path = DATA_DIR / "synthetic_lstm_dataset.json"
        if not synthetic_path.exists():
            print(f"[!] Synthetic dataset not found at {synthetic_path}. Run generate_demo_data.py first.")
            return []
        with open(synthetic_path, "r", encoding="utf-8") as f:
            records = json.load(f)
            print(f"[*] Loaded {len(records)} records from labeled synthetic dataset: {synthetic_path.name}")
            return records

    # Query real/demo database records
    conn = get_connection(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT student_id, topic, score_pct, total_questions, correct_count,
               incorrect_count, time_spent_seconds, completed_at
        FROM quiz_attempts
        ORDER BY completed_at ASC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    if len(rows) < 8:
        print(f"[*] Notice: Database contains only {len(rows)} attempts (minimum 8 needed for training).")
        synthetic_path = DATA_DIR / "synthetic_lstm_dataset.json"
        if synthetic_path.exists():
            print(f"[*] Automatically falling back to labeled synthetic dataset ({synthetic_path.name}) for demonstration.")
            with open(synthetic_path, "r", encoding="utf-8") as f:
                return json.load(f)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM student performance predictor.")
    parser.add_argument("--synthetic", action="store_true", help="Force training on synthetic demo dataset.")
    parser.add_argument("--epochs", type=int, default=LSTM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=LSTM_BATCH_SIZE, help="Training batch size.")
    args = parser.parse_args()

    print("================================================================")
    print(" LECTURE TO REVISION - LSTM PERFORMANCE PREDICTION PIPELINE")
    print("================================================================")

    if not TENSORFLOW_AVAILABLE:
        print("[*] Notice: TensorFlow/Keras not installed in current environment.")
        print("[*] Activating Vectorized NumPy LSTM Recurrent Gate Engine (Zero-dependency fallback).")
    else:
        print("[*] TensorFlow/Keras detected: Utilizing Keras Sequential LSTM Engine.")

    records = load_training_data(use_synthetic=args.synthetic)
    if not records:
        print("[ERROR] No records available to train LSTM. Run 'python scripts/generate_demo_data.py'.")
        sys.exit(1)

    predictor = LSTMStudentPredictor()
    print(f"[*] Initializing model with sequence length L = {predictor.sequence_length}")
    print(f"[*] Training for {args.epochs} epochs with batch size {args.batch_size}...")

    results = predictor.train_and_evaluate(
        records=records,
        test_ratio=0.25,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if not results.get("success"):
        print(f"[FAIL] Training halted: {results.get('error')}")
        sys.exit(1)

    print("\n------------------- EVALUATION RESULTS -------------------")
    print(f"Total Sequences Generated : {results['total_samples']}")
    print(f"Training Partition        : {results['train_samples']} samples")
    print(f"Testing Partition         : {results['test_samples']} samples")
    print(f"LSTM Model MAE            : {results['lstm_mae']:.2f}%")
    print(f"LSTM Model RMSE           : {results['lstm_rmse']:.2f}%")
    print(f"Baseline MAE (Persistence): {results['baseline_mae']:.2f}%")
    print(f"Baseline RMSE             : {results['baseline_rmse']:.2f}%")
    improvement = results["mae_improvement_pct"]
    print(f"MAE Delta vs Baseline     : {improvement:+.1f}%")
    print(f"Artifacts Saved To        : {predictor.model_path.parent}")
    print("================================================================\n")


if __name__ == "__main__":
    main()
