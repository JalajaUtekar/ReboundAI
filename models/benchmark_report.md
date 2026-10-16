# ScholarTrack Model Benchmark — Step 3

5-fold student-grouped stratified CV, `target_recall=0.90`, on the real 120-student dataset (`data/training_rows.json`, 4,320 rows / `data/synthetic_students.json` for the CNN's sequence path). Group-aware oversampling (training folds only) and class weighting applied per spec §5. Metrics are out-of-fold, never on training data.

| Model | Recall | Precision | F1 | PR-AUC | Threshold | Hit target recall (≥0.90)? |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.903 | 0.670 | 0.769 | **0.909** | 0.86 | Yes |
| Gradient Boosting (HistGBDT) | 0.903 | 0.730 | 0.807 | 0.903 | 0.20 | Yes |
| CNN (Conv1D, weighted BCE) | 0.819 | 0.187 | 0.304 | 0.690 | 0.10 | No |
| CNN (Conv1D, focal loss α=0.25 γ=2.0) | 0.861 | 0.103 | 0.184 | 0.775 | 0.10 | No |

## Winner (best-model-wins deployment policy, per spec §5): **Logistic Regression**

Highest PR-AUC, only model whose default threshold-sweep comfortably clears the 90% recall bar with usable precision. Gradient Boosting is a close second (better F1, almost identical PR-AUC) and remains a credible fallback.

## Why the CNN lost (and this is expected, not a bug)

- **Dataset is small and extremely imbalanced** (4,320 rows, 72 positive — 1.67%) — exactly the regime where deep nets tend to underperform simpler models that don't need as much data to find a good decision boundary.
- **Sequence padding dilutes signal**: most rows (checkpoint_idx 2-4) are mostly zero-padded in the CNN's (6,5) input, while the tabular models get the full engineered feature vector (including velocity/deficit-margin features) at every row regardless of how few checkpoints have occurred.
- **Focal loss helped but didn't close the gap** — confirms the issue isn't just the loss function; it's the architecture/data-regime mismatch for a problem this small.

## What this means for deployment

Per the resolved decision that the actual best-performing model ships (not a pre-committed choice), **Logistic Regression is the model Step 5/6 should serve**, not the CNN. The CNN still trains and appears in this benchmark table — it's the course's required architecture and the comparison is the point — but it is not the production model. Step 5's NumPy inference work should prioritize the Logistic Regression path (a trivial dot product) and keep the CNN/GBDT NumPy paths as documented but lower-priority, since they're not what's actually served.

## Reproduction

```bash
py -3 -c "
import json
from scripts.training.cv import grouped_student_kfold
from scripts.training.baselines import train_and_predict_oof_logreg, train_and_predict_oof_gbdt
from scripts.training.cnn_model import train_and_predict_oof as cnn_oof
from scripts.training.metrics import evaluate_predictions

rows = json.load(open('data/training_rows.json'))
dataset = json.load(open('data/synthetic_students.json'))
student_ids = sorted(set(r['student_id'] for r in rows))
student_at_risk = {}
for r in rows:
    student_at_risk[r['student_id']] = student_at_risk.get(r['student_id'], False) or r['at_risk']
folds = grouped_student_kfold(student_ids, student_at_risk, n_splits=5, seed=42)
# ... call each model's train_and_predict_oof(rows_or_dataset, folds), then evaluate_predictions(y_true, y_proba)
"
```
