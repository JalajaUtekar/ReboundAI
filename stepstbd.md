**Superseded** — this draft's structure was merged into `docs/ROADMAP.md` on 2026-10-17 with several corrections (120 students not 75, 70%/40% two-tier labels not a single 40% threshold, resolved CV fold count, firm best-model-wins deployment, RBAC scoped to Student-tier-only for the hackathon demo). Read `docs/ROADMAP.md` for the current authoritative version; this file is kept for reference only.

---

# Rebound AI (ScholarTrack + Lecture2Revision) — Complete Master Roadmap & Execution Prompts

> **Project Goal**: A production-grade, hackathon-ready early-warning and adaptive remediation system for TY Computer Engineering students (MMCOE).
> **Primary Engine**: **ScholarTrack** — forecasts student failure risk before end-term exams using dynamic in-term signals (marks, attendance, velocity, and submission delays).
> **Gated Assistant**: **Lecture2Revision** — activates exclusively for flagged student-subject pairs, providing grounded revision kits, SM-2 flashcard schedules, and LSTM score forecasting.

---

## 1. System Vision & The 11 Core Requirements

This roadmap merges the instructor/mentor checklist with continuous dynamic forecasting principles:

1. **Synthetic Data with 5–15% Minority Class**: Fixed cohort of 75 TY students (450 student-subject records) with realistic gradual decline trajectories, never abrupt drops.
2. **Class Weights**: Penalty weighting in loss functions inversely proportional to class frequencies to counteract skew.
3. **Custom Recall-Targeted Cutoff**: Threshold tuning on validation folds prioritizing high recall ($\ge 88\text{--}92\%$) to ensure at-risk students are never missed.
4. **Train-Fold-Only Oversampling**: Synthetic noise/SMOTE applied exclusively within training folds to avoid optimistic bias.
5. **Zero Test/Validation Contamination**: Validation and test sets remain strictly on empirical, untampered class distributions.
6. **Focal Loss Option**: Hard-example mining via Focal Loss ($\alpha=0.25, \gamma=2.0$) for instances where class weights alone underperform.
7. **Feature Skew Analysis & Standardization**: Log/Power transformations for skewed lateness metrics and leakage-free `StandardScaler` fitted strictly on training folds.
8. **Future Extension — Google Classroom API**: Automated ingestion pipeline for coursework, due dates, and submission timestamps.
9. **Tri-Model Benchmark**: 1D Temporal CNN matched against Scikit-Learn Logistic Regression and Gradient Boosting (HistGradientBoosting / LightGBM) under identical cross-validation.
10. **Explainable AI (SHAP)**: Granular feature attribution for each flagged student (e.g., *"Flagged: Attendance velocity fell by -24% and submission delay increased by 3.8 days"*).
11. **Imbalance-Robust Evaluation**: Primary metrics are Recall, Precision, PR-AUC, F1-Score, and Confusion Matrix (never raw accuracy).

---

## 2. Dynamic Forecasting: The "Crowd Kinematics" Analogy

In physical crowd analysis, predicting crowd congestion requires dynamic motion signals (density, flow velocity, acceleration) rather than static headcount thresholds. Similarly, ScholarTrack moves beyond binary `if attendance < 75%` logic to model **Academic Trajectory Kinematics**:

```
Static Snapshot (Inadequate)          Dynamic Kinematics (ScholarTrack)
┌─────────────────────────┐          ┌────────────────────────────────────────────────────────┐
│ Current Marks: 44%      │          │ Current Marks: 44%                                     │
│ Current Attendance: 76% │   VS     │ Grade Velocity (dv/dt): -8.5% over last 2 checkpoints  │
│ (Looks Passing / Safe)  │          │ Attendance Velocity: -12.0%/wk (downward momentum)     │
│                         │          │ Delay Acceleration: +2.1 days late per submission      │
│                         │          │ Proximity Deficit: 1.0% above failure boundary         │
└─────────────────────────┘          │ ➔ FORECAST: High Risk of Failure before Term End       │
                                     └────────────────────────────────────────────────────────┘
```

### Engineered Dynamic Features:
1. **Academic Density / Margin Deficit**:
   - $\text{Deficit}_{\text{att}} = \text{attendance\_pct} - 75.0$
   - $\text{Deficit}_{\text{marks}} = \text{internal\_marks\_pct} - 40.0$
2. **Performance Velocity ($\Delta \text{score} / \Delta t$)**:
   - First-order difference between unit test 2 and unit test 1, and rolling delta of assignment marks.
3. **Attendance Velocity / Momentum**:
   - Slope of attendance across the trailing 3-week window ($\frac{\Delta \text{att}}{\Delta \text{weeks}}$).
4. **Submission Latency Acceleration**:
   - Rate of increase in submission lateness ($\text{days\_late}_t - \text{days\_late}_{t-1}$).
5. **Consistency Metric**:
   - Rolling standard deviation of submission scores (volatility indicator).

---

## 3. Current State vs. Remaining Execution

| Phase / Step | Status | Artifacts / Locations |
| :--- | :--- | :--- |
| **Step 1: Synthetic Cohort** | **DONE** | `scripts/data_generation/`, `data/synthetic_students.json` (34 passing tests) |
| **Basic UI Scaffold** | **DONE** | `web/` (Next.js 15, Tailwind, student list, MMCOE logo) |
| **Step 2: Dynamic Feature Pipeline** | **TO BE BUILT** | `scripts/feature_engineering/`, checkpoint-prefix rows, velocity & skew logic |
| **Step 3: Model Training & SHAP** | **TO BE BUILT** | `scripts/training/`, CNN vs GBDT vs LogReg, CV, Focal Loss, SHAP explainer |
| **Step 4: Pure-NumPy Inference** | **TO BE BUILT** | Lightweight forward pass for serverless execution without heavy TensorFlow |
| **Step 5: Supabase Integration** | **TO BE BUILT** | Database migrations, auth, risk tables, and cohort seeding |
| **Step 6: Serverless Backend API** | **TO BE BUILT** | `/api/predict.py`, `/api/rebound.py` linking ScholarTrack to Lecture2Revision |
| **Step 7: Production Web Frontend** | **TO BE BUILT** | Registration gating, risk velocity cards, SHAP breakdowns, gated revision |
| **Step 8: Google Classroom Bridge**| **DOCUMENTED** | Integration architecture, scopes, and schema adapters |

---

## 4. Master Prompts for Anthropic Models

Use these self-contained prompts to execute each remaining phase sequentially with Claude / Anthropic models.

---

### PROMPT 1: Dynamic Feature Engineering & Prefix Row Generation (Step 2)

```text
You are a Senior Machine Learning Engineer building the feature engineering pipeline for Rebound AI's ScholarTrack engine.

### Context & Files:
- Spec: `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` (§3, §4, §5)
- Input Data: `data/synthetic_students.json` generated by `scripts/data_generation/generator.py`
- Checkpoint sequence: 7 chronological events defined in `CHECKPOINTS_IN_TIME_ORDER`:
  1. (submission, 0, wk2)
  2. (submission, 1, wk4)
  3. (submission, 2, wk6)
  4. (unit_test, 0, wk6)
  5. (submission, 3, wk8)
  6. (submission, 4, wk10)
  7. (unit_test, 1, wk12)

### Task:
Build a leakage-free feature engineering module in `scripts/feature_engineering/`:
1. `extractor.py`:
   - For each student-subject record, construct training rows at "as-of" checkpoint prefixes (e.g., as of checkpoint 2, 3, 4, 5, 6, 7).
   - NEVER include future checkpoints or the final ETE mark in the feature vector.
   - Ground-truth label is the student-subject's final `outcome.at_risk` boolean (0 or 1).
2. Dynamic Kinematic & Velocity Features (Inspired by crowd motion dynamics):
   - In addition to standard static features (`internal_marks_pct`, `assignment_marks_pct`, `attendance_pct`, `on_time_submission_ratio`, `avg_days_late`), calculate:
     a) `attendance_velocity`: Change in attendance percentage over the last 2 observed checkpoints (slope/trend).
     b) `marks_velocity`: Change in assignment/test scores between the latest and preceding evaluations.
     c) `delay_acceleration`: Change in `avg_days_late` between recent submissions (increasing tardiness signal).
     d) `attendance_deficit_margin`: `current_attendance_pct - 75.0` (distance to university failure threshold).
     e) `marks_deficit_margin`: `current_internal_marks_pct - 40.0` (distance to passing threshold).
3. Skew Handling & Transformer:
   - Provide an inspection function `inspect_feature_skew(df)` calculating skewness for each numeric feature.
   - Apply logarithmic or Box-Cox/Yeo-Johnson transforms on heavily right-skewed latency metrics (`avg_days_late`, `delay_acceleration`).
   - Implement `LeakageFreeStandardizer`: fits `StandardScaler` strictly on training folds and transforms validation/test sets without target or temporal leakage.
4. Export:
   - Generate `data/training_rows.json` and a flattened `data/training_rows.csv` with metadata columns (`student_id`, `subject_code`, `checkpoint_idx`) and feature columns.
5. Unit Tests:
   - Create `tests/feature_engineering/test_extractor.py` verifying:
     - No feature values change retrospectively when advancing checkpoints.
     - Velocity features correctly compute deltas.
     - Row counts match exactly: 75 students × 6 subjects × valid checkpoint prefixes.

Use test-driven development. Run pytest to ensure 100% pass rate before completing.
```

---

### PROMPT 2: Tri-Model Training, Imbalance Handling & Evaluation (Step 3)

```text
You are an expert ML Researcher implementing the model training and benchmark pipeline for ScholarTrack.

### Objective:
Train and compare a 1D Temporal CNN against Gradient Boosting (HistGradientBoosting / LightGBM) and Logistic Regression, incorporating all 11 mentor requirements for small-sample, imbalanced time-series tabular data.

### Input Data:
`data/training_rows.csv` generated by Step 2 (450 student-subject records across checkpoint prefixes, ~5-15% positive minority class).

### Requirements to Implement in `scripts/training/train_models.py`:
1. Cross-Validation Protocol:
   - Implement Stratified K-Fold CV (5 folds) grouped by `student_id` (so no student's prefix rows leak across train and validation folds).
2. Class Imbalance Countermeasures:
   - Compute `class_weight='balanced'` for all models.
   - Train-Fold-Only Oversampling: Apply SMOTE (with Gaussian noise jittering) STRICTLY within the training split of each fold.
   - Strict Constraint: Validation and test splits MUST NEVER be oversampled.
3. Focal Loss Option for Neural Net:
   - Implement binary Focal Loss ($FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ with $\alpha=0.25, \gamma=2.0$) in Keras/PyTorch as a selectable loss alongside weighted binary cross-entropy.
4. Model Implementations:
   - Model A: 1D Temporal CNN (matching course syllabus: `Conv1D(16) -> BatchNorm -> ReLU -> MaxPooling1D -> Conv1D(32) -> GlobalAveragePooling1D -> Dense(16) -> Dropout(0.3) -> Dense(1, sigmoid)`).
   - Model B: Gradient Boosting (`HistGradientBoostingClassifier` or `LGBMClassifier` with `class_weight='balanced'`, max_depth=3, min_samples_leaf=5).
   - Model C: Baseline L2-regularized `LogisticRegression(class_weight='balanced')`.
5. Threshold Tuning for High Recall:
   - On the validation folds, sweep probability thresholds from 0.10 to 0.90 to find the optimal decision cutoff hitting a target recall $\ge 90\%$.
   - Output the tuned cutoff rather than defaulting to 0.5.
6. Evaluation Metrics:
   - For every model, log: Recall, Precision, F1-Score, PR-AUC (Precision-Recall AUC), ROC-AUC, Brier Score, and Confusion Matrix. Do not report bare accuracy as a primary metric.
7. Model Artifact Export:
   - Save the best-performing models to `models/scholar_track_best_model.*`.

Create `tests/training/test_training_pipeline.py` to assert that:
- CV splits strictly separate student IDs.
- Training fold class ratio changes under SMOTE, while validation fold ratio remains untouched.
- Output metrics dictionaries contain all mandatory imbalance keys.

Execute the training script, print the comparative benchmark table in Markdown, and commit the saved model artifacts.
```

---

### PROMPT 3: Explainable AI with SHAP Interpretability (Step 3 Extension)

```text
You are an Explainable AI (XAI) specialist implementing student risk explanations for ScholarTrack.

### Objective:
Implement a SHAP (SHapley Additive exPlanations) pipeline in `scripts/training/explainability.py` so that every flagged student comes with a human-readable diagnosis of WHY they were flagged, directly referencing their dynamic signals (e.g., attendance drop, submission lateness, negative velocity).

### Specifications:
1. Explainer Setup:
   - Use `shap.TreeExplainer` for the Gradient Boosting model and `shap.Explainer` / `shap.KernelExplainer` for the 1D CNN / NumPy model.
   - Fit/background summary using the training fold distribution (`shap.kmeans` or median baseline).
2. Diagnosis Generator (`generate_student_explanation`):
   - Input: Student-subject feature vector at current checkpoint.
   - Output: Structured JSON containing:
     - `predicted_risk_probability`: float (0.0 to 1.0)
     - `is_flagged`: boolean (based on custom recall cutoff)
     - `top_risk_factors`: List of top 3 features with highest positive SHAP values (pushing towards failure).
     - `protective_factors`: List of features with negative SHAP values (mitigating risk).
     - `human_readable_summary`: Natural language string suitable for faculty advisors (e.g., "Student's attendance velocity dropped by -15% over the last 2 weeks and assignment lateness averaged 3.4 days, putting them in the high-risk zone for SPOS.").
3. Visualizations:
   - Save a cohort summary plot to `assets/shap_cohort_summary.png`.
   - Save sample waterfall / force plots for the 4 synthetic at-risk students into `assets/shap_examples/`.
4. Testing:
   - Add tests in `tests/training/test_explainability.py` ensuring that `generate_student_explanation()` returns valid keys, non-empty text, and numeric SHAP attributions summing approximately to the model output margin.

Write the code, run the unit tests, and verify generated explanations for flagged students in `data/synthetic_students.json`.
```

---

### PROMPT 4: Pure-NumPy Forward Pass for Serverless Inference (Step 4)

```text
You are a Systems and ML Engineer optimizing ScholarTrack for ultra-lightweight serverless deployment on Vercel without heavy C-dependencies (TensorFlow/Torch).

### Context:
Lecture2Revision uses a pure-NumPy LSTM implementation (`Lecture2Revision/lstm_model.py:NumPyLSTMModel`) that loads exported weights and computes forward passes with zero machine-learning framework overhead. We need the exact same pattern for ScholarTrack.

### Task in `scripts/inference/`:
1. Weight Exporter (`export_weights.py`):
   - Extract learned weights and biases from the trained Keras 1D CNN or tree parameters from the Gradient Boosting model.
   - Export parameters as compressed NumPy archive `models/scholar_track_weights.npz` and JSON config `models/scholar_track_meta.json` (containing standardizer means/scales, velocity parameters, and the tuned recall threshold).
2. Pure-NumPy Engine (`numpy_forecaster.py`):
   - Implement `NumPyScholarTrackModel`:
     - Standardizes input vector using saved mean/std arrays.
     - Computes 1D convolutions (`np.convolve` / sliding window dot product), ReLU activations, MaxPool1D, GlobalAveragePooling, and Dense linear layers with sigmoid activation.
     - For GBDT alternative: Provide a lightweight decision-tree traversal class in pure Python/NumPy if GBDT outperformed CNN.
     - Implements `predict_proba(X)` and `predict(X, threshold=None)`.
3. Numerical Verification Test (`tests/inference/test_numpy_parity.py`):
   - Load the real dataset rows from Step 2.
   - Compare predictions between the Keras/Sklearn trained model and the pure-NumPy model.
   - Assert `np.allclose(keras_preds, numpy_preds, atol=1e-5)`.
   - Measure execution latency (must be < 10ms per inference).

Implement and run pytest to guarantee parity.
```

---

### PROMPT 5: Supabase Schema, Migrations & Seed Data (Step 5)

```text
You are a Database Engineer designing the Postgres schema for Rebound AI in Supabase.

### Requirements:
Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` (§2, §3, §7) and `Lecture2Revision/database.py`.

### Task:
1. Create SQL Migration `supabase/migrations/20261017000000_rebound_core_schema.sql`:
   - `students`: `id` (UUID PK), `prn` (VARCHAR unique), `full_name`, `email` (unique), `branch` (VARCHAR, check in AI&DS, CE, IT, E&TC, ME, EE), `academic_year` (VARCHAR, check in FY, SY, TY, BE), `avatar_url`, `created_at`.
   - `student_subjects`: `id` (UUID PK), `student_id` (FK), `subject_code` (VARCHAR), `subject_name` (VARCHAR), `subject_type` (fixed, mdm, oel, elective), `created_at`.
   - `assessment_checkpoints`: `id` (UUID PK), `student_subject_id` (FK), `checkpoint_idx` (INT), `event_type` (unit_test, submission), `week_number` (INT), `score_pct` (FLOAT), `attendance_pct` (FLOAT), `is_late` (BOOLEAN), `days_late` (FLOAT).
   - `risk_predictions`: `id` (UUID PK), `student_subject_id` (FK), `checkpoint_idx` (INT), `risk_probability` (FLOAT), `is_flagged` (BOOLEAN), `attendance_velocity` (FLOAT), `marks_velocity` (FLOAT), `top_shap_factors` (JSONB), `remediation_status` (pending, active, completed), `created_at`.
   - Port Lecture2Revision tables (Postgres-ified): `lectures`, `revision_kits`, `flashcards`, `student_flashcard_reviews` (SM-2 data: ease_factor, interval, repetitions), `quizzes`, `quiz_attempts`.
2. Row-Level Security (RLS):
   - Enable RLS on all tables.
   - Students can read and update only their own rows (`auth.uid() = student_id`).
   - Instructors/Advisors (role check) can read cohort risk predictions.
3. Seed Script `supabase/seed.py`:
   - Ingests `data/synthetic_students.json` and populates the Supabase database with the 75 TY CE students and their 6 subject allocations.
4. Documentation `docs/SUPABASE_SETUP.md`:
   - Step-by-step setup guide for project creation, running migrations via Supabase CLI, setting environment variables (`NEXT_PUBLIC_SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`), and configuring Google OAuth for future rollout.

Verify the SQL schema syntax and generate the migration files.
```

---

### PROMPT 6: Unified Serverless Backend API (Step 6)

```text
You are a Backend Engineer building the serverless API linking ScholarTrack and Lecture2Revision.

### Architecture:
Next.js API routes or lightweight Python serverless functions (`web/app/api/` or `api/*.py`) running on Vercel.

### Endpoints to Implement:
1. `POST /api/predict-risk`:
   - Ingests current student subject checkpoints.
   - Calculates dynamic velocity and deficit features.
   - Runs `NumPyScholarTrackModel` inference.
   - Generates SHAP explanation JSON.
   - If `is_flagged == True`, flags the subject for Lecture2Revision intervention.
2. `POST /api/rebound/process-lecture`:
   - Gated: Only callable if student has an active risk flag in the subject.
   - Receives uploaded lecture PDF.
   - Calls ported `Lecture2Revision/lecture_processor.py` (PyMuPDF) and `Lecture2Revision/transformer_service.py` (offline extractive NLP default).
   - Generates and stores structured revision notes, definitions, flashcards, and quizzes.
3. `POST /api/rebound/sm2-review`:
   - Handles flashcard review ratings (0-5).
   - Executes SuperMemo SM-2 algorithm updating interval, repetitions, and ease factor.
4. `GET /api/rebound/adaptive-schedule`:
   - Ported from `Lecture2Revision/adaptive_engine.py`.
   - Prioritizes topics based on: `Urgency = Error_Count + SM2_Overdue + ScholarTrack_Risk_Boost`.

Write end-to-end integration tests using pytest/requests verifying that unflagged students are blocked from creating revision kits and flagged students receive tailored revision schedules.
```

---

### PROMPT 7: Production Next.js Dashboard & Gated UI (Step 7)

```text
You are a Senior Frontend Engineer building the Rebound AI user interface in Next.js 15 (App Router, Tailwind CSS, Lucide icons, TypeScript).

### Requirements:
1. Registration & Gating Flow (`/register`):
   - Step 1: Basic details (PRN, Name, Email, Photo).
   - Step 2: Branch selection: Only `CE (Computer Engineering)` is active; other branches display "Coming Soon" badges.
   - Step 3: Year selection: Only `TY (Third Year)` is active.
   - Step 4: Subject selection: Pre-populates DSML, CN, SPOS; dropdowns for 1 MDM, 1 OEL, 1 Elective.
2. ScholarTrack Early Warning Dashboard (`/dashboard`):
   - Cohort Overview: Total students (75), At-Risk Count (5-15%), Class Health Index.
   - Student Risk Table with MMCOE styling:
     - Student PRN, Name, Subject Breakdown.
     - Dynamic Velocity Badges: Colored trend arrows indicating `attendance_velocity` (e.g., "↓ -12%/wk") and `marks_velocity`.
     - Risk Score Gauge (e.g. 84% probability of falling behind).
     - "Why Flagged?" expandable drawer displaying the top 3 SHAP attribution factors.
3. Gated Lecture2Revision Workspace (`/rebound/[subjectId]`):
   - Gating rule: Accessible ONLY if the student is flagged at-risk in that subject.
   - PDF Lecture Upload zone with drag-and-drop.
   - Grounded Revision Kit viewer: Executive Summary, Key Formulas, Practice Quizzes.
   - Interactive SM-2 Flashcard Deck with spaced repetition ratings (Again, Hard, Good, Easy).
   - Adaptive Timetable displaying upcoming revision deadlines adjusted by risk severity.

Ensure full responsiveness, dark/light theme support, accessible ARIA labels, and clean component isolation under `web/app/` and `web/components/`.
```

---

### PROMPT 8: Future Extension — Google Classroom Integration (Step 8)

```text
You are a Solutions Architect designing the Google Classroom synchronization module for future release.

### Deliverable:
Create an architectural design and code specification in `docs/GOOGLE_CLASSROOM_INTEGRATION.md`:
1. OAuth & Permission Scopes:
   - `https://www.googleapis.com/auth/classroom.courses.readonly`
   - `https://www.googleapis.com/auth/classroom.coursework.students.readonly`
   - `https://www.googleapis.com/auth/classroom.student-submissions.students.readonly`
2. Data Synchronization Pipeline:
   - Polling/Webhook daemon mapping Classroom coursework to ScholarTrack `assessment_checkpoints`.
   - Real-time lateness calculation: comparing `submission.creationTime` with `courseWork.dueDate` to compute `days_late` and `delay_acceleration`.
3. Graceful Fallback:
   - Dual-mode operation: Manual CSV gradebook upload vs. automated Google Classroom sync.
4. Data Privacy:
   - Token storage security and student data protection standards.
```

---

## 5. Summary Execution Order

1. **Run Prompt 1** $\rightarrow$ Build dynamic velocity feature extractor and prefix rows (`scripts/feature_engineering/`).
2. **Run Prompt 2** $\rightarrow$ Train & benchmark CNN vs. GBDT vs. Logistic Regression with class weights, train-fold SMOTE, and recall tuning.
3. **Run Prompt 3** $\rightarrow$ Implement SHAP explainability for student risk factor diagnostics.
4. **Run Prompt 4** $\rightarrow$ Export trained model weights to pure-NumPy serverless inference module.
5. **Run Prompt 5** $\rightarrow$ Create Supabase migration scripts and seed cohort data.
6. **Run Prompt 6** $\rightarrow$ Deploy Python serverless API linking ScholarTrack flags to Lecture2Revision.
7. **Run Prompt 7** $\rightarrow$ Polish the Next.js frontend with registration gating, velocity indicators, and SHAP cards.
8. **Archive Prompt 8** $\rightarrow$ Commit Google Classroom integration architecture into project docs.
