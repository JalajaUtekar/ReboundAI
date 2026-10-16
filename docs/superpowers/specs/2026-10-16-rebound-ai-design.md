# Rebound AI — Design Spec

*"Fall behind less. Bounce back faster."*

Department of Computer Engineering, Marathwada Mitramandal's College of Engineering (MMCOE), Pune. Built for the CE24MDM305 — Foundations of GenAI mini-hackathon, merging two problem statements:

1. **ScholarTrack — Detecting Students at Risk of Falling Behind** (the main problem statement)
2. **Lecture2Revision — Convert Lectures Into Adaptive Revision** (the assistant feature, activated only for students ScholarTrack flags)

---

## 1. Product framing

ScholarTrack is the product. It is a genuine early-warning classifier that watches a student's in-progress semester (marks, attendance, submission timeliness) and flags *which subject* they're at risk of falling behind in, **before** the end-of-semester exam happens — not a dashboard that tells you what you already know.

Lecture2Revision is *not* a parallel feature sitting next to it. It only activates as the response to a flag: when ScholarTrack marks a student at risk in, say, CN, the dashboard surfaces "here's your Rebound plan for CN" — upload the CN lecture, get grounded revision notes/flashcards/quizzes, and an adaptive SM-2 + LSTM-driven revision schedule for that subject. A student with no flags never needs to think about Lecture2Revision at all.

---

## 2. Registration & onboarding flow

This is a real app for MMCOE students, so registration has to gate correctly even though only one cohort is live at launch.

**Step 1 — Basic details.** Name, PRN, Email, Photograph.

**Step 2 — Branch selection.** Options: AI&DS, CE, IT, E&TC, ME, EE. Only **CE (Computer Engineering)** is selectable; the other five render disabled/greyed with a "coming soon" state. (Enabling them later is just flipping a flag — no schema change needed, the `branch` column already accepts any of the six values.)

**Step 3 — Academic year selection.** Options: FY, SY, TY, BE. Only **TY (Third Year)** is selectable; same disabled treatment for the rest.

**Step 4 — Subject setup (TY CE only, for now).** Three subjects are fixed and pre-entered for every TY CE student: **DSML, CN, SPOS**. Three more are chosen via dropdown, one each:

- **MDM** (Multidisciplinary Minor): GenAI, Privacy and Security Engineering, Robotics and Automation, Electric Vehicle Technology, Sustainable and Smart Systems, Embedded Systems
- **OEL** (Open Elective): Practical Japanese Communication, Workplace Ethics in German Language, Product and Brand Management, Strategic Management, Operations and Supply Chain Management, Organizational Behaviour
- **Elective**: Cloud Computing, Information Systems, Prompt Engineering, Game Development

So every TY CE student ends up with exactly **6 subjects**: DSML, CN, SPOS + their chosen MDM + OEL + Elective.

---

## 3. Subjects & grading schema

| Subject slot | Fixed / Choice | Components |
|---|---|---|
| Data Science and Machine Learning (DSML) | Fixed | TH(40) + ETE(60) + PR(25) |
| Computer Networks (CN) | Fixed | TH(40) + ETE(60) + TW(25) |
| System Programming and Operating Systems (SPOS) | Fixed | TH(40) + ETE(60) + PR(25) |
| MDM (1 of 6, student's choice) | Choice | TH(40) + ETE(60) + PR(25) |
| OEL (1 of 6, student's choice) | Choice | TH(40) + ETE(60) + TW(25) |
| Elective (1 of 4, student's choice) | Choice | TH(40) + ETE(60) + TW(25) |

- **TH (40)** — in-semester theory, modeled as 2 periodic unit tests (e.g. 20 marks each) during the term. Available *during* the semester.
- **PR / TW (25)** — practical / term work, modeled as 5-6 lab/assignment submissions across the term, each with a due date, so lateness is a real, derivable signal. Available *during* the semester.
- **Attendance** — modeled weekly across the term. Available *during* the semester.
- **ETE (60)** — end-of-semester exam. **Held out from the model's input features entirely** — it only exists at the very end, which is exactly what makes this a forecasting problem rather than a classification-of-a-finished-trend problem.

---

## 4. Synthetic dataset

- **75 students**, Department of Computer Engineering, TY. Each is randomly assigned one MDM, one OEL, one Elective (from the lists above) in addition to the 3 fixed subjects — 6 subject-records per student, 450 subject-level training rows.
- **5-15% minority at risk**, enforced at generation time (not an emergent property of random noise — the generator targets a risk rate in that range per run).
- **Per-subject, per-checkpoint feature vector** (5 features, same shape as Lecture2Revision's existing LSTM, deliberately — one consistent "sequence feature" convention across the whole app):
  `[internal_marks_pct, assignment_marks_pct, attendance_pct, on_time_submission_ratio, avg_days_late]`
- **Label, grounded in real rules, not an arbitrary synthetic cutoff:** a student is *at risk* in a subject if attendance ends below 75% (the actual university ETE-eligibility rule) **OR** the weighted aggregate (TH+ETE+PR/TW, normalized to 100) ends below 40% (the actual SPPU passing threshold). ETE is used only here, in computing the label — never as an input feature.
- **No-leakage training construction:** for each student-subject, generate training examples at multiple "as-of" checkpoints through the term (after unit test 1, after unit test 1+2, etc.), where X = everything known *as of that checkpoint* and y = the eventual end-of-semester outcome. The model never sees the checkpoint that defines its own label.

Output: a reproducible `scripts/generate_demo_data.py`-equivalent that writes directly into the Supabase schema (Section 7), plus a flat CSV/JSON export for the training notebook.

---

## 5. ScholarTrack modeling pipeline

Directly maps to the mentor's review checklist:

1. **Synthetic 75-student, 5-15% minority generation** — Section 4.
2. **Class weights** passed to the loss function during training.
3. **Stratified k-fold (or leave-one-out) cross-validation**, not a single train/val/test split — with ~4-11 positives out of 75, a single split is too noisy to trust. Cutoff threshold is tuned per the CV protocol against a target recall, not left at the default 0.5.
4. **Oversampling** (SMOTE or random) applied to **training folds only**.
5. **Test/validation folds are never oversampled** — evaluation stays on the real class distribution.
6. **Focal loss** offered as a selectable alternative to weighted cross-entropy, for if the imbalance proves too extreme for class-weighting alone.
7. **Feature skew check + standardization** (fit strictly on training folds, same leakage discipline as Lecture2Revision's existing `StandardScaler` usage) before any model sees the data.
8. **Baseline comparison**: Logistic Regression and Gradient Boosting (scikit-learn), trained and evaluated under the identical CV protocol as the Conv1D model — the neural net has to beat the baselines, not just exist.
9. **SHAP** on top of the trained model, so every flagged student comes with *why* ("attendance trend −32%, 3 late TW submissions") instead of a bare probability. SHAP's model-agnostic explainers (e.g. `KernelExplainer`) work directly against a plain prediction function, so this works whether the production model is served via Keras or the NumPy forward pass — no conflict with Section 6's serving approach.
10. **Metrics reported**: recall, precision, F1, PR-AUC, and a confusion matrix — never bare accuracy, since always predicting "not at risk" already scores 85-95% on this class balance and would be a meaningless win.
11. **Google Classroom integration** is noted as future work in the README; not built now.

**Architecture**: Conv1D (as in the original ScholarTrack notebook, matching the course's CNN unit) over the *checkpoints available so far* — not the whole completed term — producing a per-subject risk probability. Trained in a notebook/script with Keras; weights then exported and re-implemented as a small pure-NumPy forward pass (mirroring `lstm_model.py`'s existing `NumPyLSTMModel` pattern) so the production serverless function never needs TensorFlow installed.

---

## 6. Lecture2Revision integration — unchanged logic, gated activation

Carried over from Lecture2Revision almost unmodified:

- `lecture_processor.py` (PyMuPDF extraction) — unchanged.
- `transformer_service.py` — unchanged. `offline_extractive` (frequency-scored sentence extraction + pattern-matched definitions/MCQs) stays the *default and the real "modeled" technique* for the course; Gemini/HF/OpenAI stay available as an optional switch, never the headline.
- `flashcard_service.py` (SM-2 spaced repetition) — unchanged.
- `lstm_model.py` (per-topic next-quiz-score forecasting, already TensorFlow-free at inference) — unchanged.
- `adaptive_engine.py` — unchanged, **plus one new term**: a subject's current ScholarTrack risk probability feeds into `calculate_topic_priority` as an additional urgency boost, so topics under a flagged subject jump to the top of that student's revision plan automatically.

**Activation rule**: a student only sees the Lecture2Revision workflow for a subject once ScholarTrack has flagged that subject at risk. Subjects in good standing show their ScholarTrack status only, with no revision-kit UI.

---

## 7. Architecture & stack

- **Frontend**: Next.js (App Router) + TypeScript + Tailwind — UI only, calls the backend over HTTP. MMCOE branding (`assets/mmcoe-logo.png`, "Department of Computer Engineering") on login/dashboard.
- **Backend**: Vercel Python Serverless Functions (`/api/*.py`) — PyMuPDF, the extractive generator, SM-2, the LSTM, and the new ScholarTrack NumPy-served CNN all run here, carried over from the two source projects with minimal changes (Section 5 and 6).
- **Database**: Supabase Postgres. Same entities as Lecture2Revision's existing SQLite schema, extended with ScholarTrack's academic structure:
  - `students` (name, prn, email, photo_url, branch, academic_year — gated to CE/TY for now)
  - `student_subjects` (student_id, subject_slug, chosen_variant — e.g. which MDM/OEL/Elective)
  - `assessment_checkpoints` (student_id, subject_slug, checkpoint_type [unit_test/submission/attendance_week], value, due_date, submitted_date)
  - `risk_predictions` (student_id, subject_slug, risk_probability, shap_explanation, predicted_at)
  - `lectures`, `learning_materials`, `flashcards`, `flashcard_reviews`, `quiz_questions`, `quiz_attempts`, `quiz_responses`, `revision_schedules` — carried over from Lecture2Revision's schema, Postgres-ified.
- **Auth**: Supabase Auth.
  - **Now**: email/password, demo account seeded via a seed script, not re-entered in chat.
  - **Later**: Google OAuth — enabled from the Supabase dashboard (Authentication → Providers → Google) using a Google Cloud OAuth client ID/secret the user creates themselves; the app then just calls `supabase.auth.signInWithOAuth({ provider: 'google' })`. A step-by-step setup doc will ship with the repo.
- **Deployment**: one Vercel project — Next.js app + Python API routes side by side, Supabase as the external managed Postgres + Auth.

---

## 8. Out of scope / future work

- Branches other than CE, years other than TY (UI already gates for this; just flip the disabled flags later).
- Google Classroom sync (mentor's point 8).
- Google OAuth itself (needs the user's own Google Cloud credentials — documented, not auto-configured).
- Any backend other than `offline_extractive` being the default revision-generation path (Gemini etc. stay optional).

---

## 9. Testing

- Pytest suite carried over/extended from Lecture2Revision for the untouched modules (SM-2, extractive generator, LSTM, adaptive engine).
- New tests for: synthetic data generator's risk-rate bounds (5-15%), the no-leakage checkpoint construction, the NumPy CNN forward pass matching the Keras model's output within tolerance, and the risk-boost term in `calculate_topic_priority`.
- Model evaluation report (CV metrics, baseline comparison, SHAP examples) generated as part of the training script, not hand-written.
