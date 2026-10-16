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

- **120 students** (the department's real TY CE branch size), Department of Computer Engineering, TY. Each is randomly assigned one MDM, one OEL, one Elective (from the lists above) in addition to the 3 fixed subjects — 6 subject-records per student, 720 subject-level instances.
- **10-15% minority at risk**, enforced at generation time (not an emergent property of random noise — the generator targets a risk rate in that range per run). At 120 students this is ~12-18 at-risk students — enough for proper 5-fold grouped cross-validation (Section 5), which the earlier 75-student/4-at-risk-student draw could not support.
- **Per-subject, per-checkpoint feature vector** (5 raw features, same shape as Lecture2Revision's existing LSTM, deliberately — one consistent "sequence feature" convention across the whole app):
  `[internal_marks_pct, assignment_marks_pct, attendance_pct, on_time_submission_ratio, avg_days_late]`
  Plus **engineered dynamic (velocity) features**, computed from consecutive checkpoints during feature engineering (not stored per-checkpoint in the raw generator, derived downstream): `attendance_velocity` (slope over the trailing checkpoints), `marks_velocity` (delta between latest and preceding marks checkpoint), `delay_acceleration` (change in `avg_days_late` between recent submissions), `attendance_deficit_margin` (`attendance_pct - 75.0`), `marks_deficit_margin` (`aggregate_pct_so_far - 70.0` — anchored to the risk threshold below, not the pass/fail threshold).
- **Two-tier label, grounded in real rules, not an arbitrary synthetic cutoff:**
  - ***At risk of falling behind*** (the actual ScholarTrack training target) — a student is at risk in a subject if attendance ends below 75% (the actual university ETE-eligibility rule) **OR** the weighted aggregate (TH+ETE+PR/TW, normalized to 100) ends below **70%**. This is deliberately more sensitive than outright failing — "falling behind" is meant to catch strugglers early, before they fail, which is the whole premise of the product's name.
  - ***Failing*** (a separate, more severe, informational fact — not the training target) — aggregate ends below **40%** (the actual SPPU passing threshold). Every failing student is automatically also at-risk (40% < 70%), but not every at-risk student is failing. Displayed to students/faculty as a harder status, never trained on directly.
  - ETE is used only here, in computing both labels — never as an input feature.
- **No-leakage training construction:** for each student-subject, generate training examples at multiple "as-of" checkpoints through the term (after unit test 1, after unit test 1+2, etc.), where X = everything known *as of that checkpoint* and y = the eventual end-of-semester outcome. The model never sees the checkpoint that defines its own label.
- **At-risk trajectories are gradual, not a cliff.** A minority-class student-subject is generated as a genuine downward *trend* across checkpoints — marks, attendance, and submission timeliness each drift down with realistic noise over the term, crossing the risk thresholds only by the final checkpoints. No student jumps straight from "fine" to "below 75% attendance" between two checkpoints; the decline is visible several checkpoints before the final outcome, which is what makes early forecasting meaningful and testable (recall at checkpoint 3 vs. checkpoint 5 becomes a real metric, not a given).
- **Majority-class marks cluster near the risk threshold, not spread smoothly.** Real exam score distributions aren't a clean bell curve — moderation and grace-marking produce a visible bump of students just above a cutoff. The generator should bias a meaningful share of "doing fine" students' component marks to land just above **70%** (the new risk threshold, not 40%), rather than sampling uniformly/normally across the full passing range, so the synthetic distribution looks like a real gradebook rather than a textbook Gaussian. A smaller share of the at-risk minority should land below 40% (genuinely failing) so the "failing" fact has real positive examples too.

Output: a reproducible `scripts/generate_demo_data.py`-equivalent that writes directly into the Supabase schema (Section 7), plus a flat CSV/JSON export for the training notebook.

---

## 5. ScholarTrack modeling pipeline

Directly maps to the mentor's review checklist:

1. **Synthetic 120-student, 10-15% minority generation** — Section 4.
2. **Class weights** passed to the loss function during training.
3. **Stratified, student-grouped k-fold cross-validation (5 folds)**, not a single train/val/test split — a single split is too noisy to trust at this scale. Folds are grouped by `student_id` (not by row), since Section 4's no-leakage construction produces multiple correlated "as-of checkpoint" rows per student-subject — splitting at the row level would leak one student's prefix rows across train and validation. At 120 students / 10-15% minority (~12-18 at-risk students) this comfortably supports 5 folds; if a future run ever has fewer than 5 at-risk students, fall back to leave-one-(positive-student)-out. Cutoff threshold is tuned per the CV protocol against a target recall (≥88-92%), not left at the default 0.5.
4. **Oversampling** (SMOTE or random) applied to **training folds only**, and group-aware: a positive student-subject's 6 prefix rows are near-duplicates of the same trajectory, so naive row-level SMOTE mostly interpolates within one student's own decline rather than adding real diversity. Oversample at the student-subject level (duplicate/jitter a whole trajectory with noise) rather than treating individual prefix-rows as independent samples.
5. **Test/validation folds are never oversampled** — evaluation stays on the real class distribution.
6. **Focal loss** offered as a selectable alternative to weighted cross-entropy, for if the imbalance proves too extreme for class-weighting alone.
7. **Feature skew check + standardization** (fit strictly on training folds, same leakage discipline as Lecture2Revision's existing `StandardScaler` usage) before any model sees the data.
8. **Baseline comparison**: Logistic Regression and Gradient Boosting (scikit-learn), trained and evaluated under the identical CV protocol as the Conv1D model — the neural net has to beat the baselines, not just exist.
9. **SHAP** on top of the trained model, so every flagged student comes with *why* ("attendance trend −32%, 3 late TW submissions") instead of a bare probability. SHAP's model-agnostic explainers (e.g. `KernelExplainer`) work directly against a plain prediction function, so this works whether the production model is served via Keras or the NumPy forward pass — no conflict with Section 6's serving approach.
10. **Metrics reported**: recall, precision, F1, PR-AUC, and a confusion matrix — never bare accuracy, since always predicting "not at risk" already scores 85-95% on this class balance and would be a meaningless win.
11. **Google Classroom integration** is noted as future work in the README; not built now.

**Architecture**: Conv1D (as in the original ScholarTrack notebook, matching the course's CNN unit) over the *checkpoints available so far* — not the whole completed term — producing a per-subject risk probability. The CNN is always trained and always appears in the benchmark table (it's the course's required technique), but **the model that actually gets deployed is whichever of CNN / Gradient Boosting / Logistic Regression wins the CV benchmark** — deployment is not pre-committed to the CNN. All three therefore need a lightweight serving path: the CNN's and Logistic Regression's forward passes re-implemented as pure NumPy (mirroring `lstm_model.py`'s existing `NumPyLSTMModel` pattern), and Gradient Boosting served via a plain-Python/NumPy decision-tree traversal of its learned splits — so the production serverless function never needs TensorFlow, PyTorch, or a full scikit-learn/LightGBM runtime installed, whichever model wins.

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
- **Database**: Supabase Postgres. Same entities as Lecture2Revision's existing SQLite schema, extended with ScholarTrack's academic structure, plus the `profiles`/role table from Section 8:
  - `profiles` (id → auth.users.id, role [student/professor/hod/principal], department) — Section 8.
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

## 8. Roles & access control

Four-tier role-based access, enforced via Supabase Row-Level Security — a real expansion of Section 7's original single-student-account model, so Section 7's `students` table and auth design should be read together with this section.

- **Student** — can read/update only their own row: their own scores, risk status, and the Lecture2Revision portal for their own flagged subjects. No visibility into other students.
- **Professor** — read access to all students within their department (only CE exists for now, per Section 2's gating; the role model already supports more).
- **Head of Department (HOD)** — admin for their own department: everything a Professor sees, plus department-level management.
- **Principal** — admin across *all* departments, including ones not yet live (Section 2's branch gating).

This needs a `profiles` table distinct from `students` (a professor/HOD/principal is not a student record): `profiles` (`id` UUID → `auth.users.id`, `role` enum [`student`, `professor`, `hod`, `principal`], `department` — null/all for principal, set for professor/HOD, matching the student's own `branch` for a student-linked profile). RLS policies key off `profiles.role` and `profiles.department` rather than assuming every authenticated user is a student.

Only the **Student** tier is needed for the hackathon demo's registration flow (Section 2); Professor/HOD/Principal accounts and their dashboards are a later phase, but the schema (Section 7) should be designed for all four from the start so adding them later is additive, not a migration.

---

## 9. Out of scope / future work

- Branches other than CE, years other than TY (UI already gates for this; just flip the disabled flags later).
- Google Classroom sync (mentor's point 8).
- Google OAuth itself (needs the user's own Google Cloud credentials — documented, not auto-configured).
- Any backend other than `offline_extractive` being the default revision-generation path (Gemini etc. stay optional).
- Professor/HOD/Principal dashboards and workflows (Section 8 defines the access model now; the UI for these roles is future work).

---

## 10. Testing

- Pytest suite carried over/extended from Lecture2Revision for the untouched modules (SM-2, extractive generator, LSTM, adaptive engine).
- New tests for: synthetic data generator's risk-rate bounds (10-15%), the no-leakage checkpoint construction, the NumPy CNN forward pass matching the Keras model's output within tolerance, and the risk-boost term in `calculate_topic_priority`.
- Model evaluation report (CV metrics, baseline comparison, SHAP examples) generated as part of the training script, not hand-written.
