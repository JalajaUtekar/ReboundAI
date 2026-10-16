# Rebound AI — Build Roadmap

Read this first when resuming work, especially in a fresh session with no memory of prior conversations. It lists what's done, what's next, and a self-contained prompt for each remaining step — paste the prompt as-is to pick up exactly where things left off.

Source of truth for all design decisions: [docs/superpowers/specs/2026-10-16-rebound-ai-design.md](superpowers/specs/2026-10-16-rebound-ai-design.md). Every step below should be read against that spec; if anything here and the spec disagree, the spec wins (fix this file, don't fix the spec).

---

## Done

- **Spec** — `docs/superpowers/specs/2026-10-16-rebound-ai-design.md`. Full design: registration flow, subject/grading structure, ScholarTrack's leakage-free forecasting design, the mentor's 11-point ML methodology, Lecture2Revision's gated-assistant role, the Next.js + Python-on-Vercel + Supabase stack.
- **Plan 1 — Synthetic dataset generator** — `docs/superpowers/plans/2026-10-17-synthetic-dataset-generator.md`. Implemented via subagent-driven-development, reviewed, merged to `main`.
  - Code: `scripts/data_generation/subjects.py`, `scripts/data_generation/generator.py`, `scripts/generate_synthetic_data.py`
  - Output: `data/synthetic_students.json` — 75 TY Computer Engineering students, 6 subjects each (DSML/CN/SPOS fixed + chosen MDM/OEL/Elective), 4 genuinely at-risk (gradual decline, cumulative-mean attendance, no ETE leakage into features).
  - Tests: `tests/data_generation/` — 34 passing.
  - Key interfaces later steps depend on: `generate_dataset(num_students=75, seed=42) -> list[dict]` and the per-subject record shape — see `scripts/data_generation/generator.py:188-204` for the exact dict shape, and `CHECKPOINTS_IN_TIME_ORDER` (`generator.py:119-122`) for the 7-checkpoint chronological order: `[("submission",0,wk2), ("submission",1,wk4), ("submission",2,wk6), ("unit_test",0,wk6), ("submission",3,wk8), ("submission",4,wk10), ("unit_test",1,wk12)]`.

---

## Remaining steps, in order

Each step is its own brainstorm → spec-addition (if needed) → plan → subagent-driven-development cycle, same process as Plan 1. Don't skip straight to code — follow the process each time.

### Step 2: ScholarTrack feature engineering (leakage-free training rows)

**Goal:** turn `data/synthetic_students.json`'s full-term records into the actual training rows ScholarTrack's CNN will train on — per spec §4/§5, this means building rows at multiple "as-of checkpoint" prefixes (not the completed record), so the model learns to forecast from partial-term data, never from data that already contains its own answer.

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Section 5 (ScholarTrack modeling pipeline) and `scripts/data_generation/generator.py` (especially `CHECKPOINTS_IN_TIME_ORDER` and the shape `generate_subject_instance` returns, including the nested `"outcome"` key). Use the `superpowers:brainstorming` skill (this is bounded — the flow already exists in the spec, so a short in-chat design should suffice, escalate to full architectural treatment only if you find it isn't bounded after all) to design the feature-engineering step: for each student-subject record in `data/synthetic_students.json`, build one training row per "as-of" checkpoint prefix (e.g. after checkpoint 2, 3, 4, 5, 6, 7 of the 7 in `CHECKPOINTS_IN_TIME_ORDER`), each row holding the 5-feature vector `[internal_marks_pct, assignment_marks_pct, attendance_pct, on_time_submission_ratio, avg_days_late]` computed ONLY from data available as of that checkpoint (cumulative unit-test/submission scores so far, cumulative-mean attendance through that checkpoint's week, on-time ratio and average lateness of submissions so far), labeled with the subject's actual final `outcome.at_risk` — the model never sees data past the prefix it's given. Then use `superpowers:writing-plans` to write the implementation plan (TDD, bite-sized tasks, save to `docs/superpowers/plans/`), then execute it with `superpowers:subagent-driven-development`.

---

### Step 3: ScholarTrack model training & evaluation

**Goal:** the actual CNN training pipeline per spec §5's 11-point methodology — class weights, stratified k-fold/leave-one-out CV (not a single split, given the small N), train-fold-only oversampling, optional focal loss, feature standardization, Logistic Regression + Gradient Boosting baselines compared against the Conv1D model under identical CV, custom recall-targeted cutoff tuning, SHAP explainability, and imbalance-appropriate metrics (recall/precision/F1/PR-AUC/confusion matrix, never bare accuracy).

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Section 5 in full, and the training-rows output from Step 2 (check `docs/ROADMAP.md` for where that step left its output file and schema — update this prompt with the exact path once Step 2 is done). Use `superpowers:brainstorming` (architectural — this is a new subsystem with real design choices: CV strategy, oversampling library, focal-loss implementation, SHAP integration) to design the training pipeline implementing every point in spec §5's mentor checklist. Confirm the design, write it as a spec addendum or new spec file, then `superpowers:writing-plans` for the implementation plan, then `superpowers:subagent-driven-development` to execute. Flag early: decide whether training code lives as a one-off script/notebook (consistent with "basic, not epic") or needs to be importable by Step 4's export step — it needs to be, so design the module boundary with that in mind from the start.

---

### Step 4: NumPy export for serverless inference

**Goal:** per spec §5, the CNN is trained with Keras but served via a pure-NumPy forward pass (mirroring Lecture2Revision's existing `NumPyLSTMModel` pattern in the original `lstm_model.py`), so the production serverless function never needs TensorFlow installed. Verify the NumPy forward pass numerically matches the Keras model's output within tolerance.

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Section 5's architecture paragraph, and look at how `Lecture2Revision/lstm_model.py`'s `NumPyLSTMModel` class implements this same pattern (train with Keras, export weights, reimplement the forward pass in plain NumPy) for a concrete reference to follow. Use `superpowers:brainstorming` (bounded — the pattern to follow already exists in the codebase) to confirm the approach, then `superpowers:writing-plans` and `superpowers:subagent-driven-development` to build a `NumPyConv1DModel` (or similarly named) class for ScholarTrack, with a test asserting its output matches the trained Keras model's predictions within a small numeric tolerance on the real training data from Step 2/3.

---

### Step 5: Supabase schema + auth

**Goal:** stand up the actual Supabase project — Postgres schema per spec §7 (`students`, `student_subjects`, `assessment_checkpoints`, `risk_predictions`, plus Lecture2Revision's carried-over tables), and Supabase Auth with the demo account. Google OAuth setup is documented for later, not auto-configured (needs the user's own Google Cloud credentials).

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Section 7 (Architecture & stack) for the exact table list and auth plan. Use `superpowers:brainstorming` (bounded — the schema is already specified) to confirm the exact column-level schema for each table (cross-reference `scripts/data_generation/generator.py`'s output shape and the original `Lecture2Revision/database.py`'s `CREATE TABLE` statements for the tables being carried over, Postgres-ifying types as needed). Write the schema as SQL migration file(s) under `supabase/migrations/`, a seed script for the demo account (email/password only — ask the user to confirm credentials rather than reusing anything from old chat history), and a short `docs/SUPABASE_SETUP.md` walking through: creating the Supabase project, running the migrations, where the connection string/env vars go, and the step-by-step for enabling Google OAuth later (Authentication → Providers → Google in the Supabase dashboard, needs the user's own Google Cloud OAuth client). Use `superpowers:writing-plans` + `superpowers:subagent-driven-development` for the implementation.

---

### Step 6: Backend API (Python serverless functions)

**Goal:** `/api/*.py` Vercel Python functions wrapping — largely unchanged — Lecture2Revision's `lecture_processor.py` (PyMuPDF), `transformer_service.py`, `flashcard_service.py`, `lstm_model.py`, and `adaptive_engine.py` (plus the new risk-boost term), the ScholarTrack NumPy inference from Step 4, and `database.py` Postgres-ified per Step 5's schema.

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Sections 6 and 7. The source modules to port (mostly unchanged, per spec §6) are in `Lecture2Revision/` at the repo root — read `lecture_processor.py`, `transformer_service.py`, `flashcard_service.py`, `lstm_model.py`, `adaptive_engine.py`, and `database.py` before starting. Use `superpowers:brainstorming` (architectural — this wires together everything built so far into one API surface) to design the `/api/*.py` endpoint structure (what each route does, request/response shapes, how a subject's ScholarTrack risk score feeds into `adaptive_engine.calculate_topic_priority`'s new urgency-boost term, and the activation rule: Lecture2Revision's UI/endpoints only matter for a student-subject ScholarTrack has flagged). Then `superpowers:writing-plans` + `superpowers:subagent-driven-development`.

---

### Step 7: Next.js frontend

**Goal:** the actual "elegant educational UI" — registration flow (Section 2: basic details → branch gating → year gating → subject setup), the ScholarTrack-first dashboard with MMCOE branding (`assets/mmcoe-logo.png`), and the gated Lecture2Revision workflow for flagged subjects only.

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Sections 1-3 and 7 for the product framing, registration flow, and branding requirements. Use `superpowers:brainstorming` (architectural — new frontend, real page/component design decisions) to design the Next.js App Router page structure and component boundaries: registration (4 steps per spec §2), the ScholarTrack-first dashboard, per-subject risk detail, and the gated Lecture2Revision pages (upload lecture, revision notes, quiz, flashcards, progress, revision plan) that only appear for a flagged subject. Confirm design, then `superpowers:writing-plans` + `superpowers:subagent-driven-development`. This step depends on Step 6's API being up, at least with stub/mock responses if the backend isn't fully done yet — flag that dependency explicitly if picking this up out of order.

---

## If you're not sure where things stand

Run `git log --oneline` and check which plan files under `docs/superpowers/plans/` have corresponding merged commits — a plan with no matching recent commits hasn't been executed yet. Check `data/` and `supabase/` for which output artifacts already exist. When in doubt, re-read the spec section for the step you're about to start before writing any code.
