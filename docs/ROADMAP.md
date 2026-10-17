# Rebound AI — Build Roadmap

Read this first when resuming work, especially in a fresh session with no memory of prior conversations. It lists what's done, what's next, and a self-contained prompt for each remaining step — paste the prompt as-is to pick up exactly where things left off.

Source of truth for all design decisions: [docs/superpowers/specs/2026-10-16-rebound-ai-design.md](superpowers/specs/2026-10-16-rebound-ai-design.md). Every step below should be read against that spec; if anything here and the spec disagree, the spec wins (fix this file, don't fix the spec).

---

## Target goal

A deployed, demoable Rebound AI for the CE24MDM305 Foundations of GenAI mini-hackathon, with **ScholarTrack as the primary problem statement and Lecture2Revision as its gated assistant** (spec §1):

- **ScholarTrack actually forecasts, not diagnoses.** A trained CNN (Conv1D, matching the course's CNN unit) is always trained and always appears in the benchmark table (a course requirement), compared against Logistic Regression and Gradient Boosting baselines under proper cross-validation (not a single split, given the small N) — but **the model that gets deployed is whichever of the three wins the CV benchmark**, not pre-committed to the CNN. All three predict a TY Computer Engineering student's per-subject risk from in-term signals only — marks, attendance, submission timeliness — never from the held-out end-term exam. Class imbalance is handled per the mentor's checklist (class weights, train-fold-only group-aware oversampling, a recall-targeted cutoff), and each flagged student is explainable via SHAP ("why was this student flagged"). The training target is the two-tier label from spec §4: **"at risk of falling behind"** (aggregate < 70% OR attendance < 75%) is what the model trains on; **"failing"** (aggregate < 40%) is a separate, more severe, informational-only fact, never the training target.
- **Lecture2Revision activates only for a flagged student-subject** — real PDF upload, grounded revision notes/flashcards/quizzes (offline-extractive by default, not a bare LLM call), SM-2 spaced repetition, and the existing LSTM forecaster boosting that subject's priority in the adaptive plan.
- **Registration and data are real for the TY CE cohort** (spec §2-§3): branch/year gating, the actual DSML/CN/SPOS + MDM/OEL/Elective subject structure and grading weights, backed by Supabase (Postgres + Auth, Google OAuth documented for later). The cohort is **120 students**, 10-15% at risk (~12-18 students) — large enough to support proper 5-fold grouped cross-validation.
- **Role-based access is schema-ready for all four tiers, built for one.** Spec §8 defines Student / Professor / HOD / Principal via a `profiles` table and Supabase RLS. Only the **Student** tier's registration, dashboard, and Lecture2Revision gating are built for the hackathon demo; the schema is designed so Professor/HOD/Principal dashboards (spec §9, future work) are additive later, not a migration.
- **One deployable app on Vercel** — Next.js frontend + Python serverless API, MMCOE-branded, not a loose collection of scripts and notebooks.
- **Honest throughout development**: every intermediate build (like the current frontend) shows real data and is honestly labeled as such — never a mocked demo pretending to be further along than it is.

No fixed deadline has been given — treat "done" as the state above, not a specific date.

---

## Done

- **Spec** — `docs/superpowers/specs/2026-10-16-rebound-ai-design.md`. Full design: registration flow, subject/grading structure, ScholarTrack's leakage-free forecasting design, the mentor's 11-point ML methodology, Lecture2Revision's gated-assistant role, the Next.js + Python-on-Vercel + Supabase stack.
- **Plan 1 — Synthetic dataset generator** — `docs/superpowers/plans/2026-10-17-synthetic-dataset-generator.md`. Implemented via subagent-driven-development, reviewed, merged to `main`.
  - Code: `scripts/data_generation/subjects.py`, `scripts/data_generation/generator.py`, `scripts/generate_synthetic_data.py`
  - Output: `data/synthetic_students.json` — 75 TY Computer Engineering students, 6 subjects each (DSML/CN/SPOS fixed + chosen MDM/OEL/Elective), 4 genuinely at-risk (gradual decline, cumulative-mean attendance, no ETE leakage into features).
  - Tests: `tests/data_generation/` — 34 passing.
  - Key interfaces later steps depend on: `generate_dataset(num_students=75, seed=42) -> list[dict]` and the per-subject record shape — see `scripts/data_generation/generator.py:188-204` for the exact dict shape, and `CHECKPOINTS_IN_TIME_ORDER` (`generator.py:119-122`) for the 7-checkpoint chronological order: `[("submission",0,wk2), ("submission",1,wk4), ("submission",2,wk6), ("unit_test",0,wk6), ("submission",3,wk8), ("submission",4,wk10), ("unit_test",1,wk12)]`.
  - **Regenerated** (commit `021095f`) to match the current spec: `data/synthetic_students.json` now has **120 students**, 12 at-risk (10.0%), 5 genuinely failing, two-tier `outcome.at_risk`/`outcome.is_failing` labels anchored to 70%/40% respectively. 43/43 tests passing. `web/data/synthetic_students.json` and `web/lib/dataset.ts` synced to match.
- **Step 2 — Feature engineering** (commit `b7ceb9c` extraction, `9de3bee` skew/standardizer). `scripts/feature_engineering/extraction.py` (no-leakage prefix-row extraction, structurally leakage-free — only ever indexes by prefix length) + `scripts/feature_engineering/standardizer.py` (skew inspection, `LeakageFreeStandardizer`) + `scripts/build_training_rows.py` CLI. Output: `data/training_rows.json`/`.csv` — 4,320 rows (120 students × 6 subjects × 6 valid "as-of" prefixes), 72 at_risk rows, 30 is_failing rows. 64/64 tests passing at this point.
- **Step 3 — Tri-model training & evaluation** (commits `48cf680` CV/metrics, `3545aaa` baselines, `a6a822e` CNN, `1c6e304` held-out split, `1e86d83`/`3a935d4` tuning, `d387efa` final report). Full writeup: [`models/benchmark_report.md`](../models/benchmark_report.md).
  - Built: `scripts/training/cv.py` (student-grouped CV, group-aware oversampling, held-out test split), `scripts/training/metrics.py`, `scripts/training/baselines.py` (LogReg + GBDT), `scripts/training/cnn_model.py` (Conv1D + focal loss), `scripts/training/tune_logreg.py`, `scripts/training/tune_gbdt.py`.
  - **Final winner: Logistic Regression** (C=0.1, oversample_ratio=0.5, threshold=0.90) — the only model clearing all four metrics (recall 0.833, precision 0.909, F1 0.870, PR-AUC 0.880) on the genuinely held-out 24-student test set. Gradient Boosting looked better on CV alone (overfit the tuning pool) but fell short on held-out data; the CNN's precision collapsed on held-out data despite decent recall. This CV→held-out reversal is itself documented in the report as the reason the held-out split matters.
  - 124+ tests passing at this point (check `py -3 -m pytest tests/ -v` for current count).
- **Step 4 — SHAP explainability** (commit `0bbb778`). `scripts/training/explainability.py`: `train_production_model()` fits the winning LogReg config (C=0.1, oversample_ratio=0.5) as one model on all 120 students; `generate_student_explanation()` uses `shap.LinearExplainer` to produce per-student "why flagged" output (top risk/protective factors + a human-readable summary). Verified additivity holds on real rows; real example — a known at-risk student's explanation genuinely implicates `marks_deficit_margin`, `attendance_velocity`, `avg_days_late`. 130/130 tests passing at this point.
- **Step 5a — NumPy inference** (commit `65dc286`). `scripts/inference/`: exact-parity NumPy LogReg (deployed) and GBDT, close-tolerance CNN. 150/150 tests passing at this point.
- **Step 5b — Supabase schema, auth, RLS — deployed to the live project** (commits `c3ce897` schema/RLS/seed script, `cd42296` seed-script fixes). 13 tables live at `kyhprimwjgdcjiifzqwn.supabase.co`, RLS enabled on all of them, 120 real student profiles + Supabase Auth users, 720 student_subjects, 15,120 assessment_checkpoints, 12 real SHAP-backed risk_predictions. Google OAuth configured and verified live (provider enabled, redirect URI set in Google Cloud Console). `.env` holds `SUPABASE_URL`/`SUPABASE_ANON_KEY`/`SUPABASE_SERVICE_ROLE_KEY`/`DATABASE_URL` (gitignored). Seed script usage: `py -3 supabase/seed.py --create-auth-users` (idempotent, safe to re-run).
- **Step 6 — Unified serverless backend API** (commits `0cf2527` shared auth/db lib + risk prediction, `e3c9ccc` SM-2 + adaptive schedule, `2a1b076` lecture processing). `api/_lib/{db,auth,scholartrack}.py` + four Vercel Python functions: `api/predict_risk.py`, `api/rebound/{process_lecture,sm2_review,adaptive_schedule}.py`. All gated correctly (Lecture2Revision only activates for a subject flagged "at risk," never on "failing" alone), risk-boost term wired into `calculate_topic_priority` (70/30 blend matching the existing LSTM-blend ratio), ScholarTrack risk endpoint recomputes SHAP values exactly via closed-form linear algebra (no `shap` import needed at serving time — verified bit-for-bit against the real `shap.LinearExplainer`). 212/212 tests passing, all tested against the real live Supabase database.
- **Step 7 — Production frontend** (commits `a64fc31` auth/registration, `be7db5e` dashboard, `fcbe62a` gated Rebound workspace). Real pages: `/login` (Google OAuth + email/password), `/register` (4-step flow per spec §2), `/dashboard` (per-subject two-tier risk badges, "Why flagged?" SHAP detail, CTA only for flagged subjects), `/rebound/[subjectId]` (gated — upload lecture, revision kit viewer, flashcards with real SM-2 review, quiz practice, adaptive schedule showing the risk-boost reasoning). Shared `web/lib/api.ts`/`supabaseClient.ts`/`contexts/AuthContext.tsx`. `npm run build` clean, 0 TypeScript errors, all 5 routes compile together. Known, documented simplifications: no photo upload persistence, no GET-to-refetch-a-past-kit endpoint, quiz scoring is client-only (not persisted) — none block the demo flow.
- **Basic frontend scaffold** — `web/` (Next.js App Router + TypeScript + Tailwind). Not part of a numbered plan — scaffolded directly (bounded, no brainstorm/plan ceremony) so there's something to visually check against as each step below lands. Currently a single dashboard page (`web/app/page.tsx`) reading `web/data/synthetic_students.json` (a committed copy of the root `data/synthetic_students.json`) via `web/lib/dataset.ts`, showing cohort stats and a per-student risk table with MMCOE branding (`web/public/mmcoe-logo.png`).
  - Run it: `npm run dev` from `web/` (or `preview_start` with the `rebound-ai-web` entry in `.claude/launch.json`), or ask Claude to start it.
  - Refresh its data after regenerating the dataset: `npm run sync-data` from `web/` (copies `../data/synthetic_students.json` in).
  - **As each later step lands, extend this app rather than rebuilding it**: Step 2/3's training rows and risk predictions, Step 5's Supabase connection, Step 6's API — each should replace or augment `lib/dataset.ts`'s local-file read with a real data source, and new pages/sections get added to `app/`, following Step 7's fuller design once that step is reached.

---

## Remaining steps, in order

Each step is its own brainstorm → spec-addition (if needed) → plan → subagent-driven-development cycle, same process as Plan 1. Don't skip straight to code — follow the process each time.

### Step 2: Dynamic feature engineering & prefix-row generation — ✅ DONE, see "Done" section above

**Goal:** turn the regenerated `data/synthetic_students.json`'s full-term records into the actual training rows ScholarTrack's models will train on — per spec §4/§5, this means building rows at multiple "as-of checkpoint" prefixes (not the completed record) plus the engineered velocity/kinematic features, so the model learns to forecast from partial-term data, never from data that already contains its own answer.

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Sections 4 and 5 (ScholarTrack modeling pipeline) in full, and `scripts/data_generation/generator.py` (especially `CHECKPOINTS_IN_TIME_ORDER` and the shape `generate_subject_instance` returns, including the nested `"outcome"` key — confirm this still matches the regenerated 120-student dataset before relying on exact line numbers). Use the `superpowers:brainstorming` skill (this is bounded — the flow already exists in the spec, so a short in-chat design should suffice, escalate to full architectural treatment only if you find it isn't bounded after all) to design the feature-engineering step in `scripts/feature_engineering/`:
>
> 1. For each student-subject record, build one training row per "as-of" checkpoint prefix — checkpoints 2 through 7 of the 7 in `CHECKPOINTS_IN_TIME_ORDER` (checkpoint 1 alone is too little signal to train on) — each row holding the 5-feature vector `[internal_marks_pct, assignment_marks_pct, attendance_pct, on_time_submission_ratio, avg_days_late]` computed ONLY from data available as of that checkpoint. At 120 students × 6 subjects × 6 valid prefixes, expect **4,320 training rows** (not the old 75-student math's 450 × 6 — recompute and sanity-check the actual row count against the regenerated dataset rather than assuming this number).
> 2. Add the engineered velocity/kinematic features from spec §4: `attendance_velocity` (slope over trailing checkpoints), `marks_velocity` (delta between latest and preceding marks checkpoint), `delay_acceleration` (change in `avg_days_late` between recent submissions), `attendance_deficit_margin` (`attendance_pct - 75.0`), and **`marks_deficit_margin` anchored to `aggregate_pct_so_far - 70.0`** — the risk threshold, **not** `- 40.0` (40% is the separate "failing" threshold, informational only, never the margin used for risk features).
> 3. Label each row with the subject's actual final **"at risk of falling behind"** outcome (aggregate < 70% OR attendance < 75%) — not the "failing" (< 40%) fact, which is a separate informational field you may also carry through but must never train on. The model never sees data past the prefix it's given.
> 4. Include a feature-skew inspection + `LeakageFreeStandardizer` (fits strictly on training folds) as prep for Step 3, per spec §5 point 7.
>
> Then use `superpowers:writing-plans` to write the implementation plan (TDD, bite-sized tasks, save to `docs/superpowers/plans/`), then execute it with `superpowers:subagent-driven-development`.

---

### Step 3: Tri-model training, imbalance handling & evaluation — ✅ DONE, see "Done" section above and `models/benchmark_report.md`

**Goal:** the actual CNN + Gradient Boosting + Logistic Regression training pipeline per spec §5's 11-point methodology — class weights, stratified student-grouped 5-fold CV, train-fold-only group-aware oversampling, optional focal loss, feature standardization, all three models compared under identical CV, custom recall-targeted cutoff tuning, and imbalance-appropriate metrics (recall/precision/F1/PR-AUC/confusion matrix, never bare accuracy).

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Section 5 in full, and the training-rows output from Step 2 (check `docs/ROADMAP.md` for where that step left its output file and schema — update this prompt with the exact path once Step 2 is done). Use `superpowers:brainstorming` (architectural — this is a new subsystem with real design choices: oversampling library, focal-loss implementation) to design the training pipeline in `scripts/training/` implementing every point in spec §5's mentor checklist, in particular:
>
> 1. **CV strategy**: stratified **GroupKFold, 5 folds, grouped by `student_id`** (spec §5 point 3). This is now safely resolved — the old 75-student dataset only had 4 at-risk students, which cannot be stratified across 5 folds, but the regenerated 120-student / 10-15% cohort has ~12-18 at-risk students, comfortably supporting 5 folds. Note this resolution explicitly in the design writeup rather than silently assuming 5 folds works.
> 2. **Group-aware oversampling** (spec §5 point 4): a positive student-subject's 6 prefix rows are near-duplicates of the same trajectory, so naive row-level SMOTE mostly interpolates within one student's own decline rather than adding real diversity. Oversample at the **student-subject level** — duplicate/jitter a whole trajectory with noise — not independent prefix-rows. Applied to training folds only; validation/test folds stay on the real distribution (spec §5 point 5).
> 3. **All three models** trained and benchmarked under identical CV: the Conv1D CNN (matching the course's CNN unit — it always trains and always appears in the benchmark table, this is a course requirement), Gradient Boosting, and Logistic Regression.
> 4. Recall-targeted cutoff tuning (≥88-92%), optional focal loss, and metrics (recall/precision/F1/PR-AUC/confusion matrix — never bare accuracy).
> 5. **Deployment decision is firm, not hedged**: whichever of the three models wins the CV benchmark is the one that gets exported for serving in Step 5 — record the winner's identity and metrics clearly in the training script's output, since Step 5 needs to know which model type(s) to build a NumPy serving path for (spec §5's architecture paragraph requires all three paths to exist regardless, but only the winner actually gets deployed).
>
> Confirm the design, write it as a spec addendum or new spec file, then `superpowers:writing-plans` for the implementation plan, then `superpowers:subagent-driven-development` to execute. Flag early: decide whether training code lives as a one-off script/notebook or needs to be importable by Step 4/5's export step — it needs to be, so design the module boundary with that in mind from the start.

---

### Step 4: SHAP explainability — ✅ DONE

**Goal:** per spec §5 point 9, every flagged student comes with *why* they were flagged (e.g. "attendance trend −32%, 3 late TW submissions"), using SHAP's model-agnostic explainers so it works regardless of which model type won Step 3's benchmark.

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Section 5 point 9, and `models/benchmark_report.md` for Step 3's full result. **The winner is Logistic Regression** (`scripts/training/baselines.py::train_and_predict_oof_logreg`, config C=0.1/oversample_ratio=0.5/threshold=0.90 — see the report for why it beat Gradient Boosting's better-looking CV numbers once evaluated on the genuinely held-out test set). Use `superpowers:brainstorming` (bounded — SHAP's role is already specified and the model is now known) to design a `generate_student_explanation()` function in `scripts/training/explainability.py` that:
>
> 1. Uses `shap.LinearExplainer` (or `shap.Explainer`, which auto-selects appropriately for linear models) against the winning Logistic Regression model — simplest, most interpretable case; LogReg coefficients plus SHAP give very clean "why" explanations.
> 2. Returns structured output: predicted risk probability, is-flagged boolean (recall-tuned cutoff from Step 3), top risk factors (positive SHAP values) and protective factors (negative SHAP values), and a human-readable summary string for faculty/students.
> 3. Verify SHAP works directly against whichever serving path Step 5 exposes (Keras or the NumPy forward pass) — spec §5 point 9 notes this is a deliberate non-conflict.
>
> Then `superpowers:writing-plans` + `superpowers:subagent-driven-development`, with tests asserting valid output keys, non-empty text, and SHAP attributions summing approximately to the model's output margin.

---

### Step 5a: Pure-NumPy inference (all three model types) — ✅ DONE

(commit `65dc286`) `scripts/inference/numpy_logreg.py` (deployed — exact parity, 0.023ms/call), `numpy_gbdt.py` (full ensemble traversal, exact parity), `numpy_cnn.py` (parity within float tolerance). Exported weights: `scripts/inference/numpy_logreg_exported.npz`. 150/150 tests passing at this point.

### Step 5b: Supabase schema, auth & RLS — ✅ DONE and deployed to the live project

**Goal:** two tightly-coupled deliverables — (a) per spec §5's architecture paragraph, a lightweight NumPy/plain-Python serving path for **all three** model types (CNN forward pass, Logistic Regression dot product, Gradient Boosting decision-tree traversal), since deployment isn't pre-committed to the CNN; and (b) the Supabase Postgres schema per spec §7, now including the `profiles`/role table and RLS policies from spec §8, plus Auth with the demo account.

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Sections 5 (architecture paragraph), 7, and 8 in full. This step has two parts — treat them as sub-steps 5a/5b if that's cleaner to execute separately, but both are needed before Step 6.
>
> **5a — NumPy/lightweight inference for all three model types.** Look at how `Lecture2Revision/lstm_model.py`'s `NumPyLSTMModel` class implements the pattern (train with Keras, export weights, reimplement the forward pass in plain NumPy) as a concrete reference. Build, in `scripts/inference/`:
> 1. A CNN forward pass re-implemented in pure NumPy (Conv1D sliding-window dot products, ReLU, MaxPool1D, GlobalAveragePooling, Dense+sigmoid) mirroring the `NumPyLSTMModel` pattern.
> 2. Logistic Regression inference — trivial, it's already a dot product plus sigmoid; implement it anyway so the serving code doesn't special-case "if logreg won, import sklearn."
> 3. Gradient Boosting inference — a plain-Python/NumPy traversal of the learned decision-tree splits (no LightGBM/sklearn runtime needed in production).
> 4. A numerical parity test for the winning model type (per Step 3's recorded result) asserting the NumPy output matches the trained model's output within tolerance (`np.allclose(..., atol=1e-5)`) on real Step 2 data.
>
> **5b — Supabase schema, auth, RLS.** Cross-reference `scripts/data_generation/generator.py`'s output shape and `Lecture2Revision/database.py`'s `CREATE TABLE` statements for carried-over tables, Postgres-ifying types as needed. Write SQL migration file(s) under `supabase/migrations/` covering:
> 1. `students`, `student_subjects`, `assessment_checkpoints`, `risk_predictions` (spec §7).
> 2. **`profiles`** (`id` UUID → `auth.users.id`, `role` enum [`student`, `professor`, `hod`, `principal`], `department` — null/all for principal, set for professor/HOD, matching the student's own `branch` for a student-linked profile) and its RLS policies, per spec §8. Build the schema for all four roles now even though only the Student tier's UI/flows are in scope for the hackathon demo (spec §8's own scoping note, spec §9) — this makes adding Professor/HOD/Principal dashboards later additive, not a migration.
> 3. Lecture2Revision's carried-over tables (`lectures`, `learning_materials`, `flashcards`, `flashcard_reviews`, `quiz_questions`, `quiz_attempts`, `quiz_responses`, `revision_schedules`), Postgres-ified.
> 4. RLS enabled on all tables, keyed off `profiles.role`/`profiles.department` (not assuming every authenticated user is a student) — Student reads/updates only their own row; Professor reads all students in their department; HOD/Principal get admin scopes per spec §8 (even though their UIs aren't built yet, the RLS policies should exist).
> 5. A seed script for the demo account (Student tier, email/password — ask the user to confirm credentials rather than reusing anything from old chat history) and a short `docs/SUPABASE_SETUP.md` covering project creation, running migrations, env vars, and enabling Google OAuth later.
>
> Use `superpowers:brainstorming` (bounded — both schemas are already specified) to confirm exact column-level details for each part, then `superpowers:writing-plans` + `superpowers:subagent-driven-development` for the implementation.

---

### Step 6: Unified serverless backend API — ✅ DONE

**Goal:** `/api/*.py` Vercel Python functions wrapping — largely unchanged — Lecture2Revision's `lecture_processor.py` (PyMuPDF), `transformer_service.py`, `flashcard_service.py`, `lstm_model.py`, and `adaptive_engine.py` (plus the new risk-boost term), Step 5's NumPy ScholarTrack inference for whichever model type won, and `database.py` Postgres-ified per Step 5's schema — gated on the two-tier label (student sees Lecture2Revision only for a subject flagged "at risk of falling behind," never on "failing" alone, since failing is informational-only and every failing case is already at-risk anyway).

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Sections 5, 6, 7, and 8. The source modules to port (mostly unchanged, per spec §6) are in `Lecture2Revision/` at the repo root — read `lecture_processor.py`, `transformer_service.py`, `flashcard_service.py`, `lstm_model.py`, `adaptive_engine.py`, and `database.py` before starting. Check Step 3/5's recorded result for which model type (CNN / Gradient Boosting / Logistic Regression) actually won the benchmark and is being served — the prediction endpoint should call that model's NumPy inference path specifically, not assume the CNN. Use `superpowers:brainstorming` (architectural — this wires together everything built so far into one API surface) to design the `/api/*.py` endpoint structure: what each route does, request/response shapes, how a subject's ScholarTrack risk probability feeds into `adaptive_engine.calculate_topic_priority`'s new urgency-boost term, and the activation rule — Lecture2Revision's UI/endpoints only activate for a student-subject flagged **"at risk of falling behind"** (the two-tier label's training target, spec §4), never gated on the separate "failing" fact. Auth/RLS checks should use the `profiles` table from Step 5, scoped to the Student tier for now. Then `superpowers:writing-plans` + `superpowers:subagent-driven-development`.

---

### Step 7: Production Next.js dashboard & gated UI (Student tier only) — ✅ DONE

**Goal:** the actual "elegant educational UI" — registration flow (spec §2: basic details → branch gating → year gating → subject setup), the ScholarTrack-first dashboard with MMCOE branding (`assets/mmcoe-logo.png`), and the gated Lecture2Revision workflow for flagged subjects only — **Student tier only**; Professor/HOD/Principal dashboards are explicitly future work (spec §9) and are out of scope for this step.

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Sections 1-3, 7, 8, and 9 for the product framing, registration flow, branding requirements, and role scoping. Use `superpowers:brainstorming` (architectural — new frontend, real page/component design decisions) to design the Next.js App Router page structure and component boundaries for the **Student-tier UI only**:
>
> 1. Registration (4 steps per spec §2): basic details, branch gating (CE only selectable), year gating (TY only selectable), subject setup (DSML/CN/SPOS fixed + MDM/OEL/Elective dropdowns).
> 2. The ScholarTrack-first dashboard: cohort/own-student risk display, per-subject risk detail using the two-tier label (show "at risk of falling behind" as the primary status; "failing" as a separate, more severe informational badge, never conflated with the primary flag), and SHAP "why flagged" detail from Step 4.
> 3. The gated Lecture2Revision pages (upload lecture, revision notes, quiz, flashcards, progress, revision plan) that only appear for a subject flagged at-risk.
>
> **Do not build Professor/HOD/Principal dashboard pages or routes** — spec §9 defers these explicitly; the RLS/schema groundwork from Step 5 is sufficient preparation, and building their UI now would be scope creep ahead of the hackathon demo's actual requirement (spec §8's own scoping note). Confirm design, then `superpowers:writing-plans` + `superpowers:subagent-driven-development`. This step depends on Step 6's API being up, at least with stub/mock responses if the backend isn't fully done yet — flag that dependency explicitly if picking this up out of order.

---

### Step 8: Google Classroom integration (documented, future work)

**Goal:** per spec §9/§11 point 11, document the architecture for a future Google Classroom sync — not built now, just specified so it's ready to pick up later. create a txt file with the architecture, justifying this future scope

**Resume prompt:**
> Read `docs/superpowers/specs/2026-10-16-rebound-ai-design.md` Section 9 (Out of scope / future work) and Section 5 point 11. Write an architectural design and code specification in `docs/GOOGLE_CLASSROOM_INTEGRATION.md` covering: OAuth scopes needed (`classroom.courses.readonly`, `classroom.coursework.students.readonly`, `classroom.student-submissions.students.readonly`), a data-sync pipeline mapping Google Classroom coursework to the `assessment_checkpoints` table from Step 5 (including how `submission.creationTime` vs. `courseWork.dueDate` maps to `avg_days_late`/`delay_acceleration`), a graceful fallback (manual CSV gradebook upload stays available alongside automated sync), and data-privacy/token-storage notes. This is documentation only — no code, no `superpowers:writing-plans`/`subagent-driven-development` cycle needed, just a clear spec a future session can implement from.

---

## If you're not sure where things stand

Run `git log --oneline` and check which plan files under `docs/superpowers/plans/` have corresponding merged commits — a plan with no matching recent commits hasn't been executed yet. Check `data/` and `supabase/` for which output artifacts already exist. When in doubt, re-read the spec section for the step you're about to start before writing any code.
