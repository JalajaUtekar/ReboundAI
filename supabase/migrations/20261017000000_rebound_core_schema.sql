-- Rebound AI -- core schema
-- Section 7/8 of docs/superpowers/specs/2026-10-16-rebound-ai-design.md.
--
-- Two families of tables:
--   1. ScholarTrack  -- profiles, students, student_subjects,
--      assessment_checkpoints, risk_predictions.
--   2. Lecture2Revision, Postgres-ified from Lecture2Revision/database.py's
--      SQLite schema and re-linked to `student_subjects` (a real FK) instead
--      of the original loose `student_id` TEXT + `topic` TEXT matching.
--
-- All primary keys are UUID (gen_random_uuid(), from pgcrypto) for
-- consistency across both families -- the original SQLite
-- INTEGER PRIMARY KEY AUTOINCREMENT ids are not carried over.
--
-- RLS is enabled and policied in the companion migration
-- 20261017000001_rebound_rls_policies.sql. This file only defines shape.

create extension if not exists pgcrypto;

-- ============================================================================
-- ScholarTrack tables
-- ============================================================================

create type public.app_role as enum ('student', 'professor', 'hod', 'principal');

-- One row per authenticated user (student, professor, HOD, or principal).
-- A student profile additionally has a matching row in `students`.
create table public.profiles (
    id          uuid primary key references auth.users (id) on delete cascade,
    role        public.app_role not null,
    -- NULL for principal (all departments). Set for professor/hod/student-linked
    -- profiles, using the same branch vocabulary as students.branch.
    department  text,
    created_at  timestamptz not null default now(),
    constraint profiles_department_branch_check
        check (department is null or department in ('AI&DS', 'CE', 'IT', 'E&TC', 'ME', 'EE'))
);

comment on table public.profiles is
    'RBAC identity row for every authenticated user. role/department drive RLS; see spec section 8.';

-- A student IS a profile with role = 'student'; this table holds the
-- student-specific academic fields that a professor/HOD/principal profile
-- does not have.
create table public.students (
    id              uuid primary key default gen_random_uuid(),
    profile_id      uuid not null unique references public.profiles (id) on delete cascade,
    name            text not null,
    prn             text not null unique,
    email           text not null unique,
    photo_url       text,
    branch          text not null,
    academic_year   text not null,
    created_at      timestamptz not null default now(),
    constraint students_branch_check
        check (branch in ('AI&DS', 'CE', 'IT', 'E&TC', 'ME', 'EE')),
    constraint students_academic_year_check
        check (academic_year in ('FY', 'SY', 'TY', 'BE'))
);

create index students_profile_id_idx on public.students (profile_id);
create index students_branch_idx on public.students (branch);

comment on table public.students is
    'Student academic record. One row per student, one-to-one with a profiles row (role=student).';

-- One row per student per subject (6 rows per TY CE student today: 3 fixed +
-- chosen MDM/OEL/Elective).
create table public.student_subjects (
    id              uuid primary key default gen_random_uuid(),
    student_id      uuid not null references public.students (id) on delete cascade,
    subject_slug    text not null,
    subject_name    text not null,
    subject_type    text not null,
    -- e.g. {"TH": 40, "ETE": 60, "PR": 25} or {"TH": 40, "ETE": 60, "TW": 25}
    components      jsonb not null,
    created_at      timestamptz not null default now(),
    constraint student_subjects_type_check
        check (subject_type in ('fixed', 'mdm', 'oel', 'elective')),
    constraint student_subjects_student_slug_unique unique (student_id, subject_slug)
);

create index student_subjects_student_id_idx on public.student_subjects (student_id);
create index student_subjects_subject_slug_idx on public.student_subjects (subject_slug);

comment on table public.student_subjects is
    'One row per student-subject enrollment. Carries the grading component weights for that subject.';

-- Per-checkpoint raw data feeding the ScholarTrack feature pipeline
-- (scripts/feature_engineering/extraction.py). Mirrors generator.py's
-- unit_tests / submissions / attendance_weeks records.
create table public.assessment_checkpoints (
    id                  uuid primary key default gen_random_uuid(),
    student_subject_id  uuid not null references public.student_subjects (id) on delete cascade,
    checkpoint_type     text not null,
    checkpoint_index    integer not null,
    week_number         integer not null,
    -- NULL for attendance_week rows (they carry attendance_pct instead).
    score               double precision,
    max_marks           double precision,
    -- Only populated for attendance_week rows.
    attendance_pct      double precision,
    -- Only populated for unit_test/submission rows.
    due_day             integer,
    submitted_day       integer,
    created_at          timestamptz not null default now(),
    constraint assessment_checkpoints_type_check
        check (checkpoint_type in ('unit_test', 'submission', 'attendance_week'))
);

create index assessment_checkpoints_student_subject_id_idx
    on public.assessment_checkpoints (student_subject_id);
create index assessment_checkpoints_type_idx
    on public.assessment_checkpoints (student_subject_id, checkpoint_type, checkpoint_index);
create index assessment_checkpoints_week_idx
    on public.assessment_checkpoints (student_subject_id, week_number);

comment on table public.assessment_checkpoints is
    'Raw per-checkpoint marks/submission/attendance data. ETE is never stored here (spec section 3/4): it is held out of every input feature.';

-- ScholarTrack's model output per student-subject per as-of checkpoint.
-- checkpoint_idx mirrors scripts/feature_engineering/extraction.py's
-- prefix_len (2..7) -- the "as of checkpoint N" row the prediction was made at.
create table public.risk_predictions (
    id                      uuid primary key default gen_random_uuid(),
    student_subject_id      uuid not null references public.student_subjects (id) on delete cascade,
    checkpoint_index        integer not null,
    risk_probability        double precision not null,
    -- The 0.90-threshold ("at risk of falling behind") flag -- the real
    -- training target per spec section 4's two-tier label.
    is_flagged              boolean not null,
    -- The separate, more severe "failing" (<40% aggregate) informational
    -- fact. Never the training target, always displayed alongside it.
    is_failing              boolean not null,
    top_risk_factors        jsonb,
    protective_factors      jsonb,
    human_readable_summary  text,
    -- e.g. "logreg_v1" -- traceable across future retraining rounds
    -- (models/benchmark_report.md's winning config).
    model_version           text not null,
    predicted_at            timestamptz not null default now(),
    constraint risk_predictions_probability_check
        check (risk_probability >= 0 and risk_probability <= 1)
);

create index risk_predictions_student_subject_id_idx
    on public.risk_predictions (student_subject_id);
create index risk_predictions_flagged_idx
    on public.risk_predictions (student_subject_id, is_flagged);
create index risk_predictions_predicted_at_idx
    on public.risk_predictions (predicted_at desc);

comment on table public.risk_predictions is
    'ScholarTrack risk-model output (spec section 5/9): probability, two-tier flags, and SHAP-derived explanation.';

-- ============================================================================
-- Lecture2Revision tables (Postgres-ified, gated to flagged subjects at the
-- application layer -- see spec section 6; the schema itself does not
-- enforce the gating).
-- ============================================================================

-- A lecture upload for one student-subject (the gating unit); replaces the
-- original loose student_id+topic text matching with a real FK.
create table public.lectures (
    id                  uuid primary key default gen_random_uuid(),
    student_subject_id  uuid not null references public.student_subjects (id) on delete cascade,
    title               text not null,
    topic               text not null,
    filename            text not null,
    file_path           text not null,
    raw_text            text,
    cleaned_text        text,
    word_count          integer not null default 0,
    chunk_count         integer not null default 1,
    created_at          timestamptz not null default now()
);

create index lectures_student_subject_id_idx on public.lectures (student_subject_id);
create index lectures_topic_idx on public.lectures (student_subject_id, topic);

create table public.learning_materials (
    id                      uuid primary key default gen_random_uuid(),
    lecture_id              uuid not null references public.lectures (id) on delete cascade,
    summary                 text not null,
    revision_notes          text not null,
    key_definitions         text,
    exam_questions          text,
    source_grounding_notes  text,
    backend_used            text not null,
    created_at              timestamptz not null default now()
);

create index learning_materials_lecture_id_idx on public.learning_materials (lecture_id);

create table public.flashcards (
    id                  uuid primary key default gen_random_uuid(),
    lecture_id          uuid not null references public.lectures (id) on delete cascade,
    topic               text not null,
    front_question      text not null,
    back_answer         text not null,
    difficulty          text not null default 'Medium',
    repetitions         integer not null default 0,
    interval_days       double precision not null default 1.0,
    ease_factor         double precision not null default 2.5,
    last_reviewed_at    timestamptz,
    next_review_date    date,
    needs_practice      boolean not null default false,
    created_at          timestamptz not null default now()
);

create index flashcards_lecture_id_idx on public.flashcards (lecture_id);
create index flashcards_topic_idx on public.flashcards (topic);
create index flashcards_next_review_date_idx on public.flashcards (next_review_date);

create table public.flashcard_reviews (
    id              uuid primary key default gen_random_uuid(),
    flashcard_id    uuid not null references public.flashcards (id) on delete cascade,
    student_id      uuid not null references public.students (id) on delete cascade,
    -- 'remembered' or 'needs_practice'
    rating          text not null,
    reviewed_at     timestamptz not null default now(),
    constraint flashcard_reviews_rating_check
        check (rating in ('remembered', 'needs_practice'))
);

create index flashcard_reviews_flashcard_id_idx on public.flashcard_reviews (flashcard_id);
create index flashcard_reviews_student_id_idx on public.flashcard_reviews (student_id);

create table public.quiz_questions (
    id                  uuid primary key default gen_random_uuid(),
    lecture_id          uuid not null references public.lectures (id) on delete cascade,
    topic               text not null,
    question_text       text not null,
    option_a            text not null,
    option_b            text not null,
    option_c            text not null,
    option_d            text not null,
    correct_option      text not null,
    explanation         text not null,
    difficulty          text not null default 'Medium',
    created_at          timestamptz not null default now(),
    constraint quiz_questions_correct_option_check
        check (correct_option in ('A', 'B', 'C', 'D'))
);

create index quiz_questions_lecture_id_idx on public.quiz_questions (lecture_id);
create index quiz_questions_topic_idx on public.quiz_questions (topic);

create table public.quiz_attempts (
    id                      uuid primary key default gen_random_uuid(),
    -- Idempotency token to prevent duplicate saves on refresh.
    attempt_uuid            uuid not null unique default gen_random_uuid(),
    student_id              uuid not null references public.students (id) on delete cascade,
    lecture_id              uuid not null references public.lectures (id) on delete cascade,
    topic                   text not null,
    score_pct               double precision not null,
    total_questions         integer not null,
    correct_count           integer not null,
    incorrect_count         integer not null,
    time_spent_seconds      double precision not null default 0.0,
    completed_at            timestamptz not null default now()
);

create index quiz_attempts_student_id_idx on public.quiz_attempts (student_id);
create index quiz_attempts_lecture_id_idx on public.quiz_attempts (lecture_id);
create index quiz_attempts_completed_at_idx on public.quiz_attempts (completed_at desc);

create table public.quiz_responses (
    id                          uuid primary key default gen_random_uuid(),
    attempt_id                  uuid not null references public.quiz_attempts (id) on delete cascade,
    question_id                 uuid not null references public.quiz_questions (id) on delete cascade,
    selected_option              text not null,
    is_correct                  boolean not null,
    response_time_seconds       double precision not null default 0.0
);

create index quiz_responses_attempt_id_idx on public.quiz_responses (attempt_id);
create index quiz_responses_question_id_idx on public.quiz_responses (question_id);

-- Adaptive SM-2 + LSTM revision plan. Re-linked to student_subjects (in
-- addition to the original student_id) so the activation-gating rule
-- (spec section 6) and the risk-boost term in calculate_topic_priority can
-- be enforced/queried relationally instead of by loose topic-string matching.
create table public.revision_schedules (
    id                      uuid primary key default gen_random_uuid(),
    student_id              uuid not null references public.students (id) on delete cascade,
    student_subject_id      uuid not null references public.student_subjects (id) on delete cascade,
    topic                   text not null,
    lecture_id              uuid references public.lectures (id) on delete set null,
    priority_score          double precision not null,
    priority_reason         text not null,
    scheduled_date          date not null,
    last_revised_at         timestamptz,
    status                  text not null default 'pending',
    sm2_interval_days       double precision not null default 1.0,
    sm2_ease_factor         double precision not null default 2.5,
    sm2_repetitions         integer not null default 0,
    lstm_predicted_score    double precision,
    created_at              timestamptz not null default now(),
    updated_at              timestamptz not null default now(),
    constraint revision_schedules_status_check
        check (status in ('pending', 'completed'))
);

create index revision_schedules_student_id_idx on public.revision_schedules (student_id);
create index revision_schedules_student_subject_id_idx on public.revision_schedules (student_subject_id);
create index revision_schedules_scheduled_date_idx on public.revision_schedules (student_id, status, scheduled_date);
