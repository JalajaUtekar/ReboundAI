-- Rebound AI -- Row-Level Security policies (spec section 8).
--
-- Pattern used: SECURITY DEFINER helper functions, not repeated subqueries.
--
-- Why: a policy on `profiles` cannot subquery `profiles` itself without
-- risking recursive RLS evaluation, and every other table's policy needs
-- the SAME "what is this caller's role/department, and which branch does
-- this row's student belong to" lookup repeated across 11 tables. Instead
-- of inlining that subquery in every policy (11 tables x up to 4 policies
-- each), each lookup is written once as a `SECURITY DEFINER` SQL function.
-- SECURITY DEFINER functions execute with the privileges of the function
-- owner (not the calling role), so they read `profiles`/`students`/
-- `student_subjects` directly without being re-filtered by those tables'
-- own RLS policies -- this is the standard Supabase-recommended pattern for
-- avoiding recursive-RLS deadlock and policy duplication. Each function is
-- `stable` (safe to call multiple times per statement) and has
-- `search_path` pinned to `public` (required hardening for SECURITY
-- DEFINER functions, see security-and-hardening skill / Supabase docs).
--
-- Role model recap (spec section 8):
--   student    -- own row only (students/student_subjects linked via
--                 profile_id = auth.uid()).
--   professor  -- read-only, all rows whose student's branch matches the
--                 professor's own profiles.department.
--   hod        -- same read scope as professor, PLUS insert/update/delete
--                 within that same department (full admin for their dept).
--   principal  -- full read/write across every department, no filter.
--
-- Only the student tier has a live UI today (spec section 2/8), but every
-- policy below is written now so adding the Professor/HOD/Principal
-- dashboards later is additive -- no further migration needed.

-- ----------------------------------------------------------------------------
-- Helper functions
-- ----------------------------------------------------------------------------

create or replace function public.current_profile_role() returns text
    language sql
    security definer
    stable
    set search_path = public
as $$
    select role::text from public.profiles where id = auth.uid();
$$;

create or replace function public.current_profile_department() returns text
    language sql
    security definer
    stable
    set search_path = public
as $$
    select department from public.profiles where id = auth.uid();
$$;

-- The students.id row owned by the calling user, or NULL if they are not a
-- student (or have no student row yet).
create or replace function public.current_student_id() returns uuid
    language sql
    security definer
    stable
    set search_path = public
as $$
    select id from public.students where profile_id = auth.uid();
$$;

create or replace function public.branch_of_student(p_student_id uuid) returns text
    language sql
    security definer
    stable
    set search_path = public
as $$
    select branch from public.students where id = p_student_id;
$$;

create or replace function public.owner_profile_of_student(p_student_id uuid) returns uuid
    language sql
    security definer
    stable
    set search_path = public
as $$
    select profile_id from public.students where id = p_student_id;
$$;

create or replace function public.branch_of_student_subject(p_student_subject_id uuid) returns text
    language sql
    security definer
    stable
    set search_path = public
as $$
    select s.branch
    from public.student_subjects ss
    join public.students s on s.id = ss.student_id
    where ss.id = p_student_subject_id;
$$;

create or replace function public.owner_profile_of_student_subject(p_student_subject_id uuid) returns uuid
    language sql
    security definer
    stable
    set search_path = public
as $$
    select s.profile_id
    from public.student_subjects ss
    join public.students s on s.id = ss.student_id
    where ss.id = p_student_subject_id;
$$;

create or replace function public.branch_of_lecture(p_lecture_id uuid) returns text
    language sql
    security definer
    stable
    set search_path = public
as $$
    select s.branch
    from public.lectures l
    join public.student_subjects ss on ss.id = l.student_subject_id
    join public.students s on s.id = ss.student_id
    where l.id = p_lecture_id;
$$;

create or replace function public.owner_profile_of_lecture(p_lecture_id uuid) returns uuid
    language sql
    security definer
    stable
    set search_path = public
as $$
    select s.profile_id
    from public.lectures l
    join public.student_subjects ss on ss.id = l.student_subject_id
    join public.students s on s.id = ss.student_id
    where l.id = p_lecture_id;
$$;

create or replace function public.branch_of_flashcard(p_flashcard_id uuid) returns text
    language sql
    security definer
    stable
    set search_path = public
as $$
    select public.branch_of_lecture(f.lecture_id)
    from public.flashcards f
    where f.id = p_flashcard_id;
$$;

create or replace function public.owner_profile_of_flashcard(p_flashcard_id uuid) returns uuid
    language sql
    security definer
    stable
    set search_path = public
as $$
    select public.owner_profile_of_lecture(f.lecture_id)
    from public.flashcards f
    where f.id = p_flashcard_id;
$$;

create or replace function public.branch_of_quiz_question(p_question_id uuid) returns text
    language sql
    security definer
    stable
    set search_path = public
as $$
    select public.branch_of_lecture(q.lecture_id)
    from public.quiz_questions q
    where q.id = p_question_id;
$$;

create or replace function public.owner_profile_of_quiz_question(p_question_id uuid) returns uuid
    language sql
    security definer
    stable
    set search_path = public
as $$
    select public.owner_profile_of_lecture(q.lecture_id)
    from public.quiz_questions q
    where q.id = p_question_id;
$$;

create or replace function public.branch_of_quiz_attempt(p_attempt_id uuid) returns text
    language sql
    security definer
    stable
    set search_path = public
as $$
    select public.branch_of_student(a.student_id)
    from public.quiz_attempts a
    where a.id = p_attempt_id;
$$;

create or replace function public.owner_profile_of_quiz_attempt(p_attempt_id uuid) returns uuid
    language sql
    security definer
    stable
    set search_path = public
as $$
    select public.owner_profile_of_student(a.student_id)
    from public.quiz_attempts a
    where a.id = p_attempt_id;
$$;

-- ----------------------------------------------------------------------------
-- Enable RLS on every table
-- ----------------------------------------------------------------------------

alter table public.profiles enable row level security;
alter table public.students enable row level security;
alter table public.student_subjects enable row level security;
alter table public.assessment_checkpoints enable row level security;
alter table public.risk_predictions enable row level security;
alter table public.lectures enable row level security;
alter table public.learning_materials enable row level security;
alter table public.flashcards enable row level security;
alter table public.flashcard_reviews enable row level security;
alter table public.quiz_questions enable row level security;
alter table public.quiz_attempts enable row level security;
alter table public.quiz_responses enable row level security;
alter table public.revision_schedules enable row level security;

-- ----------------------------------------------------------------------------
-- profiles
-- ----------------------------------------------------------------------------

create policy profiles_select_own
    on public.profiles for select
    using (id = auth.uid());

create policy profiles_select_department_staff
    on public.profiles for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and department = public.current_profile_department()
    );

create policy profiles_select_principal
    on public.profiles for select
    using (public.current_profile_role() = 'principal');

-- A new auth user creates their own profile row during onboarding.
create policy profiles_insert_own
    on public.profiles for insert
    with check (id = auth.uid());

create policy profiles_update_own
    on public.profiles for update
    using (id = auth.uid())
    with check (id = auth.uid());

create policy profiles_write_hod_department
    on public.profiles for all
    using (
        public.current_profile_role() = 'hod'
        and department = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and department = public.current_profile_department()
    );

create policy profiles_write_principal
    on public.profiles for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- students
-- ----------------------------------------------------------------------------

create policy students_select_own
    on public.students for select
    using (profile_id = auth.uid());

create policy students_update_own
    on public.students for update
    using (profile_id = auth.uid())
    with check (profile_id = auth.uid());

-- A new student onboarding (Section 2) inserts their own row once.
create policy students_insert_own
    on public.students for insert
    with check (profile_id = auth.uid());

create policy students_select_department_staff
    on public.students for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and branch = public.current_profile_department()
    );

create policy students_write_hod_department
    on public.students for all
    using (
        public.current_profile_role() = 'hod'
        and branch = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and branch = public.current_profile_department()
    );

create policy students_write_principal
    on public.students for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- student_subjects
-- ----------------------------------------------------------------------------

create policy student_subjects_select_own
    on public.student_subjects for select
    using (public.owner_profile_of_student(student_id) = auth.uid());

create policy student_subjects_update_own
    on public.student_subjects for update
    using (public.owner_profile_of_student(student_id) = auth.uid())
    with check (public.owner_profile_of_student(student_id) = auth.uid());

create policy student_subjects_insert_own
    on public.student_subjects for insert
    with check (public.owner_profile_of_student(student_id) = auth.uid());

create policy student_subjects_select_department_staff
    on public.student_subjects for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_student(student_id) = public.current_profile_department()
    );

create policy student_subjects_write_hod_department
    on public.student_subjects for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_student(student_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_student(student_id) = public.current_profile_department()
    );

create policy student_subjects_write_principal
    on public.student_subjects for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- assessment_checkpoints (read-only for students: these are entered/synced
-- by the application, not hand-edited by the student)
-- ----------------------------------------------------------------------------

create policy assessment_checkpoints_select_own
    on public.assessment_checkpoints for select
    using (public.owner_profile_of_student_subject(student_subject_id) = auth.uid());

create policy assessment_checkpoints_select_department_staff
    on public.assessment_checkpoints for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_student_subject(student_subject_id) = public.current_profile_department()
    );

create policy assessment_checkpoints_write_hod_department
    on public.assessment_checkpoints for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_student_subject(student_subject_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_student_subject(student_subject_id) = public.current_profile_department()
    );

create policy assessment_checkpoints_write_principal
    on public.assessment_checkpoints for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- risk_predictions (read-only for students: model output, never student-edited)
-- ----------------------------------------------------------------------------

create policy risk_predictions_select_own
    on public.risk_predictions for select
    using (public.owner_profile_of_student_subject(student_subject_id) = auth.uid());

create policy risk_predictions_select_department_staff
    on public.risk_predictions for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_student_subject(student_subject_id) = public.current_profile_department()
    );

create policy risk_predictions_write_hod_department
    on public.risk_predictions for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_student_subject(student_subject_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_student_subject(student_subject_id) = public.current_profile_department()
    );

create policy risk_predictions_write_principal
    on public.risk_predictions for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- lectures
-- ----------------------------------------------------------------------------

create policy lectures_select_own
    on public.lectures for select
    using (public.owner_profile_of_student_subject(student_subject_id) = auth.uid());

create policy lectures_insert_own
    on public.lectures for insert
    with check (public.owner_profile_of_student_subject(student_subject_id) = auth.uid());

create policy lectures_update_own
    on public.lectures for update
    using (public.owner_profile_of_student_subject(student_subject_id) = auth.uid())
    with check (public.owner_profile_of_student_subject(student_subject_id) = auth.uid());

create policy lectures_select_department_staff
    on public.lectures for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_student_subject(student_subject_id) = public.current_profile_department()
    );

create policy lectures_write_hod_department
    on public.lectures for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_student_subject(student_subject_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_student_subject(student_subject_id) = public.current_profile_department()
    );

create policy lectures_write_principal
    on public.lectures for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- learning_materials (ties to lectures -> student_subjects)
-- ----------------------------------------------------------------------------

create policy learning_materials_select_own
    on public.learning_materials for select
    using (public.owner_profile_of_lecture(lecture_id) = auth.uid());

create policy learning_materials_insert_own
    on public.learning_materials for insert
    with check (public.owner_profile_of_lecture(lecture_id) = auth.uid());

create policy learning_materials_update_own
    on public.learning_materials for update
    using (public.owner_profile_of_lecture(lecture_id) = auth.uid())
    with check (public.owner_profile_of_lecture(lecture_id) = auth.uid());

create policy learning_materials_select_department_staff
    on public.learning_materials for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_lecture(lecture_id) = public.current_profile_department()
    );

create policy learning_materials_write_hod_department
    on public.learning_materials for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_lecture(lecture_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_lecture(lecture_id) = public.current_profile_department()
    );

create policy learning_materials_write_principal
    on public.learning_materials for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- flashcards (ties to lectures -> student_subjects)
-- ----------------------------------------------------------------------------

create policy flashcards_select_own
    on public.flashcards for select
    using (public.owner_profile_of_lecture(lecture_id) = auth.uid());

create policy flashcards_insert_own
    on public.flashcards for insert
    with check (public.owner_profile_of_lecture(lecture_id) = auth.uid());

create policy flashcards_update_own
    on public.flashcards for update
    using (public.owner_profile_of_lecture(lecture_id) = auth.uid())
    with check (public.owner_profile_of_lecture(lecture_id) = auth.uid());

create policy flashcards_select_department_staff
    on public.flashcards for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_lecture(lecture_id) = public.current_profile_department()
    );

create policy flashcards_write_hod_department
    on public.flashcards for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_lecture(lecture_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_lecture(lecture_id) = public.current_profile_department()
    );

create policy flashcards_write_principal
    on public.flashcards for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- flashcard_reviews (student_id direct)
-- ----------------------------------------------------------------------------

create policy flashcard_reviews_select_own
    on public.flashcard_reviews for select
    using (public.owner_profile_of_student(student_id) = auth.uid());

create policy flashcard_reviews_insert_own
    on public.flashcard_reviews for insert
    with check (public.owner_profile_of_student(student_id) = auth.uid());

create policy flashcard_reviews_select_department_staff
    on public.flashcard_reviews for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_student(student_id) = public.current_profile_department()
    );

create policy flashcard_reviews_write_hod_department
    on public.flashcard_reviews for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_student(student_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_student(student_id) = public.current_profile_department()
    );

create policy flashcard_reviews_write_principal
    on public.flashcard_reviews for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- quiz_questions (ties to lectures -> student_subjects)
-- ----------------------------------------------------------------------------

create policy quiz_questions_select_own
    on public.quiz_questions for select
    using (public.owner_profile_of_lecture(lecture_id) = auth.uid());

create policy quiz_questions_insert_own
    on public.quiz_questions for insert
    with check (public.owner_profile_of_lecture(lecture_id) = auth.uid());

create policy quiz_questions_select_department_staff
    on public.quiz_questions for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_lecture(lecture_id) = public.current_profile_department()
    );

create policy quiz_questions_write_hod_department
    on public.quiz_questions for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_lecture(lecture_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_lecture(lecture_id) = public.current_profile_department()
    );

create policy quiz_questions_write_principal
    on public.quiz_questions for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- quiz_attempts (student_id direct)
-- ----------------------------------------------------------------------------

create policy quiz_attempts_select_own
    on public.quiz_attempts for select
    using (public.owner_profile_of_student(student_id) = auth.uid());

create policy quiz_attempts_insert_own
    on public.quiz_attempts for insert
    with check (public.owner_profile_of_student(student_id) = auth.uid());

create policy quiz_attempts_select_department_staff
    on public.quiz_attempts for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_student(student_id) = public.current_profile_department()
    );

create policy quiz_attempts_write_hod_department
    on public.quiz_attempts for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_student(student_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_student(student_id) = public.current_profile_department()
    );

create policy quiz_attempts_write_principal
    on public.quiz_attempts for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- quiz_responses (ties to quiz_attempts -> students)
-- ----------------------------------------------------------------------------

create policy quiz_responses_select_own
    on public.quiz_responses for select
    using (public.owner_profile_of_quiz_attempt(attempt_id) = auth.uid());

create policy quiz_responses_insert_own
    on public.quiz_responses for insert
    with check (public.owner_profile_of_quiz_attempt(attempt_id) = auth.uid());

create policy quiz_responses_select_department_staff
    on public.quiz_responses for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_quiz_attempt(attempt_id) = public.current_profile_department()
    );

create policy quiz_responses_write_hod_department
    on public.quiz_responses for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_quiz_attempt(attempt_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_quiz_attempt(attempt_id) = public.current_profile_department()
    );

create policy quiz_responses_write_principal
    on public.quiz_responses for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');

-- ----------------------------------------------------------------------------
-- revision_schedules (student_id direct, plus student_subject_id for gating)
-- ----------------------------------------------------------------------------

create policy revision_schedules_select_own
    on public.revision_schedules for select
    using (public.owner_profile_of_student(student_id) = auth.uid());

create policy revision_schedules_insert_own
    on public.revision_schedules for insert
    with check (public.owner_profile_of_student(student_id) = auth.uid());

create policy revision_schedules_update_own
    on public.revision_schedules for update
    using (public.owner_profile_of_student(student_id) = auth.uid())
    with check (public.owner_profile_of_student(student_id) = auth.uid());

create policy revision_schedules_select_department_staff
    on public.revision_schedules for select
    using (
        public.current_profile_role() in ('professor', 'hod')
        and public.branch_of_student(student_id) = public.current_profile_department()
    );

create policy revision_schedules_write_hod_department
    on public.revision_schedules for all
    using (
        public.current_profile_role() = 'hod'
        and public.branch_of_student(student_id) = public.current_profile_department()
    )
    with check (
        public.current_profile_role() = 'hod'
        and public.branch_of_student(student_id) = public.current_profile_department()
    );

create policy revision_schedules_write_principal
    on public.revision_schedules for all
    using (public.current_profile_role() = 'principal')
    with check (public.current_profile_role() = 'principal');
