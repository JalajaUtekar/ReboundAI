# Supabase setup -- Rebound AI

This covers: applying the schema migrations, seeding real data, env vars,
and Google OAuth. The Supabase project itself (`kyhprimwjgdcjiifzqwn`) has
already been created by the user -- this doc starts from there.

**Status as of writing: the migrations and seed script are written and
ready to apply, but have NOT been applied to the live Supabase project.**
`DATABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are blank in `.env`, and
nothing in `supabase/` has been run against a real database. Fill in those
two values (steps 1-2 below), then run steps 3-4.

## 1. Where each env var comes from

All four live in `.env` at the repo root (never committed -- see
`.gitignore`). `SUPABASE_URL` and `SUPABASE_ANON_KEY` are already filled in.

| Variable | Where to get it | Used by |
|---|---|---|
| `SUPABASE_URL` | Supabase dashboard -> Project Settings -> API -> Project URL | Already set. Frontend (`supabase-js` client), backend |
| `SUPABASE_ANON_KEY` | Supabase dashboard -> Project Settings -> API -> `anon` `public` key | Already set. Frontend client (RLS-enforced, safe to expose) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase dashboard -> Project Settings -> API -> `service_role` `secret` key | Backend-only (Vercel serverless functions, the seed script's optional admin-API mode). **Bypasses RLS. Never ship this to the frontend or a public repo.** |
| `DATABASE_URL` | Supabase dashboard -> Project Settings -> Database -> Connection string (URI tab). Use the **pooled connection string** (port 6543, `pgbouncer=true`) for serverless functions; the direct connection string (port 5432) also works fine for one-off migration/seed runs from your own machine. | `supabase db push` / `psql`, `supabase/seed.py` |

Fill in the last two in `.env` yourself (they contain secrets this
assistant should never see or set on your behalf):

```
SUPABASE_SERVICE_ROLE_KEY=<paste from dashboard>
DATABASE_URL=<paste from dashboard>
```

## 2. Applying the migrations

Two migrations live in `supabase/migrations/`, applied in this order
(the filename timestamps already sort correctly):

1. `20261017000000_rebound_core_schema.sql` -- all table definitions
   (ScholarTrack + Lecture2Revision, Postgres-ified), indexes, constraints.
2. `20261017000001_rebound_rls_policies.sql` -- enables Row-Level Security
   on every table and defines the student/professor/HOD/principal policies
   (spec section 8).

### Option A -- Supabase CLI (recommended)

```bash
npx supabase login
npx supabase link --project-ref kyhprimwjgdcjiifzqwn
npx supabase db push
```

`supabase db push` applies every migration file in `supabase/migrations/`
that hasn't already been applied, in filename order.

### Option B -- psql directly

```bash
psql "$DATABASE_URL" -f supabase/migrations/20261017000000_rebound_core_schema.sql
psql "$DATABASE_URL" -f supabase/migrations/20261017000001_rebound_rls_policies.sql
```

Use the **direct** (non-pooled, port 5432) connection string for this --
some DDL doesn't play well through a transaction-pooling proxy.

### Verifying it applied cleanly

```sql
-- Run in the Supabase SQL Editor, or via psql
select tablename from pg_tables where schemaname = 'public' order by 1;
-- Expect 13 rows: assessment_checkpoints, flashcard_reviews, flashcards,
-- learning_materials, lectures, profiles, quiz_attempts, quiz_questions,
-- quiz_responses, revision_schedules, risk_predictions, student_subjects,
-- students.

select tablename, rowsecurity from pg_tables where schemaname = 'public';
-- Expect rowsecurity = true for all 13.
```

## 3. Seeding real data

`supabase/seed.py` reads the real 120-student synthetic cohort
(`data/synthetic_students.json`) and fits the real production model
(`scripts/training/explainability.py::train_production_model`, the
Logistic Regression config that won in `models/benchmark_report.md`) to
generate real SHAP-explained `risk_predictions` rows for every genuinely
at-risk student-subject -- nothing in the seed data is a placeholder.

```bash
# Sanity-check the data/model pipeline without touching any database:
py -3 supabase/seed.py --dry-run

# Once DATABASE_URL is set and the migrations above have been applied:
py -3 supabase/seed.py
```

The script is idempotent (deterministic UUIDs derived from each student's
PRN / subject slug / checkpoint key, upserted with `ON CONFLICT ... DO
UPDATE`) -- re-running it just refreshes the data.

**`auth.users` dependency**: `profiles.id` is a foreign key to
`auth.users.id`, so a matching Supabase Auth user must exist before a
`profiles` row can be inserted for that student. For the hackathon demo
this is normally just the one seeded demo account (`.env`'s own
documented plan: "demo account seeded via a seed script, not re-entered in
chat"). Seeding all 120 synthetic students as real `auth.users` rows is a
separate, optional step (via the Supabase Admin API with
`SUPABASE_SERVICE_ROLE_KEY`) and is not required to demo the single-student
flow -- see the comment at the top of `supabase/seed.py` for the two ways
to proceed.

## 4. Google OAuth setup (future work per spec section 9)

The app code only ever calls `supabase.auth.signInWithOAuth({ provider:
'google' })` -- no Google secret is ever handled by this codebase. All of
the actual OAuth client configuration happens in the Google Cloud Console
and the Supabase dashboard, by the user:

1. **Google Cloud Console** (console.cloud.google.com):
   - Create or select a project.
   - APIs & Services -> OAuth consent screen -> configure (App name,
     support email, scopes: `email`, `profile`, `openid`).
   - APIs & Services -> Credentials -> Create Credentials -> OAuth client
     ID -> Application type: **Web application**.
   - Authorized redirect URI: `https://kyhprimwjgdcjiifzqwn.supabase.co/auth/v1/callback`
     (Supabase shows you this exact URL on the Providers page in step 2).
   - Save, then copy the generated **Client ID** and **Client secret**.

2. **Supabase dashboard** (not this codebase):
   - Authentication -> Providers -> Google -> toggle Enabled.
   - Paste the Client ID and Client secret from step 1.
   - Save.

3. **Nothing changes in this repo.** The frontend's existing
   `supabase.auth.signInWithOAuth({ provider: 'google' })` call (once the
   frontend is built) starts working as soon as step 2 is saved. No env
   var, no migration, no code change.

The Google Client Secret must never be pasted into `.env`, a commit, or a
chat message to this assistant -- it only ever lives in the Supabase
dashboard.

## 5. Rollback / re-running

Both migrations are idempotent-safe to read but **not** auto-idempotent to
re-run as-is (`create table` / `create type` / `create policy` all fail on
a second run against the same database -- this is normal for migrations,
not a bug). To start over on a throwaway/dev project:

```sql
drop schema public cascade;
create schema public;
grant all on schema public to postgres, anon, authenticated, service_role;
```

then re-run step 2. Never do this against a database with real data you
want to keep.
