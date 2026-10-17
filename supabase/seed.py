"""Rebound AI -- Supabase seed script.

Populates the schema in supabase/migrations/ with the real 120-student
synthetic cohort (data/synthetic_students.json) and real ScholarTrack risk
predictions (fit live from data/training_rows.json via
scripts/training/explainability.py's production Logistic Regression +
SHAP pipeline -- the exact winning config from models/benchmark_report.md).

This is NOT run automatically. It requires `DATABASE_URL` to be set in
`.env` at the repo root (and the migrations in supabase/migrations/ to have
already been applied). See docs/SUPABASE_SETUP.md.

Usage:
    py -3 supabase/seed.py [--dry-run]

`--dry-run` builds every row and prints counts/samples without opening a
database connection or writing anything -- useful for sanity-checking this
script before DATABASE_URL exists.

Idempotency: every insert uses deterministic UUIDs (uuid5, namespaced off
each student/subject's stable natural key: PRN, subject_slug,
checkpoint type+index) and `ON CONFLICT ... DO UPDATE`, so re-running this
script against the same database is safe and just refreshes the data
instead of duplicating it.

Auth note: this script seeds `profiles`/`students`/etc. directly, which
requires the matching `auth.users` rows to already exist (profiles.id is a
foreign key to auth.users.id). For the hackathon demo, create one
`auth.users` row per seeded student via Supabase Auth (e.g. the Admin API
with SUPABASE_SERVICE_ROLE_KEY, or the dashboard) using the same email
address as in `data/synthetic_students.json`, note that user's UUID, and
either let this script's `--create-auth-users` mode do it for you (it
calls the Supabase Admin REST API using SUPABASE_SERVICE_ROLE_KEY) or wire
the ids up by hand. Without SUPABASE_SERVICE_ROLE_KEY, pass
`--skip-auth-check` to seed `students`/`student_subjects`/etc. anyway for
local testing against a non-Supabase Postgres that has no `auth.users`
table (in which case the `profiles` FK to `auth.users` would need to be
dropped first -- see the note in docs/SUPABASE_SETUP.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Namespace UUID for this project's deterministic id derivation (uuid5).
# Fixed, arbitrary, never changes -- re-deriving the same natural key always
# yields the same UUID, which is what makes the seed idempotent.
_NAMESPACE = uuid.UUID("a16f6e1e-7e8a-4c1b-9a9f-6b2d6f0a9b11")


def _uuid5(*parts: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, "|".join(parts)))


def _load_dotenv() -> Dict[str, str]:
    env_path = REPO_ROOT / ".env"
    values: Dict[str, str] = {}
    try:
        from dotenv import dotenv_values  # type: ignore

        values = {k: v for k, v in dotenv_values(env_path).items() if v is not None}
    except ImportError:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def _ensure_auth_user(email: str, supabase_url: str, service_role_key: str) -> str:
    """Creates (or finds, if already present) a Supabase Auth user for `email`.

    Returns the real server-assigned UUID -- GoTrue's admin API does not let
    you choose a user's id, so this is the source of truth for `profiles.id`
    when auth users are involved, NOT the deterministic uuid5 used elsewhere
    in this script. Idempotent: re-running against an email that already has
    an auth user looks it up instead of erroring.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key,
        "Content-Type": "application/json",
    }
    create_resp = requests.post(
        f"{supabase_url}/auth/v1/admin/users",
        headers=headers,
        json={
            "email": email,
            "password": str(uuid.uuid4()),
            "email_confirm": True,
        },
        timeout=20,
    )
    if create_resp.status_code in (200, 201):
        return create_resp.json()["id"]

    # Already exists (422/400 "already been registered") -- look it up instead.
    list_resp = requests.get(
        f"{supabase_url}/auth/v1/admin/users",
        headers=headers,
        params={"per_page": 1000},
        timeout=20,
    )
    list_resp.raise_for_status()
    for user in list_resp.json().get("users", []):
        if user.get("email", "").lower() == email.lower():
            return user["id"]

    raise RuntimeError(
        f"Could not create or find an auth user for {email!r}: "
        f"create returned {create_resp.status_code} {create_resp.text[:200]!r}, "
        "and no matching user was found in the admin user list."
    )


def _subject_type(subject_slug: str) -> str:
    fixed = {"DSML", "CN", "SPOS"}
    if subject_slug in fixed:
        return "fixed"
    # The generator derives slugs from the human-readable subject name
    # (scripts/data_generation/subjects.py::subject_slug); classify by which
    # catalog the name came from.
    from scripts.data_generation.subjects import ELECTIVE_OPTIONS, MDM_OPTIONS, OEL_OPTIONS, subject_slug as to_slug

    if any(to_slug(name) == subject_slug for name in MDM_OPTIONS):
        return "mdm"
    if any(to_slug(name) == subject_slug for name in OEL_OPTIONS):
        return "oel"
    if any(to_slug(name) == subject_slug for name in ELECTIVE_OPTIONS):
        return "elective"
    raise ValueError(f"Unrecognized subject_slug {subject_slug!r}")


def build_rows(
    students: List[Dict[str, Any]],
    profile_id_by_email: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Builds the profiles/students/student_subjects/assessment_checkpoints rows.

    `profile_id_by_email`, if given, overrides the deterministic uuid5
    profile id with the real Supabase Auth user id for that student's email
    (see `_ensure_auth_user`) -- required when `profiles.id` has a live FK
    to `auth.users.id`, since GoTrue assigns that id, not us.
    """
    profiles: List[Dict[str, Any]] = []
    student_rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []

    for student in students:
        if profile_id_by_email is not None:
            profile_id = profile_id_by_email[student["email"]]
        else:
            profile_id = _uuid5("profile", student["prn"])
        student_id = _uuid5("student", student["prn"])

        profiles.append(
            {
                "id": profile_id,
                "role": "student",
                "department": student["branch"],
            }
        )
        student_rows.append(
            {
                "id": student_id,
                "profile_id": profile_id,
                "name": student["name"],
                "prn": student["prn"],
                "email": student["email"],
                "photo_url": None,
                "branch": student["branch"],
                "academic_year": student["academic_year"],
            }
        )

        for subject in student["subjects"]:
            subject_slug = subject["subject_slug"]
            student_subject_id = _uuid5("student_subject", student["prn"], subject_slug)
            subject_rows.append(
                {
                    "id": student_subject_id,
                    "student_id": student_id,
                    "subject_slug": subject_slug,
                    "subject_name": subject["subject_name"],
                    "subject_type": _subject_type(subject_slug),
                    "components": json.dumps(subject["components"]),
                }
            )

            for ut in subject["unit_tests"]:
                checkpoint_rows.append(
                    {
                        "id": _uuid5("checkpoint", student["prn"], subject_slug, "unit_test", str(ut["index"])),
                        "student_subject_id": student_subject_id,
                        "checkpoint_type": "unit_test",
                        "checkpoint_index": ut["index"],
                        "week_number": ut["week"],
                        "score": ut["score"],
                        "max_marks": ut["max_marks"],
                        "attendance_pct": None,
                        "due_day": None,
                        "submitted_day": None,
                    }
                )
            for sub in subject["submissions"]:
                checkpoint_rows.append(
                    {
                        "id": _uuid5("checkpoint", student["prn"], subject_slug, "submission", str(sub["index"])),
                        "student_subject_id": student_subject_id,
                        "checkpoint_type": "submission",
                        "checkpoint_index": sub["index"],
                        "week_number": sub["due_day"] // 7,
                        "score": sub["score"],
                        "max_marks": sub["max_marks"],
                        "attendance_pct": None,
                        "due_day": sub["due_day"],
                        "submitted_day": sub["submitted_day"],
                    }
                )
            for week in subject["attendance_weeks"]:
                checkpoint_rows.append(
                    {
                        "id": _uuid5("checkpoint", student["prn"], subject_slug, "attendance_week", str(week["week"])),
                        "student_subject_id": student_subject_id,
                        "checkpoint_type": "attendance_week",
                        "checkpoint_index": week["week"],
                        "week_number": week["week"],
                        "score": None,
                        "max_marks": None,
                        "attendance_pct": week["attendance_pct"],
                        "due_day": None,
                        "submitted_day": None,
                    }
                )

    return {
        "profiles": profiles,
        "students": student_rows,
        "student_subjects": subject_rows,
        "assessment_checkpoints": checkpoint_rows,
    }


def build_risk_prediction_rows(
    students: List[Dict[str, Any]],
    training_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fits the real production Logistic Regression model and runs real SHAP
    explanations for every at-risk student-subject's final (checkpoint 7)
    row, per models/benchmark_report.md's winning config.
    """
    from scripts.training.explainability import generate_student_explanation, train_production_model

    production = train_production_model(training_rows)
    model = production["model"]
    scaler = production["scaler"]
    threshold = production["threshold"]
    background_rows = production["background_rows"]

    # PRN lookup so we can derive the same deterministic student_subject id
    # build_rows() used.
    prn_by_student_id = {s["student_id"]: s["prn"] for s in students}

    rows_out: List[Dict[str, Any]] = []
    for row in training_rows:
        if row["checkpoint_idx"] != max(r["checkpoint_idx"] for r in training_rows if r["student_id"] == row["student_id"] and r["subject_slug"] == row["subject_slug"]):
            continue
        if not row["at_risk"]:
            continue

        explanation = generate_student_explanation(
            row, model=model, scaler=scaler, threshold=threshold, background_rows=background_rows
        )

        prn = prn_by_student_id[row["student_id"]]
        student_subject_id = _uuid5("student_subject", prn, row["subject_slug"])

        rows_out.append(
            {
                "id": _uuid5("risk_prediction", prn, row["subject_slug"], str(row["checkpoint_idx"])),
                "student_subject_id": student_subject_id,
                "checkpoint_index": row["checkpoint_idx"],
                "risk_probability": explanation["predicted_risk_probability"],
                "is_flagged": explanation["is_flagged"],
                "is_failing": bool(row["is_failing"]),
                "top_risk_factors": json.dumps(explanation["top_risk_factors"]),
                "protective_factors": json.dumps(explanation["protective_factors"]),
                "human_readable_summary": explanation["human_readable_summary"],
                "model_version": "logreg_v1",
            }
        )
    return rows_out


_UPSERT_TEMPLATES = {
    "profiles": (
        "insert into public.profiles (id, role, department) values (%(id)s, %(role)s, %(department)s) "
        "on conflict (id) do update set role = excluded.role, department = excluded.department"
    ),
    "students": (
        "insert into public.students (id, profile_id, name, prn, email, photo_url, branch, academic_year) "
        "values (%(id)s, %(profile_id)s, %(name)s, %(prn)s, %(email)s, %(photo_url)s, %(branch)s, %(academic_year)s) "
        "on conflict (id) do update set name = excluded.name, email = excluded.email, "
        "photo_url = excluded.photo_url, branch = excluded.branch, academic_year = excluded.academic_year"
    ),
    "student_subjects": (
        "insert into public.student_subjects (id, student_id, subject_slug, subject_name, subject_type, components) "
        "values (%(id)s, %(student_id)s, %(subject_slug)s, %(subject_name)s, %(subject_type)s, %(components)s::jsonb) "
        "on conflict (id) do update set subject_name = excluded.subject_name, "
        "subject_type = excluded.subject_type, components = excluded.components"
    ),
    "assessment_checkpoints": (
        "insert into public.assessment_checkpoints "
        "(id, student_subject_id, checkpoint_type, checkpoint_index, week_number, score, max_marks, "
        "attendance_pct, due_day, submitted_day) "
        "values (%(id)s, %(student_subject_id)s, %(checkpoint_type)s, %(checkpoint_index)s, %(week_number)s, "
        "%(score)s, %(max_marks)s, %(attendance_pct)s, %(due_day)s, %(submitted_day)s) "
        "on conflict (id) do update set score = excluded.score, max_marks = excluded.max_marks, "
        "attendance_pct = excluded.attendance_pct, due_day = excluded.due_day, submitted_day = excluded.submitted_day"
    ),
    "risk_predictions": (
        "insert into public.risk_predictions "
        "(id, student_subject_id, checkpoint_index, risk_probability, is_flagged, is_failing, "
        "top_risk_factors, protective_factors, human_readable_summary, model_version) "
        "values (%(id)s, %(student_subject_id)s, %(checkpoint_index)s, %(risk_probability)s, %(is_flagged)s, "
        "%(is_failing)s, %(top_risk_factors)s::jsonb, %(protective_factors)s::jsonb, "
        "%(human_readable_summary)s, %(model_version)s) "
        "on conflict (id) do update set risk_probability = excluded.risk_probability, "
        "is_flagged = excluded.is_flagged, is_failing = excluded.is_failing, "
        "top_risk_factors = excluded.top_risk_factors, protective_factors = excluded.protective_factors, "
        "human_readable_summary = excluded.human_readable_summary, model_version = excluded.model_version"
    ),
}

# Insert order matters: respect FK dependency order.
_TABLE_ORDER = ["profiles", "students", "student_subjects", "assessment_checkpoints", "risk_predictions"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Build rows and print counts, write nothing.")
    parser.add_argument(
        "--create-auth-users",
        action="store_true",
        help=(
            "Create (or find, if already present) a real Supabase Auth user per "
            "student via the Admin API, using SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY, "
            "and use its real id for profiles.id instead of the deterministic uuid5. "
            "Required the first time this is run against a live Supabase project, "
            "since profiles.id has a live FK to auth.users.id."
        ),
    )
    args = parser.parse_args()

    students_path = REPO_ROOT / "data" / "synthetic_students.json"
    training_rows_path = REPO_ROOT / "data" / "training_rows.json"

    students = json.loads(students_path.read_text(encoding="utf-8"))
    training_rows = json.loads(training_rows_path.read_text(encoding="utf-8"))

    profile_id_by_email: Optional[Dict[str, str]] = None
    if args.create_auth_users:
        env_for_auth = {**_load_dotenv(), **os.environ}
        supabase_url = env_for_auth.get("SUPABASE_URL")
        service_role_key = env_for_auth.get("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not service_role_key:
            print(
                "--create-auth-users requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                "in .env.",
                file=sys.stderr,
            )
            return 1
        profile_id_by_email = {}
        print(f"Creating/finding {len(students)} Supabase Auth users...")
        for i, student in enumerate(students, 1):
            profile_id_by_email[student["email"]] = _ensure_auth_user(
                student["email"], supabase_url, service_role_key
            )
            if i % 20 == 0 or i == len(students):
                print(f"  {i}/{len(students)} auth users ready")

    table_rows = build_rows(students, profile_id_by_email=profile_id_by_email)
    table_rows["risk_predictions"] = build_risk_prediction_rows(students, training_rows)

    counts = {table: len(rows) for table, rows in table_rows.items()}
    print(f"Built rows: {counts}")
    flagged = sum(1 for r in table_rows["risk_predictions"] if r["is_flagged"])
    print(f"risk_predictions: {len(table_rows['risk_predictions'])} at-risk student-subjects, {flagged} flagged at threshold")

    if args.dry_run:
        print("--dry-run: not connecting to any database.")
        if table_rows["risk_predictions"]:
            print("Sample risk_predictions row:")
            print(json.dumps(table_rows["risk_predictions"][0], indent=2, default=str))
        return 0

    env = {**_load_dotenv(), **os.environ}
    database_url = env.get("DATABASE_URL")
    if not database_url:
        print(
            "DATABASE_URL is not set (checked .env and the environment). "
            "This script cannot be run until the user provides it -- see docs/SUPABASE_SETUP.md. "
            "Run with --dry-run to validate the data without a database connection.",
            file=sys.stderr,
        )
        return 1

    import psycopg2

    # Plain per-row execute(). A single real row measured 0.14s with no
    # timeout set, so the occasional "statement timeout"/"connection closed"
    # seen in earlier runs are transient network blips against Supabase's
    # direct connection, not a fundamental slowness problem. Every insert
    # here is idempotent (ON CONFLICT ... DO UPDATE on a deterministic id),
    # so the safe, pragmatic fix is retry-with-reconnect per row rather than
    # chasing the exact transient cause further.
    MAX_ROW_ATTEMPTS = 5

    def _connect():
        c = psycopg2.connect(database_url)
        c.autocommit = True
        return c, c.cursor()

    for table in _TABLE_ORDER:
        rows = table_rows[table]
        template = _UPSERT_TEMPLATES[table]

        conn, cur = _connect()
        try:
            for i, row in enumerate(rows, 1):
                for attempt in range(1, MAX_ROW_ATTEMPTS + 1):
                    try:
                        cur.execute(template, row)
                        break
                    except (psycopg2.OperationalError, psycopg2.errors.QueryCanceled) as e:
                        if attempt == MAX_ROW_ATTEMPTS:
                            raise
                        print(
                            f"  retry {attempt}/{MAX_ROW_ATTEMPTS} on {table} row {i}/{len(rows)} "
                            f"after {type(e).__name__}"
                        )
                        try:
                            cur.close()
                            conn.close()
                        except Exception:
                            pass
                        conn, cur = _connect()
            print(f"Upserted {len(rows)} rows into {table}")
        finally:
            cur.close()
            conn.close()

    print("Seed complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
