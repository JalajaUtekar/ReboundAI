"""Static validation of the Supabase SQL migrations -- the closest thing to
a test we can run without a live Postgres/Supabase connection (neither
DATABASE_URL nor SUPABASE_SERVICE_ROLE_KEY is available yet; see
docs/SUPABASE_SETUP.md).

Checks performed:
  1. Each migration file is non-empty and roughly balanced (parens, dollar-
     quoted `$$ ... $$` function bodies, no stray unterminated string).
  2. Every `create table` statement parses out a table name and at least one
     column.
  3. Every `references <table>` / `references public.<table>` in a foreign
     key points at a table actually defined by `create table` somewhere in
     the migration files (including auth.users, which is a Supabase-managed
     table we don't define but is allowed to be referenced).
  4. Every table enabled for RLS (`alter table ... enable row level
     security`) is a table that was actually created.
  5. Every table that has at least one `create policy` also has RLS enabled.

This intentionally does not require `sqlparse` (not installed in this
environment) -- it uses a small set of targeted regexes sufficient for this
specific, hand-written, consistently-formatted SQL. It is not a general
SQL parser and is not meant to be one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MIGRATIONS_DIR = REPO_ROOT / "supabase" / "migrations"

SCHEMA_SQL = MIGRATIONS_DIR / "20261017000000_rebound_core_schema.sql"
RLS_SQL = MIGRATIONS_DIR / "20261017000001_rebound_rls_policies.sql"

# Tables that exist outside our migrations (Supabase/Postgres-managed) and
# are legitimate FK targets even though we never CREATE them ourselves.
EXTERNAL_TABLES = {"auth.users", "users"}


def _read(path: Path) -> str:
    assert path.exists(), f"Expected migration file to exist: {path}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"Migration file is empty: {path}"
    return text


def _strip_comments(sql: str) -> str:
    """Removes `-- ...` line comments (but not inside string/dollar-quoted
    literals, which none of our comments are, by construction)."""
    return re.sub(r"--[^\n]*", "", sql)


@pytest.fixture(scope="module")
def schema_sql() -> str:
    return _read(SCHEMA_SQL)


@pytest.fixture(scope="module")
def rls_sql() -> str:
    return _read(RLS_SQL)


@pytest.fixture(scope="module")
def all_migration_files() -> List[Path]:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    assert files, f"No migration files found in {MIGRATIONS_DIR}"
    return files


# ----------------------------------------------------------------------------
# 1. Balance checks
# ----------------------------------------------------------------------------


def test_migration_files_exist_and_nonempty(all_migration_files: List[Path]) -> None:
    assert SCHEMA_SQL in all_migration_files
    assert RLS_SQL in all_migration_files


@pytest.mark.parametrize("path", [SCHEMA_SQL, RLS_SQL])
def test_balanced_parens(path: Path) -> None:
    sql = _strip_comments(_read(path))
    # Dollar-quoted function bodies ($$ ... $$) can contain semicolons/parens
    # of their own (e.g. "select ... where id = auth.uid();"), but they are
    # internally balanced SQL themselves, so simply counting over the whole
    # file (comments stripped) is safe: every '(' must have a matching ')'.
    depth = 0
    for ch in sql:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        assert depth >= 0, f"Unbalanced parens (extra ')') in {path.name}"
    assert depth == 0, f"Unbalanced parens (missing ')') in {path.name}: depth={depth}"


@pytest.mark.parametrize("path", [SCHEMA_SQL, RLS_SQL])
def test_balanced_dollar_quotes(path: Path) -> None:
    sql = _read(path)  # dollar quotes can legitimately contain `--`
    count = sql.count("$$")
    assert count % 2 == 0, f"Odd number of '$$' dollar-quote delimiters in {path.name} (count={count})"


@pytest.mark.parametrize("path", [SCHEMA_SQL, RLS_SQL])
def test_every_statement_terminated(path: Path) -> None:
    """Every top-level SQL statement should end with a semicolon. We check
    this by confirming the file (comments stripped, trailing whitespace
    trimmed) ends with ';', which would not hold if a statement were cut off
    mid-way through editing."""
    sql = _strip_comments(_read(path)).rstrip()
    assert sql.endswith(";"), f"{path.name} does not end with a terminated statement"


# ----------------------------------------------------------------------------
# 2 & 3. CREATE TABLE / FOREIGN KEY referential integrity
# ----------------------------------------------------------------------------

_CREATE_TABLE_RE = re.compile(r"create\s+table\s+(?:if\s+not\s+exists\s+)?(?:public\.)?(\w+)\s*\(", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"references\s+((?:public\.|auth\.)?\w+)\s*\(", re.IGNORECASE)


def _defined_tables(sql: str) -> Set[str]:
    return {m.group(1).lower() for m in _CREATE_TABLE_RE.finditer(sql)}


def _referenced_tables(sql: str) -> Set[str]:
    return {m.group(1).lower() for m in _REFERENCES_RE.finditer(sql)}


@pytest.fixture(scope="module")
def defined_tables(schema_sql: str) -> Set[str]:
    tables = _defined_tables(schema_sql)
    assert len(tables) >= 13, f"Expected at least 13 tables, found {len(tables)}: {sorted(tables)}"
    return tables


EXPECTED_TABLES = {
    "profiles",
    "students",
    "student_subjects",
    "assessment_checkpoints",
    "risk_predictions",
    "lectures",
    "learning_materials",
    "flashcards",
    "flashcard_reviews",
    "quiz_questions",
    "quiz_attempts",
    "quiz_responses",
    "revision_schedules",
}


def test_all_expected_tables_are_defined(defined_tables: Set[str]) -> None:
    missing = EXPECTED_TABLES - defined_tables
    assert not missing, f"Missing expected tables: {missing}"


def test_every_foreign_key_target_is_defined_or_external(schema_sql: str, defined_tables: Set[str]) -> None:
    referenced = _referenced_tables(schema_sql)
    unresolved = set()
    for ref in referenced:
        bare = ref.split(".")[-1]
        if ref in EXTERNAL_TABLES or bare in EXTERNAL_TABLES:
            continue
        if bare not in defined_tables:
            unresolved.add(ref)
    assert not unresolved, f"Foreign keys reference undefined tables: {unresolved}"


# ----------------------------------------------------------------------------
# 4 & 5. RLS coverage
# ----------------------------------------------------------------------------

_ENABLE_RLS_RE = re.compile(r"alter\s+table\s+(?:public\.)?(\w+)\s+enable\s+row\s+level\s+security", re.IGNORECASE)
_CREATE_POLICY_RE = re.compile(r"create\s+policy\s+\w+\s+on\s+(?:public\.)?(\w+)", re.IGNORECASE)


def test_every_defined_table_has_rls_enabled(defined_tables: Set[str], rls_sql: str) -> None:
    rls_enabled = {m.group(1).lower() for m in _ENABLE_RLS_RE.finditer(rls_sql)}
    missing = defined_tables - rls_enabled
    assert not missing, f"Tables missing 'enable row level security': {missing}"


def test_every_rls_enabled_table_has_at_least_one_policy(rls_sql: str) -> None:
    rls_enabled = {m.group(1).lower() for m in _ENABLE_RLS_RE.finditer(rls_sql)}
    policy_tables = {m.group(1).lower() for m in _CREATE_POLICY_RE.finditer(rls_sql)}
    missing = rls_enabled - policy_tables
    assert not missing, f"Tables with RLS enabled but no policies defined: {missing}"


def test_student_can_select_own_row_on_core_tables(rls_sql: str) -> None:
    """Spec section 8: a student can read their own students/student_subjects
    rows. Sanity-check the ownership policies exist and key off auth.uid()."""
    assert "profile_id = auth.uid()" in rls_sql
    assert "owner_profile_of_student" in rls_sql
    assert "owner_profile_of_student_subject" in rls_sql


def test_principal_has_unrestricted_policy_on_every_table(defined_tables: Set[str], rls_sql: str) -> None:
    policy_blocks = re.findall(
        r"create\s+policy\s+(\w+)\s+on\s+(?:public\.)?(\w+)[^;]*?current_profile_role\(\)\s*=\s*'principal'[^;]*;",
        rls_sql,
        re.IGNORECASE | re.DOTALL,
    )
    tables_with_principal_policy = {table.lower() for _, table in policy_blocks}
    missing = defined_tables - tables_with_principal_policy
    assert not missing, f"Tables missing a principal-scope policy: {missing}"


# ----------------------------------------------------------------------------
# Helper-function sanity (RLS migration)
# ----------------------------------------------------------------------------


def test_helper_functions_are_security_definer_and_search_path_pinned(rls_sql: str) -> None:
    function_blocks = re.findall(
        r"create\s+or\s+replace\s+function[^;]*?;",
        rls_sql,
        re.IGNORECASE | re.DOTALL,
    )
    assert len(function_blocks) >= 10, f"Expected >=10 helper functions, found {len(function_blocks)}"
    for block in function_blocks:
        assert re.search(r"security\s+definer", block, re.IGNORECASE), f"Function missing SECURITY DEFINER:\n{block[:200]}"
        assert re.search(r"set\s+search_path\s*=\s*public", block, re.IGNORECASE), f"Function missing pinned search_path:\n{block[:200]}"
