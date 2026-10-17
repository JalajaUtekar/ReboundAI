"""Rebound AI -- shared database access for Vercel Python Serverless Functions.

Every `api/*.py` endpoint (ScholarTrack's `predict_risk`, Lecture2Revision's
`process_lecture`, SM-2's `sm2_review`/`adaptive_schedule`, ...) gets its
database connection from here, so there is exactly one place that knows how
`DATABASE_URL` is loaded and connected to.

Vercel Python functions are cold-started per invocation -- there is no
long-lived process to hold a connection pool across requests, so this is
intentionally a plain `psycopg2.connect()` per call, matching the pattern
`supabase/seed.py` already uses against this same live Supabase project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_dotenv() -> Dict[str, str]:
    """Reads `.env` at the repo root into a dict.

    Mirrors `supabase/seed.py::_load_dotenv`'s fallback-without-python-dotenv
    behavior exactly, so this module has no hard dependency on the
    `python-dotenv` package being present in whatever runtime installs
    `api/`'s requirements (Vercel's own environment variables -- set via its
    dashboard/CLI for a real deployment -- are merged in via `os.environ`
    regardless of whether `.env` parsing works).
    """
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import dotenv_values  # type: ignore

        return {k: v for k, v in dotenv_values(env_path).items() if v is not None}
    except ImportError:
        values: Dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
        return values


def get_db_connection():
    """Returns a new `psycopg2` connection to the live Postgres database.

    Reads `DATABASE_URL` from `.env` (repo root) merged with the real process
    environment (`os.environ` wins on conflict -- a real Vercel deployment's
    dashboard-configured env vars take priority over a checked-in `.env`,
    which should not be deployed anyway). Raises `RuntimeError` if
    `DATABASE_URL` is not set anywhere, with the same message shape
    `supabase/seed.py` uses so it's recognizable in logs.

    No pooling, no retry-with-backoff here by design -- see the module
    docstring. Callers that need the `supabase/seed.py`-style retry-on-
    transient-network-blip behavior for a bulk/background job should layer
    that on top of this, rather than this helper hiding it from a request
    handler that just wants a connection or a clear failure.
    """
    import psycopg2

    env = {**_load_dotenv(), **os.environ}
    database_url = env.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set (checked .env and the environment)."
        )
    return psycopg2.connect(database_url)
