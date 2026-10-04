"""Dynamic import shim for the PostgreSQL driver (psycopg2).

Isolates psycopg2 imports and error handling so other persistence modules
(such as mappers or repositories) can inspect driver availability and catch
database errors without triggering premature connection attempts.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only when psycopg2 is installed
    import psycopg2
    from psycopg2 import errors as pg_errors
    from psycopg2.extras import RealDictCursor

    PSYCOPG2_AVAILABLE = True
    PSYCOPG2_IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - fallback when psycopg2 is missing
    psycopg2 = None  # type: ignore[assignment]
    pg_errors = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]
    PSYCOPG2_AVAILABLE = False
    PSYCOPG2_IMPORT_ERROR = str(exc)


class DatabaseUnavailable(RuntimeError):
    """Raised when PostgreSQL cannot be reached or the driver is missing.

    Distinct from query or syntax errors: the desktop application catches this exception
    to enter a graceful, persistence-free degraded mode rather than crashing on launch.
    """


__all__ = [
    "PSYCOPG2_AVAILABLE",
    "PSYCOPG2_IMPORT_ERROR",
    "DatabaseUnavailable",
    "RealDictCursor",
    "pg_errors",
    "psycopg2",
]
