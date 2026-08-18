"""psycopg2 import shim and the "no database" error.

Isolated so every module can ask whether the driver is present without
importing the connection machinery — ``postgres_mappers`` needs only the
``PSYCOPG2_AVAILABLE`` flag and the error classes, not ``Database``.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only when the driver is absent
    import psycopg2
    from psycopg2 import errors as pg_errors
    from psycopg2.extras import RealDictCursor

    PSYCOPG2_AVAILABLE = True
    PSYCOPG2_IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover
    psycopg2 = None  # type: ignore[assignment]
    pg_errors = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]
    PSYCOPG2_AVAILABLE = False
    PSYCOPG2_IMPORT_ERROR = str(exc)


class DatabaseUnavailable(RuntimeError):
    """The database could not be reached or the driver is not installed.

    Distinct from a SQL error: the app catches this to run in a degraded,
    persistence-free mode instead of refusing to start.
    """


__all__ = [
    "PSYCOPG2_AVAILABLE",
    "PSYCOPG2_IMPORT_ERROR",
    "DatabaseUnavailable",
    "RealDictCursor",
    "pg_errors",
    "psycopg2",
]
