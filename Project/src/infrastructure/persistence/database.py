"""PostgreSQL connection management, retry mechanisms, and schema deployment.

Designed for serverless PostgreSQL (such as Neon) using psycopg2:
- Automatic Reconnection and Retry: Neon scales to zero and terminates idle connections.
  Queries routed through Database.run_in_transaction automatically reconnect
  and retry once on transient connection drops (while failing fast on genuine SQL errors).
- Non-blocking Lazy Initialization: Connections are opened on-demand, preventing
  cold-start latency from delaying initial application rendering.
- Thread Safety: Access is synchronized using a re-entrant lock (threading.RLock),
  permitting background execution threads (for example, Qt worker threads) and UI readers to safely
  interact with shared database handles.
- Enforced Security: TLS and SSL enforcement is integrated during DSN resolution.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence, TypeVar

from .config import DatabaseConfigError, describe_dsn, resolve_dsn
from .driver import (
    PSYCOPG2_AVAILABLE,
    PSYCOPG2_IMPORT_ERROR,
    DatabaseUnavailable,
    RealDictCursor,
    psycopg2,
)
from .schema import SCHEMA_STATEMENTS

_T = TypeVar("_T")


class Database:
    """Manages the PostgreSQL connection lifecycle and schema state.

    Instantiated during application startup and injected into domain repositories.
    Connection instantiation is deferred until the first executed query to prevent
    blocking GUI initialization.
    """

    def __init__(self, dsn: str | None = None, *, connect_timeout: int = 10) -> None:
        """Initialize the database manager with an optional DSN and timeout.

        Args:
            dsn: Explicit connection string. If None, resolved via environment or .env.
            connect_timeout: Network connection timeout in seconds.
        """
        self._dsn = resolve_dsn(dsn)
        self._connect_timeout = connect_timeout
        self._lock = threading.RLock()
        self._connection: Any = None
        self._schema_ready = False

    # ------------------------------------------------------------------
    # Connection Properties
    # ------------------------------------------------------------------
    @property
    def dsn(self) -> str:
        """Sanitized connection string with sensitive credentials masked."""
        return describe_dsn(self._dsn)

    @property
    def is_connected(self) -> bool:
        """True if an active, unclosed connection is currently held."""
        with self._lock:
            return self._connection is not None and self._connection.closed == 0

    # ------------------------------------------------------------------
    # Internal Connection Helpers
    # ------------------------------------------------------------------
    def _connect(self) -> Any:
        """Establish a new PostgreSQL connection. Caller must acquire self._lock."""
        if not PSYCOPG2_AVAILABLE:
            raise DatabaseUnavailable(
                "The psycopg2 driver is not installed.\n\n"
                "Install it with:  pip install psycopg2-binary\n"
                f"(import failed: {PSYCOPG2_IMPORT_ERROR})"
            )
        try:
            connection = psycopg2.connect(
                self._dsn,
                connect_timeout=self._connect_timeout,
                cursor_factory=RealDictCursor,
            )
        except Exception as exc:  # psycopg2.OperationalError, socket timeouts, etc.
            raise DatabaseUnavailable(
                f"Could not connect to PostgreSQL at {self.dsn}.\n\n{exc}"
            ) from exc

        connection.autocommit = False
        self._connection = connection
        return connection

    def _ensure_connection(self) -> Any:
        """Return the active connection or establish a new one if closed or disconnected."""
        if self._connection is None or self._connection.closed != 0:
            return self._connect()
        return self._connection

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Determine whether an exception represents a dropped connection rather than a bad SQL statement.

        Connection drops are safe to retry because the server automatically rolls back
        the interrupted transaction, avoiding duplicated writes.
        """
        if not PSYCOPG2_AVAILABLE:
            return False
        return isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError))

    def _discard_connection(self) -> None:
        """Close and discard the current connection reference to force a fresh connection."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None

    # ------------------------------------------------------------------
    # Schema and Lifecycle Management
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Open the connection eagerly and execute all schema statements.

        Raises:
            DatabaseUnavailable: If the driver is missing or the database cannot be contacted.
        """
        with self._lock:
            self._ensure_connection()
            self.create_schema()

    def create_schema(self) -> None:
        """Execute table and index creation statements atomically in a single transaction."""
        def operation(cursor: Any) -> None:
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)

        self.run_in_transaction(operation)
        self._schema_ready = True

    # ------------------------------------------------------------------
    # Transaction Execution
    # ------------------------------------------------------------------
    def run_in_transaction(self, operation: Callable[[Any], _T]) -> _T:
        """Execute a callable inside a managed transaction and return its result.

        Guarantees:
        - Commits automatically on successful return.
        - Rolls back on any unhandled exception.
        - Automatically reconnects and retries the entire operation once if a transient
          connection drop occurs (essential for serverless databases that scale to zero).

        Args:
            operation: A callable accepting an active database cursor and returning a value.

        Returns:
            The value returned by operation.

        Raises:
            DatabaseUnavailable: If connection cannot be established after retry.
        """
        with self._lock:
            last_error: Exception | None = None
            for attempt in (1, 2):
                connection = self._ensure_connection()
                cursor = connection.cursor()
                try:
                    result = operation(cursor)
                    connection.commit()
                    return result
                except Exception as exc:
                    try:
                        connection.rollback()
                    except Exception:
                        # Rollback on a severed connection is expected to fail safely.
                        pass
                    if attempt == 1 and self._is_connection_error(exc):
                        self._discard_connection()
                        last_error = exc
                        continue
                    raise
                finally:
                    try:
                        cursor.close()
                    except Exception:
                        pass
            # Both attempts encountered connection failures.
            raise DatabaseUnavailable(
                f"Lost the connection to PostgreSQL at {self.dsn}.\n\n{last_error}"
            ) from last_error

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Context manager yielding a cursor within a synchronized transaction block.

        Commits on normal exit and rolls back on exception. Reconnects before yielding
        if the connection was dropped while idle.
        """
        with self._lock:
            connection = self._ensure_connection()
            cursor = connection.cursor()
            try:
                yield cursor
                connection.commit()
            except Exception:
                try:
                    connection.rollback()
                except Exception:
                    pass
                raise
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Query Execution Helpers
    # ------------------------------------------------------------------
    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Execute a SELECT query and return all matching rows as dictionaries."""
        def operation(cursor: Any) -> list[dict[str, Any]]:
            cursor.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

        return self.run_in_transaction(operation)

    def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        """Execute a SELECT query and return the first matching row or None."""
        def operation(cursor: Any) -> dict[str, Any] | None:
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
            return dict(row) if row is not None else None

        return self.run_in_transaction(operation)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute an INSERT, UPDATE, or DELETE query and return the affected row count."""
        def operation(cursor: Any) -> int:
            cursor.execute(sql, tuple(params))
            return cursor.rowcount

        return self.run_in_transaction(operation)

    def execute_returning(
        self, sql: str, params: Sequence[Any] = ()
    ) -> dict[str, Any] | None:
        """Execute a write statement with a RETURNING clause and return the resulting row dictionary.

        Uses PostgreSQL's atomic RETURNING clause to reliably obtain auto-generated IDs
        and column defaults without race conditions.
        """
        def operation(cursor: Any) -> dict[str, Any] | None:
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
            return dict(row) if row is not None else None

        return self.run_in_transaction(operation)

    # ------------------------------------------------------------------
    # Diagnostic and Lifecycle Utilities
    # ------------------------------------------------------------------
    def table_names(self) -> list[str]:
        """Return the names of all tables in the current schema for diagnostics and testing."""
        rows = self.fetch_all(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() "
            "ORDER BY tablename"
        )
        return [row["tablename"] for row in rows]

    def drop_all(self) -> None:
        """Drop all application tables and reset schema status (used primarily in test suites)."""
        self.execute(
            "DROP TABLE IF EXISTS evaluations, experiments, datasets, users CASCADE"
        )
        self._schema_ready = False

    def close(self) -> None:
        """Close the active database connection."""
        with self._lock:
            self._discard_connection()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


# Re-export driver status flags and schema statements for backward compatibility.
__all__ = [
    "PSYCOPG2_AVAILABLE",
    "PSYCOPG2_IMPORT_ERROR",
    "SCHEMA_STATEMENTS",
    "Database",
    "DatabaseConfigError",
    "DatabaseUnavailable",
]
