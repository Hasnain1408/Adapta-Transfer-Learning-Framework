"""PostgreSQL connection management and schema creation.

Targets a Neon serverless database over ``psycopg2``. Three consequences of
that hosting model shape this module:

- **Neon scales to zero and drops idle connections.** A desktop app sits idle
  for long stretches, so a single long-lived connection *will* be closed out
  from under it. Every statement therefore goes through
  :meth:`Database._execute_with_retry`, which reconnects and retries once when
  the failure is a dropped connection (and never when it is a real SQL error).
- **The first query after an idle period pays a cold-start delay.** Connecting
  is done lazily and can fail, so callers get :class:`DatabaseUnavailable`
  rather than a hard crash — the app stays usable without persistence.
- **TLS is mandatory.** Handled in ``config.resolve_dsn``.

Access is serialised with an ``RLock``: the use case runs on a QThread while
the UI reads on the main thread, and a psycopg2 connection is not thread-safe
for concurrent statements on the same cursor.
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
    """Owns the PostgreSQL connection and guarantees the schema exists.

    Construct once at start-up, hand to the repositories, :meth:`close` on
    shutdown. Connecting is lazy — nothing touches the network until the first
    statement — so a slow Neon cold start does not block the window from
    appearing.
    """

    def __init__(self, dsn: str | None = None, *, connect_timeout: int = 10) -> None:
        self._dsn = resolve_dsn(dsn)
        self._connect_timeout = connect_timeout
        self._lock = threading.RLock()
        self._connection: Any = None
        self._schema_ready = False

    # ------------------------------------------------------------------
    @property
    def dsn(self) -> str:
        """The connection string, with the password masked."""
        return describe_dsn(self._dsn)

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connection is not None and self._connection.closed == 0

    # ------------------------------------------------------------------
    def _connect(self) -> Any:
        """Open a fresh connection. Caller must hold the lock."""
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
        except Exception as exc:  # psycopg2.OperationalError and friends
            raise DatabaseUnavailable(
                f"Could not connect to PostgreSQL at {self.dsn}.\n\n{exc}"
            ) from exc

        connection.autocommit = False
        self._connection = connection
        return connection

    def _ensure_connection(self) -> Any:
        """Return a live connection, reconnecting if the old one died."""
        if self._connection is None or self._connection.closed != 0:
            return self._connect()
        return self._connection

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Whether *exc* means "the link dropped" rather than "the SQL was bad".

        Only the former is safe to retry: a dropped connection guarantees the
        server rolled the transaction back, so replaying it cannot double-apply
        a write. A constraint violation replayed would just fail again.
        """
        if not PSYCOPG2_AVAILABLE:
            return False
        return isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError))

    def connect(self) -> None:
        """Open the connection and apply the schema. Raises DatabaseUnavailable."""
        with self._lock:
            self._ensure_connection()
            self.create_schema()

    def create_schema(self) -> None:
        """Create every table and index if absent. Safe to call repeatedly.

        Runs as one transaction, so a database is never left half-migrated.
        """

        def operation(cursor: Any) -> None:
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)

        self.run_in_transaction(operation)
        self._schema_ready = True

    # ------------------------------------------------------------------
    def run_in_transaction(self, operation: Callable[[Any], _T]) -> _T:
        """Run *operation(cursor)* in a transaction and return its result.

        Commits on success, rolls back on any exception — so a half-recorded
        experiment (a row without its evaluation) can never reach the database.

        Reconnects and retries the **whole** operation once when the failure is
        a dropped connection, which on Neon is routine rather than exceptional.
        Retrying at this level rather than inside a context manager is
        deliberate and necessary: a ``@contextmanager`` generator may only
        yield once, so re-entering the ``with`` body after an exception has
        been thrown into it raises ``RuntimeError: generator didn't stop after
        throw()``. Taking the operation as a callable lets it simply be called
        again.

        Because the server rolls back an interrupted transaction, replaying it
        cannot double-apply a write.
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
                        # Rollback on an already-dead connection is expected.
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
            # Both attempts hit a connection error.
            raise DatabaseUnavailable(
                f"Lost the connection to PostgreSQL at {self.dsn}.\n\n{last_error}"
            ) from last_error

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Yield a cursor inside a locked transaction.

        Commits on success and rolls back on any exception. Unlike
        :meth:`run_in_transaction` this does **not** retry a dropped
        connection — it cannot, for the generator reason described above. It
        does reconnect *before* yielding, so a connection that died while idle
        is replaced transparently; only one that dies mid-transaction
        surfaces as an error. Prefer :meth:`run_in_transaction` for writes.
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

    def _discard_connection(self) -> None:
        """Drop a dead connection so the next call opens a fresh one."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None

    # ------------------------------------------------------------------
    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run a SELECT and return every row as a plain dict."""

        def operation(cursor: Any) -> list[dict[str, Any]]:
            cursor.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

        return self.run_in_transaction(operation)

    def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        def operation(cursor: Any) -> dict[str, Any] | None:
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
            return dict(row) if row is not None else None

        return self.run_in_transaction(operation)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run a write and return the number of affected rows."""

        def operation(cursor: Any) -> int:
            cursor.execute(sql, tuple(params))
            return cursor.rowcount

        return self.run_in_transaction(operation)

    def execute_returning(
        self, sql: str, params: Sequence[Any] = ()
    ) -> dict[str, Any] | None:
        """Run a write with a ``RETURNING`` clause and hand back that row.

        Postgres' ``RETURNING`` replaces SQLite's ``lastrowid`` and is race
        free — the generated id comes back from the same statement that
        produced it.
        """

        def operation(cursor: Any) -> dict[str, Any] | None:
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
            return dict(row) if row is not None else None

        return self.run_in_transaction(operation)

    # ------------------------------------------------------------------
    def table_names(self) -> list[str]:
        """Tables in the current schema, for diagnostics and tests."""
        rows = self.fetch_all(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() "
            "ORDER BY tablename"
        )
        return [row["tablename"] for row in rows]

    def drop_all(self) -> None:
        """Remove every table. Used by the tests; never called by the app."""
        self.execute(
            "DROP TABLE IF EXISTS evaluations, experiments, datasets, users CASCADE"
        )
        self._schema_ready = False

    def close(self) -> None:
        with self._lock:
            self._discard_connection()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


#: ``SCHEMA_STATEMENTS``, the driver flags, and ``DatabaseUnavailable`` are
#: re-exported from their own modules so existing ``from .database import ...``
#: sites keep working after the split.
__all__ = [
    "PSYCOPG2_AVAILABLE",
    "PSYCOPG2_IMPORT_ERROR",
    "SCHEMA_STATEMENTS",
    "Database",
    "DatabaseConfigError",
    "DatabaseUnavailable",
]
