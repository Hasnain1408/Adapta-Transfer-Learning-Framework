"""PostgreSQL implementation of ``UserRepository``.

Placeholders are ``%s`` (psycopg2's paramstyle), never f-strings — values are
always passed as parameters so nothing user-supplied is interpolated into a
statement. Inserts and updates end with ``RETURNING *`` and rebuild the
entity from what the database actually stored, so defaults filled in by the
schema (``created_at``) come back populated rather than being guessed at on
the Python side.
"""

from __future__ import annotations

from domain.entities import User, UserRole

from .database import Database
from .postgres_mappers import _is_unique_violation, _to_user


class PostgresUserRepository:
    """CRUD for ``users``."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def add(self, user: User) -> User:
        try:
            row = self._db.execute_returning(
                """
                INSERT INTO users (name, email, password_hash, role)
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (
                    user.name.strip(),
                    user.email.strip(),
                    user.password_hash,
                    UserRole(user.role).value,
                ),
            )
        except Exception as exc:
            # The unique index is on lower(email), so this is the only way a
            # duplicate can arrive; translate it into a message the UI can show.
            if _is_unique_violation(exc):
                raise ValueError(
                    f"A user with the email {user.email!r} already exists."
                ) from exc
            raise
        assert row is not None  # RETURNING on a successful INSERT always yields a row
        return _to_user(row)

    def update(self, user: User) -> User:
        if user.id is None:
            raise ValueError("Cannot update a user that has no id.")
        try:
            row = self._db.execute_returning(
                """
                UPDATE users
                   SET name = %s, email = %s, password_hash = %s, role = %s
                 WHERE id = %s
                RETURNING *
                """,
                (
                    user.name.strip(),
                    user.email.strip(),
                    user.password_hash,
                    UserRole(user.role).value,
                    user.id,
                ),
            )
        except Exception as exc:
            if _is_unique_violation(exc):
                raise ValueError(
                    f"A user with the email {user.email!r} already exists."
                ) from exc
            raise
        if row is None:
            raise ValueError(f"No user with id {user.id}.")
        return _to_user(row)

    def delete(self, user_id: int) -> bool:
        # experiments.user_id is ON DELETE SET NULL, so the owner's history
        # survives this and simply becomes unattributed.
        return self._db.execute("DELETE FROM users WHERE id = %s", (user_id,)) > 0

    def get(self, user_id: int) -> User | None:
        row = self._db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
        return _to_user(row) if row else None

    def get_by_email(self, email: str) -> User | None:
        row = self._db.fetch_one(
            "SELECT * FROM users WHERE lower(email) = lower(%s)", (email.strip(),)
        )
        return _to_user(row) if row else None

    def list_all(self) -> list[User]:
        rows = self._db.fetch_all("SELECT * FROM users ORDER BY name, id")
        return [_to_user(row) for row in rows]

    def count(self) -> int:
        row = self._db.fetch_one("SELECT count(*) AS total FROM users")
        return int(row["total"]) if row else 0


__all__ = ["PostgresUserRepository"]
