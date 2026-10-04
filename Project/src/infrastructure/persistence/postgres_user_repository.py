"""PostgreSQL implementation of UserRepository.

Queries use parameterized placeholders (%s) to eliminate SQL injection risks.
Insert and update operations utilize PostgreSQL RETURNING * clause to reconstruct
entities directly from the committed database row, ensuring server-generated timestamps
and default values are accurately reflected.
"""

from __future__ import annotations

from domain.entities import User, UserRole

from .database import Database
from .postgres_mappers import _is_unique_violation, _to_user


class PostgresUserRepository:
    """Provides CRUD operations and queries for the users table."""

    def __init__(self, database: Database) -> None:
        """Initialize the repository with a shared database manager."""
        self._db = database

    def add(self, user: User) -> User:
        """Insert a new user record into the database.

        Args:
            user: The user entity to insert.

        Returns:
            User: The created user entity populated with database-assigned ID and defaults.

        Raises:
            ValueError: If a user with the specified email address already exists.
        """
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
            # The database enforces uniqueness via the idx_users_email_lower index.
            if _is_unique_violation(exc):
                raise ValueError(
                    f"A user with the email {user.email!r} already exists."
                ) from exc
            raise
        assert row is not None  # RETURNING clause guarantees a row dictionary on success.
        return _to_user(row)

    def update(self, user: User) -> User:
        """Update an existing user record.

        Args:
            user: The user entity containing updated attributes.

        Returns:
            User: The updated user entity.

        Raises:
            ValueError: If the user ID is None, no user exists with that ID,
                        or the email collides with another user.
        """
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
        """Delete a user record by ID.

        Note: Foreign keys in experiments.user_id use ON DELETE SET NULL,
        preserving experiment records as unattributed entries.

        Args:
            user_id: The ID of the user to delete.

        Returns:
            bool: True if a user was deleted, False otherwise.
        """
        return self._db.execute("DELETE FROM users WHERE id = %s", (user_id,)) > 0

    def get(self, user_id: int) -> User | None:
        """Fetch a single user by primary key ID."""
        row = self._db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
        return _to_user(row) if row else None

    def get_by_email(self, email: str) -> User | None:
        """Fetch a single user by email address (case-insensitive comparison)."""
        row = self._db.fetch_one(
            "SELECT * FROM users WHERE lower(email) = lower(%s)", (email.strip(),)
        )
        return _to_user(row) if row else None

    def list_all(self) -> list[User]:
        """Fetch all registered users ordered by name and ID."""
        rows = self._db.fetch_all("SELECT * FROM users ORDER BY name, id")
        return [_to_user(row) for row in rows]

    def count(self) -> int:
        """Return the total number of registered users."""
        row = self._db.fetch_one("SELECT count(*) AS total FROM users")
        return int(row["total"]) if row else 0


__all__ = ["PostgresUserRepository"]
