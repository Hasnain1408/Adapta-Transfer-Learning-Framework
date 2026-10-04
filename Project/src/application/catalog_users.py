"""User account operations and default user management mixin."""

from __future__ import annotations

from domain.entities import User, UserRole


class CatalogUsersMixin:
    """Provides user account CRUD operations and default user provisioning."""

    def create_user(
        self, name: str, email: str, role: str | UserRole, password_hash: str = ""
    ) -> User:
        """Create and persist a new user account."""
        return self._users.add(
            User(
                name=name.strip(),
                email=email.strip(),
                role=UserRole.from_value(str(role)),
                password_hash=password_hash,
            )
        )

    def update_user(self, user: User) -> User:
        """Update an existing user account."""
        return self._users.update(user)

    def delete_user(self, user_id: int) -> bool:
        """Delete a user account by ID."""
        return self._users.delete(user_id)

    def get_user(self, user_id: int) -> User | None:
        """Fetch a user account by ID."""
        return self._users.get(user_id)

    def list_users(self) -> list[User]:
        """Fetch all registered user accounts."""
        return self._users.list_all()

    def ensure_default_user(self) -> User:
        """Retrieve the primary user account or initialize a default account on first startup."""
        existing = self._users.list_all()
        if existing:
            return existing[0]
        return self._users.add(
            User(name="Default User", email="default@local", role=UserRole.RESEARCHER)
        )


__all__ = ["CatalogUsersMixin"]
