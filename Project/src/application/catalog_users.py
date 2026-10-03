"""User accounts in the catalogue."""

from __future__ import annotations

from domain.entities import User, UserRole


class CatalogUsersMixin:
    """Account CRUD, plus the starter account a fresh database needs."""

    def create_user(
        self, name: str, email: str, role: str | UserRole, password_hash: str = ""
    ) -> User:
        return self._users.add(
            User(
                name=name.strip(),
                email=email.strip(),
                role=UserRole.from_value(str(role)),
                password_hash=password_hash,
            )
        )

    def update_user(self, user: User) -> User:
        return self._users.update(user)

    def delete_user(self, user_id: int) -> bool:
        return self._users.delete(user_id)

    def get_user(self, user_id: int) -> User | None:
        return self._users.get(user_id)

    def list_users(self) -> list[User]:
        return self._users.list_all()

    def ensure_default_user(self) -> User:
        """Return some usable user, creating a starter account if the table is empty.

        A brand-new database would otherwise leave every experiment
        unattributed, and force the user through a dialog before their first
        run. The account is a normal row and can be renamed or deleted.
        """
        existing = self._users.list_all()
        if existing:
            return existing[0]
        return self._users.add(
            User(name="Default User", email="default@local", role=UserRole.RESEARCHER)
        )


__all__ = ["CatalogUsersMixin"]
