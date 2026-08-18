"""Registration and login.

Kept separate from ``ExperimentCatalogUseCase``: authentication is a
cross-cutting concern the presentation layer needs resolved before anything
else runs, whereas the catalogue records experiments once a session is
already underway. Both sit on top of the same ``UserRepository``.
"""

from __future__ import annotations

from domain.auth import hash_password, verify_password
from domain.entities import User, UserRole
from domain.repositories import UserRepository

#: Deliberately generous rather than a full complexity policy: this is a
#: small desktop tool, not a public-facing service, so the bar is "hard to
#: guess by accident" rather than "meets a compliance checklist".
_MIN_PASSWORD_LENGTH = 8


class AuthenticationError(ValueError):
    """Invalid credentials or registration input — safe to show verbatim."""


class AuthenticationUseCase:
    """Register new accounts and verify credentials for existing ones."""

    def __init__(self, users: UserRepository) -> None:
        self._users = users

    def register(
        self,
        name: str,
        email: str,
        password: str,
        role: str | UserRole = UserRole.STUDENT,
    ) -> User:
        name = name.strip()
        email = email.strip()
        # These messages are shown to the user verbatim by the auth dialog,
        # so they are written as finished sentences — no !r quoting, which
        # renders as programmer-style 'quotes' in a dialog.
        if not name:
            raise AuthenticationError("Enter your name.")
        if not email or "@" not in email or email.startswith("@") or email.endswith("@"):
            raise AuthenticationError(f"“{email}” doesn't look like an email address.")
        if len(password) < _MIN_PASSWORD_LENGTH:
            raise AuthenticationError(
                f"Choose a password of at least {_MIN_PASSWORD_LENGTH} characters."
            )
        # Checked here too (not just the unique index) so the failure reads as
        # a normal validation message instead of a raw driver exception.
        if self._users.get_by_email(email) is not None:
            raise AuthenticationError(f"An account already exists for {email}.")

        user = User(
            name=name,
            email=email,
            role=UserRole.from_value(str(role)),
            password_hash=hash_password(password),
        )
        return self._users.add(user)

    def login(self, email: str, password: str) -> User:
        user = self._users.get_by_email(email.strip())
        if user is None or not verify_password(password, user.password_hash):
            # One message for both "no such user" and "wrong password" —
            # distinguishing them would let a caller enumerate registered
            # emails by trying passwords against them.
            raise AuthenticationError("Incorrect email or password.")
        return user


__all__ = ["AuthenticationError", "AuthenticationUseCase"]
