"""User registration and authentication use case implementation.

Kept separate from ExperimentCatalogUseCase: authentication is a cross-cutting
concern resolved by the presentation layer before starting user sessions. Both
rely upon UserRepository for entity persistence.
"""

from __future__ import annotations

from domain.auth import hash_password, verify_password
from domain.entities import User, UserRole
from domain.repositories import UserRepository

# Minimum password length validation constraint.
_MIN_PASSWORD_LENGTH = 8


class AuthenticationError(ValueError):
    """Raised when authentication credentials or registration parameters fail validation."""


class AuthenticationUseCase:
    """Provides user account registration and credential verification operations."""

    def __init__(self, users: UserRepository) -> None:
        """Initialize authentication use case with user repository dependency."""
        self._users = users

    def register(
        self,
        name: str,
        email: str,
        password: str,
        role: str | UserRole = UserRole.STUDENT,
    ) -> User:
        """Register a new user account.

        Args:
            name: Full name of the user.
            email: Unique email address.
            password: Plaintext password.
            role: Assigned authorization role.

        Returns:
            User: Created user entity populated with database ID and password hash.

        Raises:
            AuthenticationError: If inputs are invalid or email is already registered.
        """
        name = name.strip()
        email = email.strip()
        if not name:
            raise AuthenticationError("Enter your name.")
        if not email or "@" not in email or email.startswith("@") or email.endswith("@"):
            raise AuthenticationError(f"{email} does not look like a valid email address.")
        if len(password) < _MIN_PASSWORD_LENGTH:
            raise AuthenticationError(
                f"Choose a password of at least {_MIN_PASSWORD_LENGTH} characters."
            )
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
        """Verify user credentials and return authenticated user entity.

        Args:
            email: Account email address.
            password: Plaintext password.

        Returns:
            User: Authenticated user entity.

        Raises:
            AuthenticationError: If credentials do not match an existing record.
        """
        user = self._users.get_by_email(email.strip())
        if user is None or not verify_password(password, user.password_hash):
            raise AuthenticationError("Incorrect email or password.")
        return user


__all__ = ["AuthenticationError", "AuthenticationUseCase"]
