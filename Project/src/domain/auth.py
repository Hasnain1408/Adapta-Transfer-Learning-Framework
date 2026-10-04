"""Password hashing and verification utilities for user accounts.

Uses PBKDF2-HMAC-SHA256 with per-user cryptographic salts and self-describing
encoded hash strings (algorithm$iterations$salt$hash).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 260_000
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    """Hash a plaintext password for secure storage using salted PBKDF2."""
    salt = secrets.token_hex(_SALT_BYTES)
    digest = _derive(password, salt, _ITERATIONS)
    return f"{_ALGORITHM}${_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a candidate password against a stored PBKDF2 hash using constant-time comparison."""
    if not stored_hash:
        return False
    try:
        algorithm, iterations, salt, digest = stored_hash.split("$")
    except ValueError:
        return False
    if algorithm != _ALGORITHM:
        return False
    computed = _derive(password, salt, int(iterations))
    return hmac.compare_digest(computed, digest)


def _derive(password: str, salt: str, iterations: int) -> str:
    """Derive hex digest using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations
    ).hex()


__all__ = ["hash_password", "verify_password"]
