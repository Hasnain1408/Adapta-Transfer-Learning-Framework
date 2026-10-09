"""Password hashing for user accounts.

Stdlib-only PBKDF2-HMAC-SHA256 with a random salt per user, so no third-party
dependency needs adding for something this small. The stored value is a single
self-describing string (``algorithm$iterations$salt$hash``) so the iteration
count can be raised later without invalidating hashes already in the database.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 260_000
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    """Hash *password* for storage. Never returns or logs the plaintext."""
    salt = secrets.token_hex(_SALT_BYTES)
    digest = _derive(password, salt, _ITERATIONS)
    return f"{_ALGORITHM}${_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Whether *password* matches a hash produced by :func:`hash_password`.

    Returns ``False`` (rather than raising) for an empty or malformed stored
    hash, so a row with no password on file — e.g. one created by an admin
    without setting one — simply never authenticates instead of crashing.
    """
    if not stored_hash:
        return False
    try:
        algorithm, iterations, salt, digest = stored_hash.split("$")
    except ValueError:
        return False
    if algorithm != _ALGORITHM:
        return False
    computed = _derive(password, salt, int(iterations))
    # Constant-time comparison: a timing difference between "wrong at
    # character 1" and "wrong at character 30" would leak the hash byte by
    # byte to an attacker measuring response times.
    return hmac.compare_digest(computed, digest)


def _derive(password: str, salt: str, iterations: int) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations
    ).hex()


__all__ = ["hash_password", "verify_password"]
