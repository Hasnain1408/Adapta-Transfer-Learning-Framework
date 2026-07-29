"""Where the PostgreSQL connection details come from.

Credentials are never written into the source tree. They are resolved, in
order of precedence:

1. an explicit DSN passed to :func:`resolve_dsn`,
2. the ``DATABASE_URL`` environment variable — this is the connection string
   Neon hands you on the project dashboard,
3. a ``DATABASE_URL`` line in ``Project/.env`` (gitignored),
4. the individual ``PGHOST`` / ``PGDATABASE`` / ``PGUSER`` / ``PGPASSWORD`` /
   ``PGPORT`` / ``PGSSLMODE`` variables, assembled into a DSN.

The tiny ``.env`` reader below exists so the project does not need
``python-dotenv`` for a single line of configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

#: ``Project/.env`` — resolved from this file so it is found regardless of the
#: working directory the app was launched from.
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

#: Neon terminates unencrypted connections, so this is the floor, not a
#: preference. Overridable via ``PGSSLMODE`` for a local server.
DEFAULT_SSLMODE = "require"


class DatabaseConfigError(RuntimeError):
    """Raised when no connection details could be found at all."""


def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse a minimal ``KEY=value`` file. Missing file yields ``{}``.

    Deliberately small: blank lines, ``#`` comments, an optional ``export``
    prefix, and surrounding quotes. Anything more elaborate belongs in a real
    dotenv library.
    """
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip one matched pair of surrounding quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _lookup(key: str, env_file: dict[str, str]) -> str | None:
    """Real environment first, then the ``.env`` fallback."""
    value = os.environ.get(key) or env_file.get(key)
    return value.strip() if value and value.strip() else None


def _dsn_from_parts(env_file: dict[str, str]) -> str | None:
    """Assemble a DSN from the individual ``PG*`` variables.

    Requires at least a host and a database name; anything less is treated as
    "not configured" rather than guessed at.
    """
    host = _lookup("PGHOST", env_file)
    database = _lookup("PGDATABASE", env_file)
    if not host or not database:
        return None

    user = _lookup("PGUSER", env_file) or "postgres"
    password = _lookup("PGPASSWORD", env_file) or ""
    port = _lookup("PGPORT", env_file) or "5432"
    sslmode = _lookup("PGSSLMODE", env_file) or DEFAULT_SSLMODE

    # Passwords routinely contain ``@``, ``/`` and ``:``; without escaping,
    # those silently corrupt the URL's authority section.
    credentials = quote_plus(user)
    if password:
        credentials += f":{quote_plus(password)}"

    return f"postgresql://{credentials}@{host}:{port}/{database}?sslmode={sslmode}"


def resolve_dsn(explicit: str | None = None) -> str:
    """Return the connection string to use, or raise :class:`DatabaseConfigError`.

    ``explicit`` wins when given, which is how the tests point at a scratch
    database without touching the environment.
    """
    if explicit and explicit.strip():
        return explicit.strip()

    env_file = load_env_file()

    dsn = _lookup("DATABASE_URL", env_file)
    if dsn:
        return _ensure_sslmode(dsn)

    dsn = _dsn_from_parts(env_file)
    if dsn:
        return dsn

    raise DatabaseConfigError(
        "No PostgreSQL connection details found.\n\n"
        "Set DATABASE_URL to your Neon connection string, either as an "
        f"environment variable or in {ENV_FILE}:\n\n"
        "    DATABASE_URL=postgresql://user:password@host.neon.tech/dbname?sslmode=require\n"
    )


def _ensure_sslmode(dsn: str) -> str:
    """Append ``sslmode=require`` when the URL omits it.

    A Neon URL pasted without its query string fails to connect with an
    obscure server-side error; defaulting it here turns that into a working
    connection.
    """
    if "sslmode=" in dsn:
        return dsn
    # Local servers are commonly plaintext, so do not force SSL on them.
    if "localhost" in dsn or "127.0.0.1" in dsn:
        return dsn
    separator = "&" if "?" in dsn else "?"
    return f"{dsn}{separator}sslmode={DEFAULT_SSLMODE}"


def describe_dsn(dsn: str) -> str:
    """A log/UI-safe rendering of *dsn* with the password removed."""
    if "://" not in dsn:
        return dsn
    scheme, _, remainder = dsn.partition("://")
    if "@" not in remainder:
        return dsn
    authority, _, host_part = remainder.rpartition("@")
    user = authority.partition(":")[0]
    return f"{scheme}://{user}:****@{host_part}"
