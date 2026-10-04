"""PostgreSQL connection configuration and DSN resolution.

Credentials are never hardcoded in the source tree. Connection details are
resolved in the following order of precedence:

1. Explicit DSN:
   Passed directly to resolve_dsn (used by tests to target a scratch
   database without touching the environment).

2. DATABASE_URL Environment Variable:
   Read from the system environment (for example, the connection string provided on
   the Neon project dashboard).

3. DATABASE_URL in a .env File:
   Read from a local, gitignored .env file (discovered beside the app
   executable, in the current working directory, or at the project root).

4. Individual PG* Variables:
   Assembled into a connection DSN from discrete variables (found in the system
   environment or .env):
   - PGHOST (required)
   - PGDATABASE (required)
   - PGUSER (default: postgres)
   - PGPASSWORD (default: empty string)
   - PGPORT (default: 5432)
   - PGSSLMODE (default: require)

A lightweight built-in .env parser is used below to avoid adding an
external dependency on python-dotenv for basic configuration reading.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

# Default source checkout configuration file path (used when running directly from source).
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def _env_file_candidates() -> tuple[Path, ...]:
    """Return candidate paths for locating the .env configuration file.

    Handles both standard source execution and frozen PyInstaller single-binary launches:
    1. Executable Directory: Next to sys.executable (works for frozen bundles and virtualenvs).
    2. PyInstaller MEIPASS: Temporary extraction folder if bundled as app data (sys._MEIPASS).
    3. Script Directory: Directory of sys.argv[0] (for symlinks or desktop launchers).
    4. Working Directory: Current working directory (Path.cwd()).
    5. Source Root: Project root directory (ENV_FILE).

    Returns:
        tuple[Path, ...]: De-duplicated list of candidate file paths in priority order.
    """
    candidates: list[Path] = []

    # 1. Next to the actual executable (works for both frozen and source runs).
    executable_dir = Path(sys.executable).resolve().parent
    candidates.append(executable_dir / ".env")

    # 2. For frozen one-file builds, bundled data files are extracted to sys._MEIPASS.
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / ".env")

        # argv[0] may differ from sys.executable when launched via a symlink or desktop wrapper.
        argv0_dir = Path(sys.argv[0]).resolve().parent
        candidates.append(argv0_dir / ".env")

        # Also check current working directory (e.g. running ./Adapta directly from shell).
        candidates.append(Path.cwd() / ".env")

    # 3. Source-checkout location (meaningful when running directly from Python source).
    candidates.append(ENV_FILE)

    # De-duplicate while preserving discovery order.
    return tuple(dict.fromkeys(candidates))


# Default SSL mode. Neon terminates unencrypted connections, so SSL is enforced
# by default unless overridden via PGSSLMODE (e.g., for local development).
DEFAULT_SSLMODE = "require"


class DatabaseConfigError(RuntimeError):
    """Raised when database connection details cannot be resolved from any source."""


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse a minimal KEY=value configuration file.

    Supports simple .env syntax:
    - Skips blank lines and comment lines (starting with #).
    - Strips optional export prefixes.
    - Strips matching single or double surrounding quotes.
    - Returns an empty dictionary if the file does not exist.

    Args:
        path: Explicit path to a .env file. If None, checks candidate locations.

    Returns:
        dict[str, str]: Parsed key-value configuration pairs.
    """
    paths = (path,) if path is not None else _env_file_candidates()

    values: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
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
        if values:
            break
    return values


def _lookup(key: str, env_file: dict[str, str]) -> str | None:
    """Retrieve a configuration value, prioritizing OS environment over .env."""
    value = os.environ.get(key) or env_file.get(key)
    return value.strip() if value and value.strip() else None


def _dsn_from_parts(env_file: dict[str, str]) -> str | None:
    """Assemble a PostgreSQL DSN from individual PG* configuration variables.

    Requires at minimum PGHOST and PGDATABASE to be set; returns None
    otherwise so fallback handling or informative errors can be triggered.
    Escapes user and password characters to prevent URI parsing errors.
    """
    host = _lookup("PGHOST", env_file)
    database = _lookup("PGDATABASE", env_file)
    if not host or not database:
        return None

    user = _lookup("PGUSER", env_file) or "postgres"
    password = _lookup("PGPASSWORD", env_file) or ""
    port = _lookup("PGPORT", env_file) or "5432"
    sslmode = _lookup("PGSSLMODE", env_file) or DEFAULT_SSLMODE

    # Passwords routinely contain characters like @, /, and :; without URL-encoding,
    # those would corrupt the authority component of the connection string.
    credentials = quote_plus(user)
    if password:
        credentials += f":{quote_plus(password)}"

    return f"postgresql://{credentials}@{host}:{port}/{database}?sslmode={sslmode}"


def resolve_dsn(explicit: str | None = None) -> str:
    """Resolve and return the active PostgreSQL connection string (DSN).

    Resolution Order:
    1. Explicit argument (explicit).
    2. DATABASE_URL environment variable.
    3. DATABASE_URL in the discovered .env file.
    4. Individual PGHOST, PGDATABASE, etc. variables.

    Args:
        explicit: Optional connection string override (used for testing scratch databases).

    Returns:
        str: Fully-qualified PostgreSQL DSN.

    Raises:
        DatabaseConfigError: If no valid connection configuration could be found.
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
        "environment variable or in a .env file beside the application:\n\n"
        "    DATABASE_URL=postgresql://user:password@host.neon.tech/dbname?sslmode=require\n"
    )


def _ensure_sslmode(dsn: str) -> str:
    """Ensure sslmode=require is present on remote connection strings.

    If a remote Neon URL is pasted without a query string, it defaults to unencrypted
    and fails on the server side. Localhost and 127.0.0.1 connections are left unmodified.
    """
    if "sslmode=" in dsn:
        return dsn
    # Local servers are commonly plaintext, so do not force SSL on them.
    if "localhost" in dsn or "127.0.0.1" in dsn:
        return dsn
    separator = "&" if "?" in dsn else "?"
    return f"{dsn}{separator}sslmode={DEFAULT_SSLMODE}"


def describe_dsn(dsn: str) -> str:
    """Return a sanitized connection string with credentials masked for logs and UI display."""
    if "://" not in dsn:
        return dsn
    scheme, _, remainder = dsn.partition("://")
    if "@" not in remainder:
        return dsn
    authority, _, host_part = remainder.rpartition("@")
    user = authority.partition(":")[0]
    return f"{scheme}://{user}:****@{host_part}"
