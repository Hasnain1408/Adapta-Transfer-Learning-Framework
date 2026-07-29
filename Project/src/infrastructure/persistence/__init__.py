"""PostgreSQL-backed implementations of the domain repository protocols.

Targets a Neon serverless database; see ``config.py`` for where the connection
string is read from and ``database.py`` for the schema.
"""

from .config import DatabaseConfigError, describe_dsn, resolve_dsn
from .database import PSYCOPG2_AVAILABLE, Database, DatabaseUnavailable
from .postgres_repositories import (
    PostgresDatasetRepository,
    PostgresEvaluationRepository,
    PostgresExperimentRepository,
    PostgresUserRepository,
)

__all__ = [
    "PSYCOPG2_AVAILABLE",
    "Database",
    "DatabaseConfigError",
    "DatabaseUnavailable",
    "PostgresDatasetRepository",
    "PostgresEvaluationRepository",
    "PostgresExperimentRepository",
    "PostgresUserRepository",
    "describe_dsn",
    "resolve_dsn",
]
