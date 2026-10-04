"""PostgreSQL-backed implementations of domain repository protocols.

Provides concrete repository implementations targeting PostgreSQL (optimized for
Neon serverless):
- config: DSN resolution and .env configuration parsing.
- driver: Dynamic psycopg2 import shim and driver availability flags.
- schema: Database DDL statements and migration definitions.
- database: Connection lifecycle, transaction management, and automatic retry.
- postgres_mappers: Database row to domain entity conversion helpers.
- postgres_user_repository: User entity persistence.
- postgres_dataset_repository: Dataset metadata catalogue persistence.
- postgres_experiment_repository: Experiment record and history persistence.
- postgres_evaluation_repository: Evaluation score and metric persistence.
"""

from .config import DatabaseConfigError, describe_dsn, resolve_dsn
from .database import PSYCOPG2_AVAILABLE, Database, DatabaseUnavailable
from .postgres_dataset_repository import PostgresDatasetRepository
from .postgres_evaluation_repository import PostgresEvaluationRepository
from .postgres_experiment_repository import PostgresExperimentRepository
from .postgres_user_repository import PostgresUserRepository

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
