"""Domain repository protocol contracts for persisted entities.

Defines structural Protocol interfaces that concrete storage adapters in
infrastructure.persistence must implement.
"""

from __future__ import annotations

from typing import Protocol

from .entities import (
    DatasetRecord,
    EvaluationRecord,
    ExperimentHistoryEntry,
    ExperimentRecord,
    User,
)


class UserRepository(Protocol):
    """Storage contract for User entity persistence."""

    def add(self, user: User) -> User:
        """Insert a user record and return it with assigned ID."""
        ...

    def update(self, user: User) -> User:
        """Update an existing user record."""
        ...

    def delete(self, user_id: int) -> bool:
        """Delete a user record by ID."""
        ...

    def get(self, user_id: int) -> User | None:
        """Fetch a user by ID."""
        ...

    def get_by_email(self, email: str) -> User | None:
        """Fetch a user by case-insensitive email address."""
        ...

    def list_all(self) -> list[User]:
        """Fetch all users ordered by name."""
        ...


class DatasetRepository(Protocol):
    """Storage contract for DatasetRecord catalogue persistence."""

    def add(self, dataset: DatasetRecord) -> DatasetRecord:
        """Insert a new dataset record."""
        ...

    def update(self, dataset: DatasetRecord) -> DatasetRecord:
        """Update an existing dataset record."""
        ...

    def delete(self, dataset_id: int) -> bool:
        """Delete a dataset record by ID."""
        ...

    def get(self, dataset_id: int) -> DatasetRecord | None:
        """Fetch a dataset record by ID."""
        ...

    def get_by_path(self, source_path: str) -> DatasetRecord | None:
        """Fetch a dataset record by its unique file path."""
        ...

    def upsert_by_path(self, dataset: DatasetRecord) -> DatasetRecord:
        """Insert a dataset record or update existing statistics if the path exists."""
        ...

    def list_all(self) -> list[DatasetRecord]:
        """Fetch all registered dataset records."""
        ...


class ExperimentRepository(Protocol):
    """Storage contract for ExperimentRecord persistence and history read models."""

    def add(self, experiment: ExperimentRecord) -> ExperimentRecord:
        """Insert a new experiment record."""
        ...

    def update(self, experiment: ExperimentRecord) -> ExperimentRecord:
        """Update an existing experiment record."""
        ...

    def set_status(
        self, experiment_id: int, status: str, error_message: str | None = None
    ) -> None:
        """Update the status of an experiment record without full row rewrite."""
        ...

    def delete(self, experiment_id: int) -> bool:
        """Delete an experiment record by ID."""
        ...

    def get(self, experiment_id: int) -> ExperimentRecord | None:
        """Fetch an experiment record by ID."""
        ...

    def list_all(self, user_id: int | None = None) -> list[ExperimentRecord]:
        """Fetch all experiments, optionally filtered by user ID."""
        ...

    def history(self, user_id: int | None = None) -> list[ExperimentHistoryEntry]:
        """Fetch aggregated history entries joining experiments with users and scores."""
        ...

    def count(self) -> int:
        """Return the total number of registered experiments."""
        ...


class EvaluationRepository(Protocol):
    """Storage contract for EvaluationRecord metrics persistence."""

    def add(self, evaluation: EvaluationRecord) -> EvaluationRecord:
        """Insert evaluation scores, replacing existing scores for the same experiment."""
        ...

    def get(self, evaluation_id: int) -> EvaluationRecord | None:
        """Fetch an evaluation record by ID."""
        ...

    def get_for_experiment(self, experiment_id: int) -> EvaluationRecord | None:
        """Fetch the evaluation record associated with a given experiment ID."""
        ...

    def delete(self, evaluation_id: int) -> bool:
        """Delete an evaluation record by ID."""
        ...

    def list_all(self) -> list[EvaluationRecord]:
        """Fetch all evaluation records."""
        ...
