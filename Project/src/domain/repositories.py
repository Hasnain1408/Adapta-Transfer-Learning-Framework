"""Repository contracts for the persisted entities.

Structural ``Protocol``s, matching the style of ``DatasetLoader`` in
``domain.interfaces``: the domain states what persistence must offer, and
``infrastructure.persistence`` supplies it. Nothing here imports sqlite3 or any
other storage technology, so swapping SQLite for something else touches only
the infrastructure layer.

Convention across all four: ``add`` returns the entity with its ``id``
populated, getters return ``None`` when the row is absent, and ``delete``
returns ``True`` only if a row was actually removed.
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
    def add(self, user: User) -> User:
        """Insert *user* and return it with ``id`` set.

        Raises ``ValueError`` if the email is already registered.
        """
        ...

    def update(self, user: User) -> User:
        """Persist changes to an existing user (``user.id`` must be set)."""
        ...

    def delete(self, user_id: int) -> bool:
        """Remove a user. Their experiments survive with a null owner."""
        ...

    def get(self, user_id: int) -> User | None: ...

    def get_by_email(self, email: str) -> User | None:
        """Look up by the natural key. Matching is case-insensitive."""
        ...

    def list_all(self) -> list[User]:
        """Every user, ordered by name."""
        ...


class DatasetRepository(Protocol):
    def add(self, dataset: DatasetRecord) -> DatasetRecord: ...

    def update(self, dataset: DatasetRecord) -> DatasetRecord: ...

    def delete(self, dataset_id: int) -> bool:
        """Remove a catalogue entry. Experiments referencing it keep their
        denormalised domain names and lose only the link."""
        ...

    def get(self, dataset_id: int) -> DatasetRecord | None: ...

    def get_by_path(self, source_path: str) -> DatasetRecord | None:
        """Look up by file path — the natural key used to avoid duplicates."""
        ...

    def upsert_by_path(self, dataset: DatasetRecord) -> DatasetRecord:
        """Insert *dataset*, or refresh the existing row with the same path.

        Re-running against a file that has since changed should update the
        catalogued sample/feature counts rather than add a second row.
        """
        ...

    def list_all(self) -> list[DatasetRecord]: ...


class ExperimentRepository(Protocol):
    def add(self, experiment: ExperimentRecord) -> ExperimentRecord: ...

    def update(self, experiment: ExperimentRecord) -> ExperimentRecord: ...

    def set_status(
        self, experiment_id: int, status: str, error_message: str | None = None
    ) -> None:
        """Move an experiment to a terminal state without rewriting the row."""
        ...

    def delete(self, experiment_id: int) -> bool:
        """Remove an experiment. Its evaluation row is cascaded away."""
        ...

    def get(self, experiment_id: int) -> ExperimentRecord | None: ...

    def list_all(self, user_id: int | None = None) -> list[ExperimentRecord]:
        """Newest first; restricted to one owner when *user_id* is given."""
        ...

    def history(self, user_id: int | None = None) -> list[ExperimentHistoryEntry]:
        """Experiments joined with owner and scores, newest first."""
        ...

    def count(self) -> int: ...


class EvaluationRepository(Protocol):
    def add(self, evaluation: EvaluationRecord) -> EvaluationRecord:
        """Insert scores, replacing any existing row for the same experiment."""
        ...

    def get(self, evaluation_id: int) -> EvaluationRecord | None: ...

    def get_for_experiment(self, experiment_id: int) -> EvaluationRecord | None: ...

    def delete(self, evaluation_id: int) -> bool: ...

    def list_all(self) -> list[EvaluationRecord]: ...
