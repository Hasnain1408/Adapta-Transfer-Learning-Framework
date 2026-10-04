"""PostgreSQL implementation of ExperimentRepository."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from domain.entities import ExperimentHistoryEntry, ExperimentRecord, ExperimentStatus

from .database import Database
from .postgres_mappers import _to_evaluation, _to_experiment, _to_user

# Standard column selection for experiment records.
_EXPERIMENT_COLUMNS = """
    user_id, source_dataset_id, target_dataset_id, source_domain, target_domain,
    tl_algorithm, classifier, normalization_type, standardization_type,
    date, status, latent_feature, regularization_value, kernel_type, error_message
"""


class PostgresExperimentRepository:
    """Provides CRUD operations for experiments and joined history read models."""

    def __init__(self, database: Database) -> None:
        """Initialize the experiment repository with a shared database manager."""
        self._db = database

    @staticmethod
    def _values(experiment: ExperimentRecord) -> tuple[Any, ...]:
        """Convert an ExperimentRecord entity into a tuple for parameterized SQL queries."""
        return (
            experiment.user_id,
            experiment.source_dataset_id,
            experiment.target_dataset_id,
            experiment.source_domain,
            experiment.target_domain,
            experiment.tl_algorithm,
            experiment.classifier,
            experiment.normalization_type,
            experiment.standardization_type,
            experiment.date or datetime.now().astimezone(),
            ExperimentStatus(experiment.status).value,
            experiment.latent_feature,
            experiment.regularization_value,
            experiment.kernel_type,
            experiment.error_message,
        )

    def add(self, experiment: ExperimentRecord) -> ExperimentRecord:
        """Persist a new experiment record.

        Args:
            experiment: The experiment entity to insert.

        Returns:
            ExperimentRecord: The persisted entity with generated ID and default timestamps.
        """
        row = self._db.execute_returning(
            f"""
            INSERT INTO experiments ({_EXPERIMENT_COLUMNS})
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            self._values(experiment),
        )
        assert row is not None
        return _to_experiment(row)

    def update(self, experiment: ExperimentRecord) -> ExperimentRecord:
        """Update an existing experiment record.

        Args:
            experiment: The experiment entity containing updated field values.

        Returns:
            ExperimentRecord: The updated entity.

        Raises:
            ValueError: If the entity has no ID or the row is not found.
        """
        if experiment.id is None:
            raise ValueError("Cannot update an experiment that has no id.")
        row = self._db.execute_returning(
            """
            UPDATE experiments
               SET user_id = %s, source_dataset_id = %s, target_dataset_id = %s,
                   source_domain = %s, target_domain = %s, tl_algorithm = %s,
                   classifier = %s, normalization_type = %s, standardization_type = %s,
                   date = %s, status = %s, latent_feature = %s,
                   regularization_value = %s, kernel_type = %s, error_message = %s
             WHERE id = %s
            RETURNING *
            """,
            (*self._values(experiment), experiment.id),
        )
        if row is None:
            raise ValueError(f"No experiment with id {experiment.id}.")
        return _to_experiment(row)

    def set_status(
        self, experiment_id: int, status: str, error_message: str | None = None
    ) -> None:
        """Update the status and optional error message of an experiment without modifying other columns."""
        self._db.execute(
            "UPDATE experiments SET status = %s, error_message = %s WHERE id = %s",
            (ExperimentStatus(status).value, error_message, experiment_id),
        )

    def delete(self, experiment_id: int) -> bool:
        """Delete an experiment record by ID.

        Note: Associated evaluations are deleted automatically via ON DELETE CASCADE.
        """
        return (
            self._db.execute("DELETE FROM experiments WHERE id = %s", (experiment_id,)) > 0
        )

    def get(self, experiment_id: int) -> ExperimentRecord | None:
        """Fetch an experiment record by its primary key ID."""
        row = self._db.fetch_one(
            "SELECT * FROM experiments WHERE id = %s", (experiment_id,)
        )
        return _to_experiment(row) if row else None

    def list_all(self, user_id: int | None = None) -> list[ExperimentRecord]:
        """Fetch all experiments, optionally filtered by user ID, ordered by date descending."""
        if user_id is None:
            rows = self._db.fetch_all(
                "SELECT * FROM experiments ORDER BY date DESC, id DESC"
            )
        else:
            rows = self._db.fetch_all(
                "SELECT * FROM experiments WHERE user_id = %s ORDER BY date DESC, id DESC",
                (user_id,),
            )
        return [_to_experiment(row) for row in rows]

    def history(self, user_id: int | None = None) -> list[ExperimentHistoryEntry]:
        """Fetch aggregated experiment history entries including user and evaluation data.

        Executes a single optimized query with LEFT JOIN operations across users
        and evaluations to prevent N+1 query overhead. Column aliases avoid naming
        collisions in the result set.

        Args:
            user_id: Optional user ID to filter history entries.

        Returns:
            list[ExperimentHistoryEntry]: Combined history view models sorted by date descending.
        """
        sql = """
            SELECT e.*,
                   u.id         AS u_id,
                   u.name       AS u_name,
                   u.email      AS u_email,
                   u.role       AS u_role,
                   u.created_at AS u_created_at,
                   v.id                  AS v_id,
                   v.accuracy            AS v_accuracy,
                   v."precision"         AS v_precision,
                   v.recall              AS v_recall,
                   v.f1_score            AS v_f1_score,
                   v.total_samples       AS v_total_samples,
                   v.correct_predictions AS v_correct_predictions,
                   v.wrong_predictions   AS v_wrong_predictions,
                   v.auc                 AS v_auc,
                   v.overlap_before      AS v_overlap_before,
                   v.overlap_after       AS v_overlap_after,
                   v.mmd_before          AS v_mmd_before,
                   v.mmd_after           AS v_mmd_after,
                   v.alignment_status    AS v_alignment_status,
                   v.execution_seconds   AS v_execution_seconds,
                   v.adaptation_seconds  AS v_adaptation_seconds,
                   v.baseline_accuracy   AS v_baseline_accuracy,
                   v.baseline_f1         AS v_baseline_f1
              FROM experiments e
              LEFT JOIN users       u ON u.id = e.user_id
              LEFT JOIN evaluations v ON v.experiment_id = e.id
        """
        params: tuple[Any, ...] = ()
        if user_id is not None:
            sql += " WHERE e.user_id = %s"
            params = (user_id,)
        sql += " ORDER BY e.date DESC, e.id DESC"

        entries: list[ExperimentHistoryEntry] = []
        for row in self._db.fetch_all(sql, params):
            user = (
                _to_user(
                    {
                        "id": row["u_id"],
                        "name": row["u_name"],
                        "email": row["u_email"],
                        "role": row["u_role"],
                        "created_at": row["u_created_at"],
                    }
                )
                if row.get("u_id") is not None
                else None
            )
            evaluation = (
                _to_evaluation(
                    {
                        "id": row["v_id"],
                        "experiment_id": row["id"],
                        "accuracy": row["v_accuracy"],
                        "precision": row["v_precision"],
                        "recall": row["v_recall"],
                        "f1_score": row["v_f1_score"],
                        "total_samples": row["v_total_samples"],
                        "correct_predictions": row["v_correct_predictions"],
                        "wrong_predictions": row["v_wrong_predictions"],
                        "auc": row["v_auc"],
                        "overlap_before": row["v_overlap_before"],
                        "overlap_after": row["v_overlap_after"],
                        "mmd_before": row["v_mmd_before"],
                        "mmd_after": row["v_mmd_after"],
                        "alignment_status": row["v_alignment_status"],
                        "execution_seconds": row["v_execution_seconds"],
                        "adaptation_seconds": row["v_adaptation_seconds"],
                        "baseline_accuracy": row["v_baseline_accuracy"],
                        "baseline_f1": row["v_baseline_f1"],
                    }
                )
                if row.get("v_id") is not None
                else None
            )
            entries.append(
                ExperimentHistoryEntry(
                    experiment=_to_experiment(row), user=user, evaluation=evaluation
                )
            )
        return entries

    def count(self) -> int:
        """Return the total number of registered experiments."""
        row = self._db.fetch_one("SELECT count(*) AS total FROM experiments")
        return int(row["total"]) if row else 0


__all__ = ["PostgresExperimentRepository"]
