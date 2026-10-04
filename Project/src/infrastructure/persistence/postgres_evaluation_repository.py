"""PostgreSQL implementation of EvaluationRepository."""

from __future__ import annotations

from domain.entities import EvaluationRecord

from .database import Database
from .postgres_mappers import _to_evaluation


class PostgresEvaluationRepository:
    """Provides CRUD operations for evaluation metrics (1:1 relationship with experiments)."""

    def __init__(self, database: Database) -> None:
        """Initialize the evaluation repository with a shared database manager."""
        self._db = database

    def add(self, evaluation: EvaluationRecord) -> EvaluationRecord:
        """Persist evaluation metrics, replacing existing records for the same experiment.

        Uses ON CONFLICT (experiment_id) DO UPDATE to ensure idempotency when re-running
        or re-evaluating existing experiment runs.
        """
        row = self._db.execute_returning(
            """
            INSERT INTO evaluations (
                experiment_id, accuracy, "precision", recall, f1_score,
                total_samples, correct_predictions, wrong_predictions,
                auc, mmd_before, mmd_after, alignment_status,
                execution_seconds, adaptation_seconds,
                baseline_accuracy, baseline_f1,
                overlap_before, overlap_after
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id) DO UPDATE SET
                accuracy            = EXCLUDED.accuracy,
                "precision"         = EXCLUDED."precision",
                recall              = EXCLUDED.recall,
                f1_score            = EXCLUDED.f1_score,
                total_samples       = EXCLUDED.total_samples,
                correct_predictions = EXCLUDED.correct_predictions,
                wrong_predictions   = EXCLUDED.wrong_predictions,
                auc                 = EXCLUDED.auc,
                mmd_before          = EXCLUDED.mmd_before,
                mmd_after           = EXCLUDED.mmd_after,
                alignment_status    = EXCLUDED.alignment_status,
                execution_seconds   = EXCLUDED.execution_seconds,
                adaptation_seconds  = EXCLUDED.adaptation_seconds,
                baseline_accuracy   = EXCLUDED.baseline_accuracy,
                baseline_f1         = EXCLUDED.baseline_f1,
                overlap_before      = EXCLUDED.overlap_before,
                overlap_after       = EXCLUDED.overlap_after
            RETURNING *
            """,
            (
                evaluation.experiment_id,
                evaluation.accuracy,
                evaluation.precision,
                evaluation.recall,
                evaluation.f1_score,
                evaluation.total_samples,
                evaluation.correct_predictions,
                evaluation.wrong_predictions,
                evaluation.auc,
                evaluation.mmd_before,
                evaluation.mmd_after,
                evaluation.alignment_status,
                evaluation.execution_seconds,
                evaluation.adaptation_seconds,
                evaluation.baseline_accuracy,
                evaluation.baseline_f1,
                evaluation.overlap_before,
                evaluation.overlap_after,
            ),
        )
        assert row is not None
        return _to_evaluation(row)

    def get(self, evaluation_id: int) -> EvaluationRecord | None:
        """Fetch an evaluation record by its primary key ID."""
        row = self._db.fetch_one(
            "SELECT * FROM evaluations WHERE id = %s", (evaluation_id,)
        )
        return _to_evaluation(row) if row else None

    def get_for_experiment(self, experiment_id: int) -> EvaluationRecord | None:
        """Fetch the evaluation record associated with a given experiment ID."""
        row = self._db.fetch_one(
            "SELECT * FROM evaluations WHERE experiment_id = %s", (experiment_id,)
        )
        return _to_evaluation(row) if row else None

    def delete(self, evaluation_id: int) -> bool:
        """Delete an evaluation record by ID."""
        return (
            self._db.execute("DELETE FROM evaluations WHERE id = %s", (evaluation_id,)) > 0
        )

    def list_all(self) -> list[EvaluationRecord]:
        """Fetch all evaluation records ordered from newest to oldest."""
        rows = self._db.fetch_all("SELECT * FROM evaluations ORDER BY id DESC")
        return [_to_evaluation(row) for row in rows]

    def count(self) -> int:
        """Return the total number of evaluation records."""
        row = self._db.fetch_one("SELECT count(*) AS total FROM evaluations")
        return int(row["total"]) if row else 0


__all__ = ["PostgresEvaluationRepository"]
