"""PostgreSQL implementation of ``EvaluationRepository``."""

from __future__ import annotations

from domain.entities import EvaluationRecord

from .database import Database
from .postgres_mappers import _to_evaluation


class PostgresEvaluationRepository:
    """CRUD for ``evaluations`` — at most one row per experiment."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def add(self, evaluation: EvaluationRecord) -> EvaluationRecord:
        """Insert scores, replacing any existing row for the same experiment.

        ``ON CONFLICT (experiment_id)`` upholds the 1:1 relationship: re-running
        and re-saving an experiment overwrites its scores rather than failing
        on the unique constraint.
        """
        row = self._db.execute_returning(
            """
            INSERT INTO evaluations (
                experiment_id, accuracy, "precision", recall, f1_score,
                total_samples, correct_predictions, wrong_predictions,
                auc, mmd_before, mmd_after, alignment_status,
                execution_seconds, adaptation_seconds,
                baseline_accuracy, baseline_f1
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s)
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
                baseline_f1         = EXCLUDED.baseline_f1
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
            ),
        )
        assert row is not None
        return _to_evaluation(row)

    def get(self, evaluation_id: int) -> EvaluationRecord | None:
        row = self._db.fetch_one(
            "SELECT * FROM evaluations WHERE id = %s", (evaluation_id,)
        )
        return _to_evaluation(row) if row else None

    def get_for_experiment(self, experiment_id: int) -> EvaluationRecord | None:
        row = self._db.fetch_one(
            "SELECT * FROM evaluations WHERE experiment_id = %s", (experiment_id,)
        )
        return _to_evaluation(row) if row else None

    def delete(self, evaluation_id: int) -> bool:
        return (
            self._db.execute("DELETE FROM evaluations WHERE id = %s", (evaluation_id,)) > 0
        )

    def list_all(self) -> list[EvaluationRecord]:
        rows = self._db.fetch_all("SELECT * FROM evaluations ORDER BY id DESC")
        return [_to_evaluation(row) for row in rows]

    def count(self) -> int:
        row = self._db.fetch_one("SELECT count(*) AS total FROM evaluations")
        return int(row["total"]) if row else 0


__all__ = ["PostgresEvaluationRepository"]
