"""Experiment execution catalogue mixin for recording runs and evaluations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from domain.entities import (
    EvaluationRecord,
    ExperimentHistoryEntry,
    ExperimentRecord,
    ExperimentStatus,
)
from domain.models import TransferLearningRequest, TransferLearningResult

# Unused parameter marker mapped to NULL in database records.
_UNUSED = None


class CatalogExperimentsMixin:
    """Provides experiment execution recording, evaluations persistence, and history retrieval."""

    def record_success(
        self,
        request: TransferLearningRequest,
        result: TransferLearningResult,
        user_id: int | None,
        supported_parameters: frozenset[str] | None = None,
    ) -> ExperimentRecord | None:
        """Persist a successfully completed experiment run, its datasets, and evaluation metrics.

        Args:
            request: Transfer learning execution request parameters.
            result: Execution result payload containing metrics and domain models.
            user_id: ID of the authenticated user running the experiment.
            supported_parameters: Optional set of parameters applicable to the method.

        Returns:
            ExperimentRecord | None: Persisted experiment record or None if user_id is None.
        """
        if user_id is None:
            return None

        source_record = self.register_dataset(request.source_path, result.source_dataset)
        target_record = self.register_dataset(request.target_path, result.target_dataset)

        experiment = self._experiments.add(
            self._build_record(
                request,
                user_id=user_id,
                status=ExperimentStatus.COMPLETED,
                source_dataset_id=source_record.id if source_record else None,
                target_dataset_id=target_record.id if target_record else None,
                supported_parameters=supported_parameters,
            )
        )

        metrics = result.metrics
        if metrics is not None and experiment.id is not None:
            alignment = result.alignment
            analysis = result.negative_transfer
            self._evaluations.add(
                EvaluationRecord(
                    experiment_id=experiment.id,
                    accuracy=float(metrics.accuracy),
                    precision=float(metrics.precision),
                    recall=float(metrics.recall),
                    f1_score=float(metrics.f1_score),
                    total_samples=int(metrics.total_samples),
                    correct_predictions=int(metrics.correct_predictions),
                    wrong_predictions=int(metrics.wrong_predictions),
                    auc=metrics.auc,
                    overlap_before=alignment.overlap_before if alignment else None,
                    overlap_after=alignment.overlap_after if alignment else None,
                    mmd_before=alignment.mmd_before if alignment else None,
                    mmd_after=alignment.mmd_after if alignment else None,
                    alignment_status=alignment.status if alignment else None,
                    execution_seconds=result.duration_seconds,
                    adaptation_seconds=result.adaptation_seconds,
                    baseline_accuracy=analysis.baseline.accuracy if analysis else None,
                    baseline_f1=analysis.baseline.f1_score if analysis else None,
                )
            )
        return experiment

    def record_failure(
        self,
        request: TransferLearningRequest,
        user_id: int | None,
        error_message: str,
        supported_parameters: frozenset[str] | None = None,
    ) -> ExperimentRecord | None:
        """Persist a failed experiment configuration and its associated error message."""
        if user_id is None:
            return None

        return self._experiments.add(
            self._build_record(
                request,
                user_id=user_id,
                status=ExperimentStatus.FAILED,
                error_message=error_message,
                supported_parameters=supported_parameters,
            )
        )

    def _build_record(
        self,
        request: TransferLearningRequest,
        *,
        user_id: int | None,
        status: ExperimentStatus,
        source_dataset_id: int | None = None,
        target_dataset_id: int | None = None,
        error_message: str | None = None,
        supported_parameters: frozenset[str] | None = None,
    ) -> ExperimentRecord:
        supported = supported_parameters
        uses = (lambda field: True) if supported is None else (lambda field: field in supported)

        return ExperimentRecord(
            user_id=user_id,
            source_dataset_id=source_dataset_id,
            target_dataset_id=target_dataset_id,
            source_domain=Path(request.source_path).stem or request.source_path,
            target_domain=Path(request.target_path).stem or request.target_path,
            tl_algorithm=request.method_name,
            classifier=request.classifier,
            normalization_type=request.normalization,
            standardization_type=request.standardization,
            date=datetime.now(timezone.utc).astimezone(),
            status=status,
            latent_feature=int(request.latent_dim) if uses("latent_dim") else _UNUSED,
            regularization_value=(
                float(request.regularization) if uses("regularization") else _UNUSED
            ),
            kernel_type=request.kernel_type if uses("kernel_type") else _UNUSED,
            error_message=error_message,
        )

    def list_experiments(self, user_id: int | None = None) -> list[ExperimentRecord]:
        """Fetch all experiment records, optionally filtered by user ID."""
        return self._experiments.list_all(user_id)

    def history(self, user_id: int | None = None) -> list[ExperimentHistoryEntry]:
        """Fetch joined experiment history records including user and evaluation data."""
        return self._experiments.history(user_id)

    def get_experiment(self, experiment_id: int) -> ExperimentRecord | None:
        """Fetch a specific experiment record by ID."""
        return self._experiments.get(experiment_id)

    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete an experiment record by ID (cascading its evaluations)."""
        return self._experiments.delete(experiment_id)

    def get_evaluation(self, experiment_id: int) -> EvaluationRecord | None:
        """Fetch the evaluation record associated with a given experiment ID."""
        return self._evaluations.get_for_experiment(experiment_id)


__all__ = ["CatalogExperimentsMixin"]
