"""Fanning one configuration out over every ordered dataset pair."""

from __future__ import annotations

from datetime import datetime
from itertools import permutations
from time import perf_counter

from domain.progress import STAGE_FINALIZING
from domain.models import (
    BatchTransferLearningRequest,
    BatchTransferLearningResult,
    EvaluationMetrics,
    PairTransferLearningResult,
    TransferLearningRequest,
)
from domain.run_control import RunCancelled, RunControl


class RunBatchExecutionMixin:
    """``execute_batch``: every ordered pair, one shared configuration."""

    def execute_batch(
        self, request: BatchTransferLearningRequest, control: RunControl | None = None
    ) -> BatchTransferLearningResult:
        """Run the same configuration over every ordered (source, target) pair.

        A single bad pair (unreadable file, incompatible shapes, ...) is
        recorded as an error on that pair rather than aborting the whole
        batch, so one broken file doesn't cost every other combination.

        *control*, when given, is checked once per pair (and again inside
        each pair's own ``execute``/``method.run``, for iterative methods).
        A cancellation stops the batch after whichever pair was in flight;
        pairs that already finished are kept and ``cancelled`` is set on the
        returned result rather than raising, so the caller sees a partial
        batch instead of losing it entirely.
        """
        if len(request.file_paths) < 2:
            raise ValueError("Batch mode requires at least two dataset files.")

        started_at = datetime.now()
        batch_started = perf_counter()
        pairs: list[PairTransferLearningResult] = []
        cancelled = False
        ordered_pairs = list(permutations(request.file_paths, 2))
        for position, (source_path, target_path) in enumerate(ordered_pairs, start=1):
            if control is not None:
                try:
                    control.checkpoint()
                except RunCancelled:
                    cancelled = True
                    break
                # Stamped on every update the pair's own execute() emits, so
                # the panel can show "pair 3 of 12 — Adapting" without
                # execute() needing to know it is part of a batch.
                control.set_pair(position, len(ordered_pairs))

            pair_request = TransferLearningRequest(
                method_name=request.method_name,
                source_path=source_path,
                target_path=target_path,
                normalization=request.normalization,
                standardization=request.standardization,
                kernel_type=request.kernel_type,
                regularization=request.regularization,
                latent_dim=request.latent_dim,
                mu=request.mu,
                classifier=request.classifier,
            )
            try:
                result = self.execute(pair_request, control=control)
                pairs.append(
                    PairTransferLearningResult(source_path=source_path, target_path=target_path, result=result)
                )
            except RunCancelled:
                cancelled = True
                break
            except Exception as exc:  # noqa: BLE001 - surfaced per-pair, not raised
                pairs.append(
                    PairTransferLearningResult(source_path=source_path, target_path=target_path, error=str(exc))
                )

        if control is not None:
            control.report(STAGE_FINALIZING, f"averaging {len(pairs)} pair(s)")

        metric_sets = [pair.result.metrics for pair in pairs if pair.result is not None and pair.result.metrics is not None]
        average_metrics = None
        if metric_sets:
            count = len(metric_sets)
            average_metrics = EvaluationMetrics(
                accuracy=sum(m.accuracy for m in metric_sets) / count,
                precision=sum(m.precision for m in metric_sets) / count,
                recall=sum(m.recall for m in metric_sets) / count,
                f1_score=sum(m.f1_score for m in metric_sets) / count,
                total_samples=sum(m.total_samples for m in metric_sets),
                correct_predictions=sum(m.correct_predictions for m in metric_sets),
                wrong_predictions=sum(m.wrong_predictions for m in metric_sets),
            )

        return BatchTransferLearningResult(
            pairs=pairs,
            average_metrics=average_metrics,
            cancelled=cancelled,
            started_at=started_at,
            completed_at=datetime.now(),
            duration_seconds=max(
                0.0,
                perf_counter()
                - batch_started
                - (control.paused_seconds if control is not None else 0.0),
            ),
        )


__all__ = ["RunBatchExecutionMixin"]
