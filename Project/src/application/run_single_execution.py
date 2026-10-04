"""Single-pair transfer learning execution pipeline mixin."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

from threadpoolctl import threadpool_limits

from application.run_pipeline import (
    MAX_COMPUTE_THREADS,
    adapt_and_evaluate,
    ignore_progress,
    load_domains,
    measure_baseline,
    preprocess_domains,
)
from domain.models import TransferLearningRequest, TransferLearningResult
from domain.run_control import RunControl


class RunSingleExecutionMixin:
    """Provides execution pipeline orchestration for a single source-target dataset pair."""

    def execute(
        self, request: TransferLearningRequest, control: RunControl | None = None
    ) -> TransferLearningResult:
        """Run complete transfer learning pipeline for a single experiment request."""
        started_at = datetime.now()
        run_started = perf_counter()
        if control is not None:
            control.checkpoint()

        if request.method_name not in self._methods:
            supported = ", ".join(sorted(self._methods.keys()))
            raise ValueError(f"Unsupported method: {request.method_name}. Supported: {supported}")

        report = control.report if control is not None else ignore_progress

        source_dataset, target_dataset = load_domains(self._loader, request, report)

        with threadpool_limits(limits=MAX_COMPUTE_THREADS):
            prepared = preprocess_domains(
                source_dataset,
                target_dataset,
                request.normalization,
                request.standardization,
                report,
            )
            baseline_metrics = measure_baseline(request.classifier, prepared, report, control)
            return adapt_and_evaluate(
                self._methods[request.method_name],
                request.method_name,
                request,
                prepared,
                baseline_metrics,
                report,
                control,
                started_at,
                run_started,
            )


__all__ = ["RunSingleExecutionMixin"]
