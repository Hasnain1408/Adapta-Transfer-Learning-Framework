"""Executing one source→target transfer-learning run.

The pipeline, in order: load both datasets, preprocess, measure a
no-adaptation baseline, inject the method's tunable parameters, run the
method, classify the target, measure domain alignment, and assemble the
result. Each step reports itself *before* it starts, so the progress panel
names the thing currently blocking rather than the thing that just finished.

The steps themselves live in ``run_pipeline.py`` because the multi-method
path shares them — see that module for why the split exists. This file is
the one-method composition of them.
"""

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
    """The ``execute`` pipeline for a single dataset pair."""

    def execute(
        self, request: TransferLearningRequest, control: RunControl | None = None
    ) -> TransferLearningResult:
        started_at = datetime.now()
        run_started = perf_counter()
        if control is not None:
            control.checkpoint()

        if request.method_name not in self._methods:
            supported = ", ".join(sorted(self._methods.keys()))
            raise ValueError(f"Unsupported method: {request.method_name}. Supported: {supported}")

        # Reported before each step rather than after, so the panel names the
        # thing currently blocking rather than the thing that just finished.
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
