"""Multi-method transfer learning execution pipeline on a shared dataset pair."""

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
from domain.models import (
    MethodTransferLearningResult,
    MultiMethodTransferLearningRequest,
    MultiMethodTransferLearningResult,
    TransferLearningRequest,
)
from domain.progress import STAGE_FINALIZING
from domain.run_control import RunCancelled, RunControl

# Label used for multi-method progress counter annotations.
UNIT_LABEL = "Method"


class RunMultiMethodExecutionMixin:
    """Provides multi-method execution pipeline fanning out over algorithms on a single pair."""

    def execute_multi_method(
        self,
        request: MultiMethodTransferLearningRequest,
        control: RunControl | None = None,
    ) -> MultiMethodTransferLearningResult:
        """Run multiple transfer learning algorithms over a shared preprocessed dataset pair."""
        if not request.method_names:
            raise ValueError("Select at least one method to run.")

        unknown = [name for name in request.method_names if name not in self._methods]
        if unknown:
            supported = ", ".join(sorted(self._methods.keys()))
            raise ValueError(
                f"Unsupported method(s): {', '.join(unknown)}. Supported: {supported}"
            )

        started_at = datetime.now()
        run_started = perf_counter()
        report = control.report if control is not None else ignore_progress
        if control is not None:
            control.checkpoint()

        probe_request = TransferLearningRequest(
            method_name=request.method_names[0],
            source_path=request.source_path,
            target_path=request.target_path,
        )
        source_dataset, target_dataset = load_domains(self._loader, probe_request, report)

        entries: list[MethodTransferLearningResult] = []
        cancelled = False

        with threadpool_limits(limits=MAX_COMPUTE_THREADS):
            prepared = preprocess_domains(
                source_dataset,
                target_dataset,
                request.normalization,
                request.standardization,
                report,
            )
            baseline_metrics = measure_baseline(request.classifier, prepared, report, control)

            total = len(request.method_names)
            for position, method_name in enumerate(request.method_names, start=1):
                if control is not None:
                    try:
                        control.checkpoint()
                    except RunCancelled:
                        cancelled = True
                        break
                    control.set_pair(position, total, UNIT_LABEL)

                method_request = self._method_request(request, method_name)
                method_started_at = datetime.now()
                method_started = perf_counter()
                paused_before = control.paused_seconds if control is not None else 0.0
                try:
                    result = adapt_and_evaluate(
                        self._methods[method_name],
                        method_name,
                        method_request,
                        prepared,
                        baseline_metrics,
                        report,
                        control,
                        method_started_at,
                        method_started,
                        paused_before,
                    )
                    entries.append(
                        MethodTransferLearningResult(method_name=method_name, result=result)
                    )
                except RunCancelled:
                    cancelled = True
                    break
                except Exception as exc:
                    entries.append(
                        MethodTransferLearningResult(method_name=method_name, error=str(exc))
                    )

        if control is not None:
            control.report(STAGE_FINALIZING, f"comparing {len(entries)} method(s)")

        return MultiMethodTransferLearningResult(
            source_path=request.source_path,
            target_path=request.target_path,
            methods=entries,
            baseline=baseline_metrics,
            cancelled=cancelled,
            started_at=started_at,
            completed_at=datetime.now(),
            duration_seconds=max(
                0.0,
                perf_counter()
                - run_started
                - (control.paused_seconds if control is not None else 0.0),
            ),
        )

    @staticmethod
    def _method_request(
        request: MultiMethodTransferLearningRequest, method_name: str
    ) -> TransferLearningRequest:
        """Create an individual TransferLearningRequest corresponding to a specific method."""
        return TransferLearningRequest(
            method_name=method_name,
            source_path=request.source_path,
            target_path=request.target_path,
            normalization=request.normalization,
            standardization=request.standardization,
            kernel_type=request.kernel_type,
            regularization=request.regularization,
            latent_dim=request.latent_dim,
            mu=request.mu,
            classifier=request.classifier,
        )


__all__ = ["UNIT_LABEL", "RunMultiMethodExecutionMixin"]
