"""Running several methods against one dataset pair, in one experiment.

The orthogonal loop to ``execute_batch``: that one holds the method fixed
and varies the datasets, this holds the datasets fixed and varies the
method. Both isolate a failure to the item that caused it, so one method
that cannot handle the data does not cost the comparison the others.

The whole point is that the methods are comparable, which means everything
except the method must be identical between them. So the shared head of the
pipeline — load, preprocess, baseline — runs exactly once and every method
is handed the same ``PreparedDomains`` instance. Looping ``execute`` per
method instead would recompute all of that N times: slower, and it would
leave each method scored against its own separately-measured baseline.
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
from domain.models import (
    MethodTransferLearningResult,
    MultiMethodTransferLearningRequest,
    MultiMethodTransferLearningResult,
    TransferLearningRequest,
)
from domain.progress import STAGE_FINALIZING
from domain.run_control import RunCancelled, RunControl

#: What ``set_pair`` counts here, so the status strip reads "Method 2 of 5"
#: rather than borrowing the batch path's "Pair".
UNIT_LABEL = "Method"


class RunMultiMethodExecutionMixin:
    """``execute_multi_method``: one pair, several methods, one preparation."""

    def execute_multi_method(
        self,
        request: MultiMethodTransferLearningRequest,
        control: RunControl | None = None,
    ) -> MultiMethodTransferLearningResult:
        """Run every method in *request* over the same source/target pair.

        A method that fails (unsupported parameters, a singular matrix, an
        unlucky eigendecomposition) is recorded as an error against that
        method and the rest still run — the comparison is worth more with a
        gap in it than not at all.

        Cancellation stops after whichever method was in flight; the ones
        that already finished are kept and ``cancelled`` is set on the
        result rather than raising, so a partial comparison survives.
        """
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

        # The loading/preprocessing/baseline stages belong to no single
        # method, so they are reported without a method index — the strip
        # shows them once, before the per-method counter starts moving.
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
            # Measured once, from features no method has touched. Every
            # method's negative-transfer verdict below is against this one
            # number, which is what makes those verdicts comparable.
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
                except Exception as exc:  # noqa: BLE001 - surfaced per method, not raised
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
        """The shared configuration, addressed to one method.

        Also what gets persisted for that method, so each row in History
        reads exactly as if the method had been run on its own.
        """
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
