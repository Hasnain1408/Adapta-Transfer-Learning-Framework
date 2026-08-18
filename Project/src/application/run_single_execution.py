"""Executing one source→target transfer-learning run.

The pipeline, in order: load both datasets, preprocess, inject the method's
tunable parameters, run the method, classify the target, measure a
no-adaptation baseline, measure domain alignment, and assemble the result.
Each step reports itself *before* it starts, so the progress panel names the
thing currently blocking rather than the thing that just finished.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

from threadpoolctl import threadpool_limits

from application.classification import evaluate_classifier
from application.domain_alignment import measure_domain_alignment
from application.preprocessing import apply_normalization, apply_standardization
from application.run_tunables import apply_tunable_parameters
from domain.progress import (
    STAGE_ADAPTING,
    STAGE_ALIGNMENT,
    STAGE_BASELINE,
    STAGE_CLASSIFYING,
    STAGE_FINALIZING,
    STAGE_LOADING,
    STAGE_PREPROCESSING,
)
from domain.models import (
    NEGATIVE_TRANSFER_TOLERANCE,
    NegativeTransferAnalysis,
    TransferLearningRequest,
    TransferLearningResult,
)
from domain.run_control import RunCancelled, RunControl


def _ignore_progress(*_args, **_kwargs) -> None:
    """No-op stand-in when a run was started without a ``RunControl``.

    Lets ``execute`` call ``report(...)`` unconditionally instead of
    guarding every one of the seven call sites with ``if control is not
    None`` — the checks were already the noisiest thing in the method.
    """


#: Cap on how many threads BLAS/OpenMP may use for one run's linear algebra
#: (eigendecompositions, matrix multiplies, the classifier fit/predict).
#: Left uncapped, OpenBLAS defaults to roughly one thread per logical core;
#: measured on a 16-core dev machine, a single MEDA run alone pinned over
#: 900% total CPU. That is enough to starve everything else on the same
#: machine of scheduler time -- including, over a remote session, the
#: process actually rendering the editor window, which then reports itself
#: as "not responding" with no error from this app at all to explain why.
#: Half the cores (floor 1) keeps a run fast while leaving the rest of the
#: system able to breathe.
_MAX_COMPUTE_THREADS = max(1, (os.cpu_count() or 4) // 2)


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
        report = control.report if control is not None else _ignore_progress

        report(
            STAGE_LOADING,
            f"{Path(request.source_path).name} → {Path(request.target_path).name}",
        )
        source_dataset = self._loader.load(request.source_path, is_source=True)
        target_dataset = self._loader.load(request.target_path, is_source=False)

        if source_dataset.features.size == 0 or target_dataset.features.size == 0:
            raise ValueError("Source and target datasets must contain at least one feature row.")

        with threadpool_limits(limits=_MAX_COMPUTE_THREADS):
            report(STAGE_PREPROCESSING, request.normalization)
            source_features = apply_normalization(source_dataset.features.copy(), request.normalization)
            target_features = apply_normalization(target_dataset.features.copy(), request.normalization)

            report(STAGE_PREPROCESSING, request.standardization)
            source_features = apply_standardization(source_features, request.standardization)
            target_features = apply_standardization(target_features, request.standardization)

            method = self._methods[request.method_name]
            apply_tunable_parameters(method, request)

            # A one-shot method (CORAL, GFK, TCA) reports only this; the
            # iterative ones overwrite the detail once per iteration from inside
            # their own loop, via control.report_iteration.
            report(
                STAGE_ADAPTING,
                f"{request.method_name} on "
                f"{source_features.shape[0]}+{target_features.shape[0]} samples",
            )

            # Timed on its own so the cost attributable to the *method* can be
            # read apart from the loading and classification around it.
            adaptation_started = perf_counter()
            paused_before_adaptation = control.paused_seconds if control is not None else 0.0
            transformed_source, transformed_target = method.run(
                source_features,
                target_features,
                source_labels=source_dataset.labels,
                control=control,
            )
            # A pause blocks inside method.run's own checkpoints, so the elapsed
            # wall clock would otherwise count however long the user left it
            # paused as if it were compute.
            paused_during_adaptation = (
                control.paused_seconds - paused_before_adaptation if control is not None else 0.0
            )
            adaptation_seconds = max(
                0.0, perf_counter() - adaptation_started - paused_during_adaptation
            )

            output_feature_names = [
                f"{request.method_name.lower()}_feature_{index + 1}"
                for index in range(transformed_target.shape[1])
            ]

            metrics = None
            target_predictions = None
            target_actuals = None
            prediction_correctness = None
            negative_transfer = None

            if source_dataset.labels is not None and target_dataset.labels is not None:
                report(
                    STAGE_CLASSIFYING,
                    f"{request.classifier} on {transformed_target.shape[0]} target samples",
                )
                metrics, target_predictions = evaluate_classifier(
                    request.classifier,
                    transformed_source,
                    source_dataset.labels,
                    transformed_target,
                    target_dataset.labels,
                )
                target_actuals = target_dataset.labels
                prediction_correctness = [
                    "Right" if p == a else "Wrong"
                    for p, a in zip(target_predictions, target_actuals)
                ]

                # The same classifier over the *un-adapted* features: the score
                # you would have got by skipping domain adaptation entirely.
                # Without this there is nothing to call negative transfer against.
                if control is not None:
                    control.checkpoint()
                report(STAGE_BASELINE, f"{request.classifier} without adaptation")
                try:
                    baseline_metrics, _ = evaluate_classifier(
                        request.classifier,
                        source_features,
                        source_dataset.labels,
                        target_features,
                        target_dataset.labels,
                    )
                except RunCancelled:
                    raise
                except Exception:  # noqa: BLE001 - a missing baseline must not sink the run
                    baseline_metrics = None

                if baseline_metrics is not None:
                    accuracy_delta = metrics.accuracy - baseline_metrics.accuracy
                    negative_transfer = NegativeTransferAnalysis(
                        baseline=baseline_metrics,
                        adapted=metrics,
                        accuracy_delta=accuracy_delta,
                        f1_delta=metrics.f1_score - baseline_metrics.f1_score,
                        is_negative_transfer=accuracy_delta < -NEGATIVE_TRANSFER_TOLERANCE,
                    )

            if control is not None:
                control.checkpoint()
            report(STAGE_ALIGNMENT, "MMD before vs. after adaptation")
            alignment = measure_domain_alignment(
                source_features, target_features, transformed_source, transformed_target
            )

            report(STAGE_FINALIZING, "assembling results")
            return TransferLearningResult(
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                transformed_source=transformed_source,
                transformed_target=transformed_target,
                output_feature_names=output_feature_names,
                metrics=metrics,
                target_predictions=target_predictions,
                target_actuals=target_actuals,
                prediction_correctness=prediction_correctness,
                preprocessed_source=source_features,
                preprocessed_target=target_features,
                alignment=alignment,
                negative_transfer=negative_transfer,
                started_at=started_at,
                completed_at=datetime.now(),
                duration_seconds=max(
                    0.0,
                    perf_counter()
                    - run_started
                    - (control.paused_seconds if control is not None else 0.0),
                ),
                adaptation_seconds=adaptation_seconds,
            )


__all__ = ["RunSingleExecutionMixin"]
