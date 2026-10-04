"""Shared pipeline stages for single-run, batch, and multi-method transfer learning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from application.classification import evaluate_classifier
from application.domain_alignment import measure_domain_alignment
from application.preprocessing import apply_normalization, apply_standardization
from application.run_tunables import apply_tunable_parameters
from domain.models import (
    NEGATIVE_TRANSFER_TOLERANCE,
    Dataset,
    EvaluationMetrics,
    NegativeTransferAnalysis,
    TransferLearningRequest,
    TransferLearningResult,
)
from domain.progress import (
    STAGE_ADAPTING,
    STAGE_ALIGNMENT,
    STAGE_BASELINE,
    STAGE_CLASSIFYING,
    STAGE_FINALIZING,
    STAGE_LOADING,
    STAGE_PREPROCESSING,
)
from domain.run_control import RunCancelled, RunControl

# Upper bound on thread parallelism for BLAS linear algebra computations.
MAX_COMPUTE_THREADS = max(1, (os.cpu_count() or 4) // 2)


def ignore_progress(*_args, **_kwargs) -> None:
    """No-op progress callback used when no RunControl token is provided."""


@dataclass(frozen=True)
class PreparedDomains:
    """Container holding loaded and preprocessed source and target dataset domains."""

    source_dataset: Dataset
    target_dataset: Dataset
    source_features: np.ndarray
    target_features: np.ndarray

    @property
    def is_labelled(self) -> bool:
        """True if both source and target datasets contain ground-truth label vectors."""
        return self.source_dataset.labels is not None and self.target_dataset.labels is not None


def load_domains(loader, request, report) -> tuple[Dataset, Dataset]:
    """Load source and target dataset files from disk."""
    report(
        STAGE_LOADING,
        f"{Path(request.source_path).name} → {Path(request.target_path).name}",
    )
    source_dataset = loader.load(request.source_path, is_source=True)
    target_dataset = loader.load(request.target_path, is_source=False)

    if source_dataset.features.size == 0 or target_dataset.features.size == 0:
        raise ValueError("Source and target datasets must contain at least one feature row.")
    return source_dataset, target_dataset


def preprocess_domains(
    source_dataset: Dataset,
    target_dataset: Dataset,
    normalization: str,
    standardization: str,
    report,
) -> PreparedDomains:
    """Normalize and standardize feature matrices for both domains."""
    report(STAGE_PREPROCESSING, normalization)
    source_features = apply_normalization(source_dataset.features.copy(), normalization)
    target_features = apply_normalization(target_dataset.features.copy(), normalization)

    report(STAGE_PREPROCESSING, standardization)
    source_features = apply_standardization(source_features, standardization)
    target_features = apply_standardization(target_features, standardization)

    return PreparedDomains(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_features=source_features,
        target_features=target_features,
    )


def measure_baseline(
    classifier: str, prepared: PreparedDomains, report, control: RunControl | None = None
) -> EvaluationMetrics | None:
    """Evaluate target performance on preprocessed features without domain adaptation."""
    if not prepared.is_labelled:
        return None
    if control is not None:
        control.checkpoint()
    report(STAGE_BASELINE, f"{classifier} without adaptation")
    try:
        baseline_metrics, _ = evaluate_classifier(
            classifier,
            prepared.source_features,
            prepared.source_dataset.labels,
            prepared.target_features,
            prepared.target_dataset.labels,
        )
    except RunCancelled:
        raise
    except Exception:
        return None
    return baseline_metrics


def adapt_and_evaluate(
    method,
    method_name: str,
    request: TransferLearningRequest,
    prepared: PreparedDomains,
    baseline_metrics: EvaluationMetrics | None,
    report,
    control: RunControl | None,
    started_at: datetime,
    run_started: float,
    paused_before: float = 0.0,
) -> TransferLearningResult:
    """Execute feature adaptation for one method, run classification, and assemble results."""
    apply_tunable_parameters(method, request)

    report(
        STAGE_ADAPTING,
        f"{method_name} on "
        f"{prepared.source_features.shape[0]}+{prepared.target_features.shape[0]} samples",
    )

    adaptation_started = perf_counter()
    paused_before_adaptation = control.paused_seconds if control is not None else 0.0
    transformed_source, transformed_target = method.run(
        prepared.source_features,
        prepared.target_features,
        source_labels=prepared.source_dataset.labels,
        control=control,
    )
    paused_during_adaptation = (
        control.paused_seconds - paused_before_adaptation if control is not None else 0.0
    )
    adaptation_seconds = max(0.0, perf_counter() - adaptation_started - paused_during_adaptation)

    output_feature_names = [
        f"{method_name.lower()}_feature_{index + 1}"
        for index in range(transformed_target.shape[1])
    ]

    metrics = None
    target_predictions = None
    target_actuals = None
    prediction_correctness = None
    negative_transfer = None

    if prepared.is_labelled:
        report(
            STAGE_CLASSIFYING,
            f"{request.classifier} on {transformed_target.shape[0]} target samples",
        )
        metrics, target_predictions = evaluate_classifier(
            request.classifier,
            transformed_source,
            prepared.source_dataset.labels,
            transformed_target,
            prepared.target_dataset.labels,
        )
        target_actuals = prepared.target_dataset.labels
        prediction_correctness = [
            "Right" if p == a else "Wrong" for p, a in zip(target_predictions, target_actuals)
        ]

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
        prepared.source_features,
        prepared.target_features,
        transformed_source,
        transformed_target,
    )

    report(STAGE_FINALIZING, "assembling results")
    paused_total = control.paused_seconds if control is not None else 0.0
    return TransferLearningResult(
        source_dataset=prepared.source_dataset,
        target_dataset=prepared.target_dataset,
        transformed_source=transformed_source,
        transformed_target=transformed_target,
        output_feature_names=output_feature_names,
        metrics=metrics,
        target_predictions=target_predictions,
        target_actuals=target_actuals,
        prediction_correctness=prediction_correctness,
        preprocessed_source=prepared.source_features,
        preprocessed_target=prepared.target_features,
        alignment=alignment,
        negative_transfer=negative_transfer,
        started_at=started_at,
        completed_at=datetime.now(),
        duration_seconds=max(0.0, perf_counter() - run_started - (paused_total - paused_before)),
        adaptation_seconds=adaptation_seconds,
    )


__all__ = [
    "MAX_COMPUTE_THREADS",
    "PreparedDomains",
    "adapt_and_evaluate",
    "ignore_progress",
    "load_domains",
    "measure_baseline",
    "preprocess_domains",
]
