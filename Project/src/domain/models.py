"""Domain execution models, request payloads, and evaluation outcome dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from domain.ranking import average_metric_ranks


@dataclass(frozen=True)
class TransferLearningRequest:
    """Execution parameters for a single transfer learning experiment run."""

    method_name: str
    source_path: str
    target_path: str
    normalization: str = "None"
    standardization: str = "None"
    kernel_type: str = "linear"
    regularization: float = 1.0
    latent_dim: int = 10
    mu: float = 1.0
    classifier: str = "k-NN"


@dataclass(frozen=True)
class EvaluationMetrics:
    """Classification performance metrics computed on the target domain."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int
    correct_predictions: int
    wrong_predictions: int
    # Area under the ROC curve, or None when probabilities are unavailable.
    auc: Optional[float] = None


@dataclass(frozen=True)
class DomainAlignment:
    """Measurement of statistical divergence between source and target domains."""

    mmd_before: float
    mmd_after: float
    # ((before - after) / before) * 100; positive values indicate reduced divergence.
    change_percent: float
    status: str
    kernel: str = "rbf"
    # Domain Overlap metrics (percentage of mutual neighbourhood intermixing).
    overlap_before: Optional[float] = None
    overlap_after: Optional[float] = None

    @property
    def overlap_change_points(self) -> Optional[float]:
        """Change in overlap percentage points (after - before)."""
        if self.overlap_before is None or self.overlap_after is None:
            return None
        return self.overlap_after - self.overlap_before


@dataclass(frozen=True)
class NegativeTransferAnalysis:
    """Diagnostic comparison between domain-adapted model and unadapted baseline."""

    baseline: EvaluationMetrics
    adapted: EvaluationMetrics
    accuracy_delta: float
    f1_delta: float
    is_negative_transfer: bool


# Accuracy drop threshold beyond which adaptation is flagged as negative transfer.
NEGATIVE_TRANSFER_TOLERANCE = 0.005


@dataclass(frozen=True)
class Dataset:
    """In-memory dataset container holding feature matrix and label vectors."""

    features: np.ndarray
    labels: Optional[np.ndarray]
    feature_names: list[str]
    label_name: str
    dtypes: list[str] = field(default_factory=list)
    missing_value_count: int = 0


@dataclass(frozen=True)
class TransferLearningResult:
    """Comprehensive outcome payload of a single transfer learning execution."""

    source_dataset: Dataset
    target_dataset: Dataset
    transformed_source: np.ndarray
    transformed_target: np.ndarray
    output_feature_names: list[str]
    metrics: Optional[EvaluationMetrics] = None
    target_predictions: Optional[np.ndarray] = None
    target_actuals: Optional[np.ndarray] = None
    prediction_correctness: Optional[list[str]] = None
    preprocessed_source: Optional[np.ndarray] = None
    preprocessed_target: Optional[np.ndarray] = None
    alignment: Optional[DomainAlignment] = None
    negative_transfer: Optional[NegativeTransferAnalysis] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    adaptation_seconds: Optional[float] = None


@dataclass(frozen=True)
class BatchTransferLearningRequest:
    """Execution parameters for pairwise batch transfer learning across multiple datasets."""

    method_name: str
    file_paths: list[str]
    normalization: str = "None"
    standardization: str = "None"
    kernel_type: str = "linear"
    regularization: float = 1.0
    latent_dim: int = 10
    mu: float = 1.0
    classifier: str = "k-NN"


@dataclass(frozen=True)
class MultiMethodTransferLearningRequest:
    """Execution parameters for comparing multiple methods across a single dataset pair."""

    method_names: list[str]
    source_path: str
    target_path: str
    normalization: str = "None"
    standardization: str = "None"
    kernel_type: str = "linear"
    regularization: float = 1.0
    latent_dim: int = 10
    mu: float = 1.0
    classifier: str = "k-NN"


@dataclass(frozen=True)
class MethodTransferLearningResult:
    """Result payload for a single method within a multi-method comparison run."""

    method_name: str
    result: Optional[TransferLearningResult] = None
    error: Optional[str] = None

    @property
    def metrics(self) -> Optional[EvaluationMetrics]:
        """Evaluation metrics produced by this method run."""
        return self.result.metrics if self.result is not None else None


@dataclass(frozen=True)
class MultiMethodTransferLearningResult:
    """Aggregated outcome of multi-method benchmark execution on a shared dataset pair."""

    source_path: str
    target_path: str
    methods: list[MethodTransferLearningResult]
    baseline: Optional[EvaluationMetrics] = None
    cancelled: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    @property
    def evaluated(self) -> list[MethodTransferLearningResult]:
        """List of successfully evaluated method results."""
        return [entry for entry in self.methods if entry.metrics is not None]

    def method_ranks(self) -> dict[str, float]:
        """Compute mean competition rank across metrics for each evaluated method."""
        evaluated = self.evaluated
        metric_rows = [
            {
                "accuracy": entry.metrics.accuracy,
                "precision": entry.metrics.precision,
                "recall": entry.metrics.recall,
                "f1_score": entry.metrics.f1_score,
                "auc": entry.metrics.auc,
            }
            for entry in evaluated
        ]
        averages = average_metric_ranks(metric_rows)
        return {
            entry.method_name: average
            for entry, average in zip(evaluated, averages)
            if average is not None
        }

    def ranked(self) -> list[MethodTransferLearningResult]:
        """Return evaluated methods sorted from best to worst by rank, accuracy, and runtime."""
        ranks = self.method_ranks()

        def sort_key(entry: MethodTransferLearningResult) -> tuple[float, float, float]:
            seconds = (
                entry.result.adaptation_seconds
                if entry.result is not None and entry.result.adaptation_seconds is not None
                else float("inf")
            )
            return (
                ranks.get(entry.method_name, float("inf")),
                -entry.metrics.accuracy,
                seconds,
            )

        return sorted(self.evaluated, key=sort_key)

    @property
    def best(self) -> Optional[MethodTransferLearningResult]:
        """Return the highest-ranked method result."""
        ranked = self.ranked()
        return ranked[0] if ranked else None


@dataclass(frozen=True)
class PairTransferLearningResult:
    """Outcome payload for one source-target pair in a batch execution."""

    source_path: str
    target_path: str
    result: Optional[TransferLearningResult] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class BatchTransferLearningResult:
    """Aggregated outcome of a pairwise batch transfer learning run."""

    pairs: list[PairTransferLearningResult]
    average_metrics: Optional[EvaluationMetrics] = None
    cancelled: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None