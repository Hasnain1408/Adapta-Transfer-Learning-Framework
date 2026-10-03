from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from domain.ranking import average_metric_ranks


@dataclass(frozen=True)
class TransferLearningRequest:
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
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int
    correct_predictions: int
    wrong_predictions: int
    #: Area under the ROC curve, or ``None`` when it cannot be computed —
    #: a classifier exposing neither ``predict_proba`` nor
    #: ``decision_function``, a single-class target, or scores whose classes
    #: don't line up with the target's. Optional rather than 0.0 because
    #: "not measurable" and "no better than chance" are different claims.
    auc: Optional[float] = None


@dataclass(frozen=True)
class DomainAlignment:
    """How far apart the two domains sit, before vs. after adaptation.

    ``mmd_before`` is measured in the preprocessed input space and
    ``mmd_after`` in the method's output space -- see ``domain/mmd.py`` for
    why those two spaces are comparable in ratio but not in absolute terms.
    """

    mmd_before: float
    mmd_after: float
    #: ``((before - after) / before) * 100``; positive means the gap closed.
    change_percent: float
    #: One of the ``mmd.STATUS_*`` labels, derived from ``change_percent``.
    status: str
    kernel: str = "rbf"

    #: Domain Overlap: how much the two point clouds intermix, 0-100 (see
    #: ``domain/overlap.py``). Optional because a method whose output space
    #: makes the measure unavailable still produces a valid MMD reading,
    #: and because rows persisted before this metric existed have no value
    #: to report -- ``None`` renders as "—", never as 0%.
    overlap_before: Optional[float] = None
    overlap_after: Optional[float] = None

    @property
    def overlap_change_points(self) -> Optional[float]:
        """After minus before, in percentage *points* — not a ratio.

        ``None`` unless both readings exist, so a half-measured run cannot
        render a change that was never computed.
        """
        if self.overlap_before is None or self.overlap_after is None:
            return None
        return self.overlap_after - self.overlap_before


@dataclass(frozen=True)
class NegativeTransferAnalysis:
    """Did adaptation actually help the *task*, not just the distributions?

    ``baseline`` is the same classifier trained on the preprocessed source
    and evaluated on the preprocessed target -- i.e. the score you would get
    by ignoring domain adaptation entirely. Negative transfer is that
    comparison coming out worse, which is a stronger claim than an MMD or a
    scatter plot can make on its own.
    """

    baseline: EvaluationMetrics
    adapted: EvaluationMetrics
    accuracy_delta: float
    f1_delta: float
    #: True when adaptation lost more than ``NEGATIVE_TRANSFER_TOLERANCE``
    #: accuracy against the no-adaptation baseline.
    is_negative_transfer: bool


#: Accuracy may wobble by a hair for reasons unrelated to the method (tie
#: breaking in k-NN, say), so a loss only counts as negative transfer past
#: this margin.
NEGATIVE_TRANSFER_TOLERANCE = 0.005


@dataclass(frozen=True)
class TransferLearningResult:
    source_dataset: "Dataset"
    target_dataset: "Dataset"
    transformed_source: np.ndarray
    transformed_target: np.ndarray
    output_feature_names: list[str]
    metrics: Optional[EvaluationMetrics] = None
    target_predictions: Optional[np.ndarray] = None
    target_actuals: Optional[np.ndarray] = None
    prediction_correctness: Optional[list[str]] = None
    #: Features as they entered ``method.run`` -- normalized and standardized
    #: but not yet adapted. This is the "Before Adaptation" view in the
    #: Visualization tab and the space ``alignment.mmd_before`` is measured
    #: in; keeping it means before/after are the same samples in the same
    #: order, so the two projections are directly comparable.
    preprocessed_source: Optional[np.ndarray] = None
    preprocessed_target: Optional[np.ndarray] = None
    alignment: Optional[DomainAlignment] = None
    negative_transfer: Optional[NegativeTransferAnalysis] = None
    #: When the run itself started and finished — not when a report was
    #: generated from it, which can be much later (or repeatedly). Both are
    #: naive local times, matching what the UI and reports display.
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    #: Wall-clock seconds for the whole run: loading, preprocessing,
    #: adaptation, and both evaluations. Measured with a monotonic clock
    #: rather than subtracting the timestamps above, which would skew if the
    #: system clock stepped mid-run (NTP, DST).
    duration_seconds: Optional[float] = None
    #: Seconds spent inside the method's own ``run()`` alone. This is the
    #: figure to compare methods on: ``duration_seconds`` also carries file
    #: I/O and classifier fitting, which are the same work whichever method
    #: was selected and would flatten the differences between them.
    adaptation_seconds: Optional[float] = None

@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: Optional[np.ndarray]
    feature_names: list[str]
    label_name: str
    #: Pandas dtype of each feature column *before* numeric coercion (e.g.
    #: ``"int64"``, ``"float64"``, ``"object"``), parallel to ``feature_names``.
    #: A ``.mat`` source has no such notion, so this is empty for those.
    dtypes: list[str] = field(default_factory=list)
    #: Count of missing/non-numeric feature cells found at load time — these
    #: get filled with 0.0 before ``features`` is built, so this is the only
    #: record that they were ever missing.
    missing_value_count: int = 0


@dataclass(frozen=True)
class BatchTransferLearningRequest:
    """Same configuration as ``TransferLearningRequest``, run over every

    ordered (source, target) pair drawn from ``file_paths`` — i.e. for
    ``[a, b, c]``: a->b, a->c, b->a, b->c, c->a, c->b.
    """

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
    """Several methods against *one* dataset pair, in a single run.

    The orthogonal axis to ``BatchTransferLearningRequest``: a batch varies
    the datasets and holds the method fixed, this varies the method and
    holds the datasets fixed. Everything except ``method_names`` is shared
    across the methods, which is the point — a comparison is only worth
    reading if the only thing that differed was the method.
    """

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
    """Outcome of one method within a multi-method run."""

    method_name: str
    result: Optional[TransferLearningResult] = None
    error: Optional[str] = None

    @property
    def metrics(self) -> Optional[EvaluationMetrics]:
        return self.result.metrics if self.result is not None else None


@dataclass(frozen=True)
class MultiMethodTransferLearningResult:
    """Every method's outcome on the same pair, plus the one shared baseline.

    ``baseline`` is computed once from the preprocessed features and is
    therefore *the same number* for every method here, not one baseline per
    method that happens to agree. That is what lets the per-method
    ``accuracy_delta`` figures be compared against each other directly.
    """

    source_path: str
    target_path: str
    methods: list[MethodTransferLearningResult]
    baseline: Optional[EvaluationMetrics] = None
    #: True when the run was cancelled partway; ``methods`` then holds only
    #: those that finished before the cancellation landed.
    cancelled: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    #: Wall clock for the whole run, including the loading and preprocessing
    #: shared by every method — so it is well below the sum of what the same
    #: methods would cost run one at a time.
    duration_seconds: Optional[float] = None

    @property
    def evaluated(self) -> list["MethodTransferLearningResult"]:
        """Only the methods that produced metrics, i.e. the comparable ones."""
        return [entry for entry in self.methods if entry.metrics is not None]

    def method_ranks(self) -> dict[str, float]:
        """Mean rank per evaluated method across every metric. Lower is better.

        Not accuracy alone. On the imbalanced datasets this app is usually
        pointed at, a method whose classifier collapses onto the majority
        class scores high on accuracy and zero on precision, recall, and F1
        — ranking on accuracy would crown exactly the worst result in the
        table. Averaging across metrics is the same rule the Comparison tab
        applies to stored experiments, shared from ``domain.ranking``.
        """
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

    def ranked(self) -> list["MethodTransferLearningResult"]:
        """Evaluated methods, best first.

        Mean rank across metrics decides; accuracy breaks a tie, and
        adaptation time breaks that — two methods scoring identically are
        not equally good if one took ten times as long. Time is the method's
        own ``adaptation_seconds``, not the whole run: loading and classifier
        fitting cost the same whichever method was used and would only add
        noise to the ordering.
        """
        ranks = self.method_ranks()

        def sort_key(entry: "MethodTransferLearningResult") -> tuple[float, float, float]:
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
    def best(self) -> Optional["MethodTransferLearningResult"]:
        ranked = self.ranked()
        return ranked[0] if ranked else None


@dataclass(frozen=True)
class PairTransferLearningResult:
    """Outcome of one (source, target) pair within a batch run."""

    source_path: str
    target_path: str
    result: Optional[TransferLearningResult] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class BatchTransferLearningResult:
    pairs: list[PairTransferLearningResult]
    average_metrics: Optional[EvaluationMetrics] = None
    #: True when the batch was cancelled partway through -- ``pairs`` then
    #: holds only the pairs that finished before the cancellation landed.
    cancelled: bool = False
    #: Span of the whole batch, not of any single pair (each pair carries its
    #: own timestamps on its ``TransferLearningResult``).
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    #: Wall clock for the batch as a whole. Not the sum of the pairs': a
    #: failed pair still costs time but contributes no result to sum from.
    duration_seconds: Optional[float] = None