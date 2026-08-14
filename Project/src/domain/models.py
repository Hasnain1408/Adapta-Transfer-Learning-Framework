from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


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