from dataclasses import dataclass, field
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