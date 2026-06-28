from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TransferLearningRequest:
    method_name: str
    source_path: str
    target_path: str
  
    
@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass(frozen=True)
class TransferLearningResult:
    source_dataset: "Dataset"
    target_dataset: "Dataset"
    transformed_source: np.ndarray
    transformed_target: np.ndarray
    output_feature_names: list[str]
    metrics: Optional[EvaluationMetrics] = None

@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: Optional[np.ndarray]
    feature_names: list[str]
    label_name: str