from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TransferLearningRequest:
    method_name: str
    source_path: str
    target_path: str
  
    
@dataclass(frozen=True)
class TransferLearningResult:
    transformed_target: np.ndarray
    output_feature_names: list[str]

@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: Optional[np.ndarray]