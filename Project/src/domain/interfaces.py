from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from .models import Dataset


class DatasetLoader(Protocol):
    def load(self, path: str) -> Dataset:
        ...


class TransferLearningMethod(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self, source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
        raise NotImplementedError
