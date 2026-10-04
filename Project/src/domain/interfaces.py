"""Domain protocols and abstract base interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from .models import Dataset
from .run_control import RunControl


class DatasetLoader(Protocol):
    """Protocol for loading raw dataset files into domain Dataset models."""

    def load(self, path: str, is_source: bool = True) -> Dataset:
        """Load and return a Dataset model from the given path."""
        ...


class TransferLearningMethod(ABC):
    """Abstract base class for all transfer learning algorithm implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm identifier name."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_labels: np.ndarray | None = None,
        control: RunControl | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute domain adaptation transformation on source and target features.

        Args:
            source_features: Array of shape (n_source_samples, n_features).
            target_features: Array of shape (n_target_samples, n_features).
            source_labels: Optional ground-truth labels for source samples.
            control: Optional cooperative pause or cancellation token.

        Returns:
            tuple[np.ndarray, np.ndarray]: Adapted source and target feature matrices.
        """
        raise NotImplementedError
