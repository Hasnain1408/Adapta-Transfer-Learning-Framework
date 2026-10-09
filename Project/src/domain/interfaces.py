from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from .models import Dataset
from .run_control import RunControl


class DatasetLoader(Protocol):
    def load(self, path: str, is_source: bool = True) -> Dataset:
        ...


class TransferLearningMethod(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_labels: np.ndarray | None = None,
        control: RunControl | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """*control*, when given, is a cooperative pause/cancel token: call
        ``control.checkpoint()`` at the top of the method and, for iterative
        methods, once per iteration -- see ``RunControl`` for what that does
        and why it can only take effect between steps, not mid-computation.
        """
        raise NotImplementedError

