"""Orchestrator use case for single, batch, and multi-method transfer learning."""

from __future__ import annotations

from application.run_batch_execution import RunBatchExecutionMixin
from application.run_multi_method_execution import RunMultiMethodExecutionMixin
from application.run_single_execution import RunSingleExecutionMixin
from application.run_tunables import TUNABLE_PARAMETERS, supported_parameters_for
from domain.interfaces import DatasetLoader, TransferLearningMethod


class RunTransferLearningUseCase(
    RunSingleExecutionMixin, RunBatchExecutionMixin, RunMultiMethodExecutionMixin
):
    """Coordinates dataset loading, preprocessing, adaptation algorithms, and evaluation."""

    def __init__(self, loader: DatasetLoader, methods: list[TransferLearningMethod]) -> None:
        """Initialize use case with dataset loader and available transfer learning methods."""
        self._loader = loader
        self._methods = {method.name: method for method in methods}

    @property
    def method_names(self) -> list[str]:
        """Return the list of registered algorithm method names."""
        return list(self._methods)

    def dataset_has_combined_domains(self, path: str) -> bool:
        """Check whether the given dataset file contains combined source and target splits."""
        checker = getattr(self._loader, "has_combined_domains", None)
        return bool(checker(path)) if checker is not None else False

    def supported_parameters(self, method_name: str) -> frozenset[str]:
        """Return the set of tunable parameter names supported by the named method."""
        method = self._methods.get(method_name)
        if method is None:
            return frozenset()
        return supported_parameters_for(method)


__all__ = [
    "TUNABLE_PARAMETERS",
    "RunBatchExecutionMixin",
    "RunMultiMethodExecutionMixin",
    "RunSingleExecutionMixin",
    "RunTransferLearningUseCase",
]
