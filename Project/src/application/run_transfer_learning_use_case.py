"""``RunTransferLearningUseCase`` — orchestrates one run or a batch of runs.

The work is split across sibling modules, one per concern:

- ``run_single_execution.py`` — the ``execute`` pipeline for one pair.
- ``run_batch_execution.py`` — ``execute_batch``, fanning it over every pair.
- ``run_tunables.py`` — which request fields a method actually accepts.
- ``preprocessing.py`` — normalization/standardization transforms.
- ``classification.py`` — classifier factory + fit/score evaluation.
- ``domain_alignment.py`` — MMD/overlap measurement before vs. after.

This file is left holding the wiring: the loader and the method registry,
and the small accessors the UI asks about before starting anything.
"""

from __future__ import annotations

from application.run_batch_execution import RunBatchExecutionMixin
from application.run_single_execution import RunSingleExecutionMixin
from application.run_tunables import TUNABLE_PARAMETERS, supported_parameters_for
from domain.interfaces import DatasetLoader, TransferLearningMethod


class RunTransferLearningUseCase(RunSingleExecutionMixin, RunBatchExecutionMixin):
    """Runs a transfer-learning method over a source/target dataset pair."""

    def __init__(self, loader: DatasetLoader, methods: list[TransferLearningMethod]) -> None:
        self._loader = loader
        self._methods = {method.name: method for method in methods}

    @property
    def method_names(self) -> list[str]:
        """Registered method names, in the order they were configured."""
        return list(self._methods)

    def dataset_has_combined_domains(self, path: str) -> bool:
        """Whether *path* alone carries both a source and a target half.

        Delegates to the loader when it supports the check (only
        ``MATDatasetLoader`` does today); other ``DatasetLoader``
        implementations simply report ``False``, meaning "use two files".
        """
        checker = getattr(self._loader, "has_combined_domains", None)
        return bool(checker(path)) if checker is not None else False

    def supported_parameters(self, method_name: str) -> frozenset[str]:
        """Which tunable parameters the named method actually consumes.

        Mirrors the ``hasattr`` rule ``execute`` uses to inject them, so the UI
        can disable controls that would otherwise be silently ignored.
        """
        method = self._methods.get(method_name)
        if method is None:
            return frozenset()
        return supported_parameters_for(method)


__all__ = [
    "TUNABLE_PARAMETERS",
    "RunBatchExecutionMixin",
    "RunSingleExecutionMixin",
    "RunTransferLearningUseCase",
]
