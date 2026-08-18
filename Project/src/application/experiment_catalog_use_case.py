"""Recording and querying experiments.

Sits alongside ``RunTransferLearningUseCase``: that one *executes* an
experiment, this one *remembers* it. Keeping them separate means a failure to
reach the database can never prevent a run from completing — the UI records
opportunistically after the fact.

Everything the presentation layer needs from persistence goes through this
class, so no widget imports a repository or touches SQL.

``ExperimentCatalogUseCase`` holds the four repositories and composes one
mixin per entity, so each stays separately readable:

- ``CatalogUsersMixin`` — accounts.
- ``CatalogDatasetsMixin`` — the dataset registry.
- ``CatalogExperimentsMixin`` — recording runs and their scores.
- ``CatalogQueriesMixin`` — read-only queries for comparison and statistics.
"""

from __future__ import annotations

from application.catalog_datasets import CatalogDatasetsMixin
from application.catalog_experiments import CatalogExperimentsMixin
from application.catalog_queries import CatalogQueriesMixin
from application.catalog_users import CatalogUsersMixin
from domain.repositories import (
    DatasetRepository,
    EvaluationRepository,
    ExperimentRepository,
    UserRepository,
)


class ExperimentCatalogUseCase(
    CatalogUsersMixin,
    CatalogDatasetsMixin,
    CatalogExperimentsMixin,
    CatalogQueriesMixin,
):
    """Persist experiment runs and expose the catalogue to the UI."""

    def __init__(
        self,
        users: UserRepository,
        datasets: DatasetRepository,
        experiments: ExperimentRepository,
        evaluations: EvaluationRepository,
    ) -> None:
        self._users = users
        self._datasets = datasets
        self._experiments = experiments
        self._evaluations = evaluations


__all__ = [
    "CatalogDatasetsMixin",
    "CatalogExperimentsMixin",
    "CatalogQueriesMixin",
    "CatalogUsersMixin",
    "ExperimentCatalogUseCase",
]
