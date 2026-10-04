"""Unified experiment catalogue use case combining users, datasets, and history queries."""

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
    """Facade use case coordinating persistence, registration, and reporting queries."""

    def __init__(
        self,
        users: UserRepository,
        datasets: DatasetRepository,
        experiments: ExperimentRepository,
        evaluations: EvaluationRepository,
    ) -> None:
        """Initialize the catalog use case with injected repository dependencies."""
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
