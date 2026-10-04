"""Dataset catalogue mixin for registering and retrieving dataset metadata."""

from __future__ import annotations

from pathlib import Path

from application.catalog_summaries import class_names, dtype_summary
from domain.entities import DatasetFormat, DatasetRecord
from domain.models import Dataset


class CatalogDatasetsMixin:
    """Provides dataset metadata registration and querying operations."""

    def register_dataset(self, path: str, dataset: Dataset) -> DatasetRecord | None:
        """Register a dataset file in the catalogue using computed dataset statistics.

        Returns None if the file format is unsupported without raising an exception,
        allowing execution to proceed smoothly.

        Args:
            path: Source file path of the dataset.
            dataset: In-memory Dataset model containing features, labels, and types.

        Returns:
            DatasetRecord | None: Persisted dataset catalogue record or None.
        """
        try:
            file_format = DatasetFormat.from_path(path)
        except ValueError:
            return None

        try:
            canonical_path = str(Path(path).expanduser().resolve())
        except OSError:
            canonical_path = str(path)

        names = class_names(dataset.labels)
        record = DatasetRecord(
            name=Path(path).name,
            file_format=file_format,
            num_samples=int(dataset.features.shape[0]),
            num_features=int(dataset.features.shape[1]),
            feature_names=[str(name) for name in dataset.feature_names],
            label_name=str(dataset.label_name or ""),
            num_classes=len(names),
            class_names=names,
            dtype_summary=dtype_summary(dataset.dtypes),
            missing_value_count=dataset.missing_value_count,
            source_path=canonical_path,
        )
        return self._datasets.upsert_by_path(record)

    def list_datasets(self) -> list[DatasetRecord]:
        """Return all catalogued dataset records."""
        return self._datasets.list_all()

    def get_dataset(self, dataset_id: int | None) -> DatasetRecord | None:
        """Fetch a catalogued dataset record by ID, returning None if dataset_id is None."""
        if dataset_id is None:
            return None
        return self._datasets.get(dataset_id)

    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset record from the catalogue by ID."""
        return self._datasets.delete(dataset_id)


__all__ = ["CatalogDatasetsMixin"]
