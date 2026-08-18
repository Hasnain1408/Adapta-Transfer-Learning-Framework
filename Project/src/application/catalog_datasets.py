"""The dataset registry: cataloguing files as they get used."""

from __future__ import annotations

from pathlib import Path

from application.catalog_summaries import class_names, dtype_summary
from domain.entities import DatasetFormat, DatasetRecord
from domain.models import Dataset


class CatalogDatasetsMixin:
    """Registering and reading the catalogue of dataset files."""

    def register_dataset(self, path: str, dataset: Dataset) -> DatasetRecord | None:
        """Catalogue the file at *path* using the statistics of the loaded *dataset*.

        Returns ``None`` for a path whose extension the catalogue does not
        model, rather than raising — the run itself already succeeded, so a
        bookkeeping gap must not surface as an error.
        """
        try:
            file_format = DatasetFormat.from_path(path)
        except ValueError:
            return None

        # upsert_by_path dedupes on this exact string, so two spellings of
        # the same file — a relative path from one working directory, an
        # absolute path from another — must resolve to the same key or they
        # register as two "different" datasets. resolve() is best-effort:
        # a path that no longer exists (a stale registration) falls back to
        # the literal string rather than raising.
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
        return self._datasets.list_all()

    def get_dataset(self, dataset_id: int | None) -> DatasetRecord | None:
        """``None`` in, ``None`` out — lets callers pass an experiment's
        possibly-unset ``source_dataset_id``/``target_dataset_id`` directly."""
        if dataset_id is None:
            return None
        return self._datasets.get(dataset_id)

    def delete_dataset(self, dataset_id: int) -> bool:
        return self._datasets.delete(dataset_id)


__all__ = ["CatalogDatasetsMixin"]
