"""PostgreSQL implementation of ``DatasetRepository``."""

from __future__ import annotations

from domain.entities import DatasetFormat, DatasetRecord

from .database import Database
from .postgres_mappers import _is_unique_violation, _to_dataset


class PostgresDatasetRepository:
    """CRUD for ``datasets``, the catalogue of files that have been used."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def add(self, dataset: DatasetRecord) -> DatasetRecord:
        try:
            row = self._db.execute_returning(
                """
                INSERT INTO datasets (
                    name, file_format, num_samples, num_features,
                    feature_names, label_name, num_classes, class_names,
                    dtype_summary, missing_value_count, source_path
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    dataset.name,
                    DatasetFormat(dataset.file_format).value,
                    dataset.num_samples,
                    dataset.num_features,
                    list(dataset.feature_names),
                    dataset.label_name,
                    dataset.num_classes,
                    list(dataset.class_names),
                    dataset.dtype_summary,
                    dataset.missing_value_count,
                    dataset.source_path,
                ),
            )
        except Exception as exc:
            if _is_unique_violation(exc):
                raise ValueError(
                    f"A dataset is already registered for {dataset.source_path!r}."
                ) from exc
            raise
        assert row is not None
        return _to_dataset(row)

    def upsert_by_path(self, dataset: DatasetRecord) -> DatasetRecord:
        """Insert, or refresh the row that already claims this ``source_path``.

        Uses ``ON CONFLICT`` so the read and the write are one atomic
        statement — checking for an existing row first and then inserting
        would race against a concurrent run of the same file.
        """
        row = self._db.execute_returning(
            """
            INSERT INTO datasets (
                name, file_format, num_samples, num_features,
                feature_names, label_name, num_classes, class_names,
                dtype_summary, missing_value_count, source_path
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_path) DO UPDATE SET
                name                 = EXCLUDED.name,
                file_format          = EXCLUDED.file_format,
                num_samples          = EXCLUDED.num_samples,
                num_features         = EXCLUDED.num_features,
                feature_names        = EXCLUDED.feature_names,
                label_name           = EXCLUDED.label_name,
                num_classes          = EXCLUDED.num_classes,
                class_names          = EXCLUDED.class_names,
                dtype_summary        = EXCLUDED.dtype_summary,
                missing_value_count  = EXCLUDED.missing_value_count
            RETURNING *
            """,
            (
                dataset.name,
                DatasetFormat(dataset.file_format).value,
                dataset.num_samples,
                dataset.num_features,
                list(dataset.feature_names),
                dataset.label_name,
                dataset.num_classes,
                list(dataset.class_names),
                dataset.dtype_summary,
                dataset.missing_value_count,
                dataset.source_path,
            ),
        )
        assert row is not None
        return _to_dataset(row)

    def update(self, dataset: DatasetRecord) -> DatasetRecord:
        if dataset.id is None:
            raise ValueError("Cannot update a dataset that has no id.")
        row = self._db.execute_returning(
            """
            UPDATE datasets
               SET name = %s, file_format = %s, num_samples = %s, num_features = %s,
                   feature_names = %s, label_name = %s, num_classes = %s,
                   class_names = %s, dtype_summary = %s, missing_value_count = %s,
                   source_path = %s
             WHERE id = %s
            RETURNING *
            """,
            (
                dataset.name,
                DatasetFormat(dataset.file_format).value,
                dataset.num_samples,
                dataset.num_features,
                list(dataset.feature_names),
                dataset.label_name,
                dataset.num_classes,
                list(dataset.class_names),
                dataset.dtype_summary,
                dataset.missing_value_count,
                dataset.source_path,
                dataset.id,
            ),
        )
        if row is None:
            raise ValueError(f"No dataset with id {dataset.id}.")
        return _to_dataset(row)

    def delete(self, dataset_id: int) -> bool:
        # experiments.*_dataset_id is ON DELETE SET NULL: the experiment keeps
        # its denormalised domain names and only loses the catalogue link.
        return self._db.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,)) > 0

    def get(self, dataset_id: int) -> DatasetRecord | None:
        row = self._db.fetch_one("SELECT * FROM datasets WHERE id = %s", (dataset_id,))
        return _to_dataset(row) if row else None

    def get_by_path(self, source_path: str) -> DatasetRecord | None:
        row = self._db.fetch_one(
            "SELECT * FROM datasets WHERE source_path = %s", (source_path,)
        )
        return _to_dataset(row) if row else None

    def list_all(self) -> list[DatasetRecord]:
        rows = self._db.fetch_all("SELECT * FROM datasets ORDER BY created_at DESC, id DESC")
        return [_to_dataset(row) for row in rows]

    def count(self) -> int:
        row = self._db.fetch_one("SELECT count(*) AS total FROM datasets")
        return int(row["total"]) if row else 0


__all__ = ["PostgresDatasetRepository"]
