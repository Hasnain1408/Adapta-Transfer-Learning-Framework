"""Dataset loader interface and the composite that routes by file extension.

Each concrete loader (``CSVDatasetLoader``, ``XLSXDatasetLoader``,
``MATDatasetLoader``, ``ARFFDatasetLoader``) owns exactly one file format.
``DatasetFormatLoader`` is the interface they implement; ``DatasetLoader`` is
the composite that dispatches a path to the right one by its suffix and is
the single object the rest of the app depends on — it is what gets wired
into ``RunTransferLearningUseCase`` and structurally satisfies
``domain.interfaces.DatasetLoader``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from domain.models import Dataset


class DatasetFormatLoader(ABC):
    """Base for a loader that owns exactly one file format.

    ``extensions`` names the lowercase, dot-prefixed suffixes this loader
    claims (e.g. ``(".csv",)``) — ``DatasetLoader`` uses it to route a path
    to the right concrete loader.
    """

    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: str, is_source: bool = True) -> Dataset:
        raise NotImplementedError

    @staticmethod
    def _dataframe_to_dataset(dataframe: pd.DataFrame) -> Dataset:
        """Shared by the tabular formats (CSV, XLSX, ARFF): the last column is
        the label, every other column is coerced to numeric with NaN -> 0.0.
        """
        if dataframe.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns: features + label column.")

        raw_features_df = dataframe.iloc[:, :-1]
        dtypes = [str(dtype) for dtype in raw_features_df.dtypes]
        coerced_df = raw_features_df.apply(pd.to_numeric, errors="coerce")
        missing_value_count = int(coerced_df.isna().sum().sum())
        features_df = coerced_df.fillna(0.0)
        labels_series = dataframe.iloc[:, -1]
        feature_names = [str(column) for column in dataframe.columns[:-1]]
        label_name = str(dataframe.columns[-1])

        return Dataset(
            features=features_df.to_numpy(dtype=float),
            labels=labels_series.to_numpy(),
            feature_names=feature_names,
            label_name=label_name,
            dtypes=dtypes,
            missing_value_count=missing_value_count,
        )


class DatasetLoader:
    """Routes ``load``/``has_combined_domains`` to the loader for a path's extension.

    This is the concrete object wired into ``RunTransferLearningUseCase`` —
    everything above ``infrastructure`` depends only on this, never on a
    specific format loader, so adding a format means adding one
    ``DatasetFormatLoader`` and listing it here rather than touching any
    caller.
    """

    def __init__(self, loaders: list[DatasetFormatLoader]) -> None:
        self._by_extension: dict[str, DatasetFormatLoader] = {}
        for loader in loaders:
            for extension in loader.extensions:
                self._by_extension[extension] = loader

    def load(self, path: str, is_source: bool = True) -> Dataset:
        loader = self._loader_for(path)
        return loader.load(path, is_source=is_source)

    def has_combined_domains(self, path: str) -> bool:
        """True when *path* alone carries both a source and a target half.

        Delegates to the matched loader when it supports the check (only
        ``MATDatasetLoader`` does today); every other format reports
        ``False``, meaning "use two files".
        """
        extension = Path(path).suffix.lower()
        loader = self._by_extension.get(extension)
        checker = getattr(loader, "has_combined_domains", None)
        return bool(checker(path)) if checker is not None else False

    def _loader_for(self, path: str) -> DatasetFormatLoader:
        extension = Path(path).suffix.lower()
        loader = self._by_extension.get(extension)
        if loader is None:
            supported = ", ".join(sorted(self._by_extension))
            raise ValueError(
                f"Unsupported dataset format {extension!r}. Use one of: {supported}."
            )
        return loader


__all__ = ["DatasetFormatLoader", "DatasetLoader"]
