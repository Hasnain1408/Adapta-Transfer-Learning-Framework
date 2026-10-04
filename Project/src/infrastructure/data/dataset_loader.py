"""Dataset loader interface and composite dispatcher routing by file extension.

Each concrete loader (CSVDatasetLoader, XLSXDatasetLoader, MATDatasetLoader,
ARFFDatasetLoader) handles a specific file format. DatasetFormatLoader defines the
shared base interface, while DatasetLoader acts as the composite dispatcher that
selects the appropriate loader based on the file suffix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from domain.models import Dataset


class DatasetFormatLoader(ABC):
    """Abstract base class for format-specific dataset loaders.

    Subclasses specify claimed lowercase, dot-prefixed file extensions (such as .csv)
    in the extensions tuple.
    """

    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: str, is_source: bool = True) -> Dataset:
        """Load and parse the dataset file into a domain Dataset model.

        Args:
            path: Absolute or relative path to the dataset file.
            is_source: True if loading as the source domain, False for target domain.

        Returns:
            Dataset: Loaded domain entity containing feature matrix and label vector.
        """
        raise NotImplementedError

    @staticmethod
    def _dataframe_to_dataset(dataframe: pd.DataFrame) -> Dataset:
        """Convert a pandas DataFrame into a domain Dataset model.

        Shared conversion logic for tabular datasets (CSV, Excel, ARFF):
        - Treats the trailing column as the ground-truth label vector.
        - Treats all preceding columns as feature attributes.
        - Coerces non-numeric feature entries to numeric floats with NaN replacement.
        - Tracks missing values and column data types.

        Args:
            dataframe: Tabular dataset loaded as a DataFrame.

        Returns:
            Dataset: Normalized domain model.

        Raises:
            ValueError: If the table has fewer than 2 columns.
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
    """Composite dataset loader that delegates parsing to format-specific loaders.

    Serves as the single dataset loading dependency for application use cases.
    New formats can be supported by registering a new DatasetFormatLoader instance
    without altering existing business logic.
    """

    def __init__(self, loaders: list[DatasetFormatLoader]) -> None:
        """Initialize the composite loader with a list of format loaders.

        Args:
            loaders: List of concrete format loader instances.
        """
        self._by_extension: dict[str, DatasetFormatLoader] = {}
        for loader in loaders:
            for extension in loader.extensions:
                self._by_extension[extension] = loader

    def load(self, path: str, is_source: bool = True) -> Dataset:
        """Route the file path to its registered format loader and return the loaded Dataset.

        Args:
            path: Path to the dataset file.
            is_source: True if loading as source domain, False for target domain.

        Returns:
            Dataset: Parsed domain dataset entity.

        Raises:
            ValueError: If the file extension is not supported by any registered loader.
        """
        loader = self._loader_for(path)
        return loader.load(path, is_source=is_source)

    def has_combined_domains(self, path: str) -> bool:
        """Check whether a single dataset file contains both source and target domains.

        Delegates check to the matched format loader if supported (e.g. MATDatasetLoader).
        Returns False for formats requiring separate source and target files.

        Args:
            path: Path to the dataset file.

        Returns:
            bool: True if the file contains combined domain splits, False otherwise.
        """
        extension = Path(path).suffix.lower()
        loader = self._by_extension.get(extension)
        checker = getattr(loader, "has_combined_domains", None)
        return bool(checker(path)) if checker is not None else False

    def _loader_for(self, path: str) -> DatasetFormatLoader:
        """Retrieve the registered format loader matching the file extension."""
        extension = Path(path).suffix.lower()
        loader = self._by_extension.get(extension)
        if loader is None:
            supported = ", ".join(sorted(self._by_extension))
            raise ValueError(
                f"Unsupported dataset format {extension!r}. Use one of: {supported}."
            )
        return loader


__all__ = ["DatasetFormatLoader", "DatasetLoader"]
