"""CSV dataset format loader implementation."""

from __future__ import annotations

import pandas as pd

from domain.models import Dataset
from infrastructure.data.dataset_loader import DatasetFormatLoader


class CSVDatasetLoader(DatasetFormatLoader):
    """Loads comma-separated tabular datasets (.csv) where the trailing column is the label."""

    extensions = (".csv",)

    def load(self, path: str, is_source: bool = True) -> Dataset:
        """Parse a CSV file and convert it into a domain Dataset model.

        Args:
            path: Path to the CSV file.
            is_source: Flag indicating source or target loading role (unused for CSV).

        Returns:
            Dataset: Loaded domain dataset entity.
        """
        dataframe = pd.read_csv(path)
        return self._dataframe_to_dataset(dataframe)


__all__ = ["CSVDatasetLoader"]
