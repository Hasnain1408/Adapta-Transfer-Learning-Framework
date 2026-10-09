from __future__ import annotations

import pandas as pd

from domain.models import Dataset
from infrastructure.data.dataset_loader import DatasetFormatLoader


class CSVDatasetLoader(DatasetFormatLoader):
    """Loads a ``.csv`` file where the last column is the label."""

    extensions = (".csv",)

    def load(self, path: str, is_source: bool = True) -> Dataset:
        dataframe = pd.read_csv(path)
        return self._dataframe_to_dataset(dataframe)


__all__ = ["CSVDatasetLoader"]
