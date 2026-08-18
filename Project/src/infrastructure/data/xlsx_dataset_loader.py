from __future__ import annotations

import pandas as pd

from domain.models import Dataset
from infrastructure.data.dataset_loader import DatasetFormatLoader


class XLSXDatasetLoader(DatasetFormatLoader):
    """Loads a ``.xlsx`` file where the last column is the label."""

    extensions = (".xlsx",)

    def load(self, path: str, is_source: bool = True) -> Dataset:
        dataframe = pd.read_excel(path, engine="openpyxl")
        return self._dataframe_to_dataset(dataframe)


__all__ = ["XLSXDatasetLoader"]
