"""Excel spreadsheet dataset format loader implementation."""

from __future__ import annotations

import pandas as pd

from domain.models import Dataset
from infrastructure.data.dataset_loader import DatasetFormatLoader


class XLSXDatasetLoader(DatasetFormatLoader):
    """Loads Excel spreadsheet datasets (.xlsx) where the trailing column is the label."""

    extensions = (".xlsx",)

    def load(self, path: str, is_source: bool = True) -> Dataset:
        """Parse an Excel spreadsheet file and convert it into a domain Dataset model.

        Args:
            path: Path to the Excel file.
            is_source: Flag indicating source or target loading role (unused for Excel).

        Returns:
            Dataset: Loaded domain dataset entity.
        """
        dataframe = pd.read_excel(path, engine="openpyxl")
        return self._dataframe_to_dataset(dataframe)


__all__ = ["XLSXDatasetLoader"]
