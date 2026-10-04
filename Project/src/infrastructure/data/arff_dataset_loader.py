"""Weka ARFF dataset format loader implementation."""

from __future__ import annotations

import pandas as pd
from scipy.io import arff

from domain.models import Dataset
from infrastructure.data.dataset_loader import DatasetFormatLoader


class ARFFDatasetLoader(DatasetFormatLoader):
    """Loads Weka attribute-relation dataset files (.arff) where the trailing attribute is the label.

    Decodes byte strings produced by scipy.io.arff for nominal or textual attributes
    into standard strings prior to tabular conversion.
    """

    extensions = (".arff",)

    def load(self, path: str, is_source: bool = True) -> Dataset:
        """Parse an ARFF file and convert it into a domain Dataset model.

        Args:
            path: Path to the .arff file.
            is_source: Flag indicating source or target loading role (unused for ARFF).

        Returns:
            Dataset: Loaded domain dataset entity.
        """
        data, _meta = arff.loadarff(path)
        dataframe = pd.DataFrame(data)
        for column in dataframe.columns:
            if dataframe[column].dtype == object:
                dataframe[column] = dataframe[column].apply(
                    lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
                )
        return self._dataframe_to_dataset(dataframe)


__all__ = ["ARFFDatasetLoader"]
