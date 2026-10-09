from __future__ import annotations

import pandas as pd
from scipy.io import arff

from domain.models import Dataset
from infrastructure.data.dataset_loader import DatasetFormatLoader


class ARFFDatasetLoader(DatasetFormatLoader):
    """Loads a Weka ``.arff`` file where the last attribute is the label.

    ``scipy.io.arff`` returns nominal/string attributes as ``bytes`` (a
    quirk of its C-string origins), so those columns are decoded to ``str``
    before handing off to the same tabular conversion CSV/XLSX use.
    """

    extensions = (".arff",)

    def load(self, path: str, is_source: bool = True) -> Dataset:
        data, _meta = arff.loadarff(path)
        dataframe = pd.DataFrame(data)
        for column in dataframe.columns:
            if dataframe[column].dtype == object:
                dataframe[column] = dataframe[column].apply(
                    lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
                )
        return self._dataframe_to_dataset(dataframe)


__all__ = ["ARFFDatasetLoader"]
