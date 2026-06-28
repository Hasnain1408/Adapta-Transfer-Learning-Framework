from __future__ import annotations

from pathlib import Path

import pandas as pd

from domain.models import Dataset


class CSVDatasetLoader:
    def load(self, path: str) -> Dataset:
        file_suffix = Path(path).suffix.lower()

        if file_suffix == ".csv":
            dataframe = pd.read_csv(path)
        elif file_suffix == ".xlsx":
            dataframe = pd.read_excel(path, engine="openpyxl")
        else:
            raise ValueError("Unsupported dataset format. Use .csv or .xlsx files.")

        if dataframe.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns: features + label column.")

        features_df = dataframe.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        labels_series = dataframe.iloc[:, -1]

        return Dataset(
            features=features_df.to_numpy(dtype=float),
            labels=labels_series.to_numpy(),
        )
