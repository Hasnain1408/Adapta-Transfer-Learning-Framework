from __future__ import annotations

from pathlib import Path
import numpy as np

import pandas as pd
import scipy.io as sio

from domain.models import Dataset


class CSVDatasetLoader:
    def load(self, path: str, is_source: bool = True) -> Dataset:
        file_suffix = Path(path).suffix.lower()

        if file_suffix == ".csv":
            dataframe = pd.read_csv(path)
        elif file_suffix == ".xlsx":
            dataframe = pd.read_excel(path, engine="openpyxl")
        elif file_suffix == ".mat":
            mat_data = sio.loadmat(path)
            
            if is_source:
                x_keys = ['X_src', 'Xs', 'fea', 'X', 'x']
                y_keys = ['Y_src', 'Ys', 'gnd', 'Y', 'y', 'label', 'labels']
            else:
                x_keys = ['X_tar', 'Xt', 'tar_fea', 'X', 'x']
                y_keys = ['Y_tar', 'Yt', 'tar_gnd', 'Y', 'y', 'label', 'labels']
            
            features = None
            labels = None
            
            for k in x_keys:
                if k in mat_data:
                    features = mat_data[k]
                    break
            
            for k in y_keys:
                if k in mat_data:
                    labels = mat_data[k]
                    break
            
            if features is None:
                arrays = {k: v for k, v in mat_data.items() if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim == 2}
                if bool(arrays):
                    features = sorted(arrays.values(), key=lambda x: x.size)[-1]
            
            if features is None:
                raise ValueError("Could not find feature matrix in the given .mat file.")
                
            if labels is None:
                arrays = {k: v for k, v in mat_data.items() if not k.startswith('_') and isinstance(v, np.ndarray) and min(v.shape) == 1}
                if bool(arrays):
                    labels = sorted(arrays.values(), key=lambda x: max(x.shape))[-1]

            if labels is None:
                raise ValueError("Could not find label vector in the given .mat file.")
                
            labels = labels.flatten()
            
            if features.shape[0] == labels.shape[0]:
                pass 
            elif features.shape[1] == labels.shape[0]:
                features = features.T 
            else:
                raise ValueError("Feature dimensions do not match label dimensions.")

            return Dataset(
                features=features.astype(float),
                labels=labels,
                feature_names=[f"feature_{i+1}" for i in range(features.shape[1])],
                label_name="label",
            )
        else:
            raise ValueError("Unsupported dataset format. Use .csv, .xlsx, or .mat files.")

        if dataframe.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns: features + label column.")

        features_df = dataframe.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        labels_series = dataframe.iloc[:, -1]
        feature_names = [str(column) for column in dataframe.columns[:-1]]
        label_name = str(dataframe.columns[-1])

        return Dataset(
            features=features_df.to_numpy(dtype=float),
            labels=labels_series.to_numpy(),
            feature_names=feature_names,
            label_name=label_name,
        )
