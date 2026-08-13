from __future__ import annotations

from pathlib import Path
import numpy as np

import pandas as pd
import scipy.io as sio

from domain.models import Dataset

#: Keys that unambiguously mark a domain as source/target within a .mat file,
#: as opposed to the generic fallback keys ('fea', 'X', 'label', ...) that a
#: single-domain file also matches under *either* role. Only these decide
#: whether a file carries both halves — see ``has_combined_domains``.
_SOURCE_ONLY_X_KEYS = ("X_src", "Xs")
_TARGET_ONLY_X_KEYS = ("X_tar", "Xt", "tar_fea")


class CSVDatasetLoader:
    def has_combined_domains(self, path: str) -> bool:
        """True when *path* alone carries both a source and a target half.

        COIL20 and MNIST<->USPS ship this way: one .mat file with X_src/Y_src
        *and* X_tar/Y_tar side by side, meant to be run without picking a
        second file. Deliberately checks only the source/target-specific key
        names, not the generic fallbacks ``load`` also accepts (`fea`, `X`,
        `label`, ...) — a plain single-domain file matches those under
        *either* ``is_source`` value, which would misreport it as combined.
        """
        if Path(path).suffix.lower() != ".mat":
            return False
        try:
            mat_data = sio.loadmat(path)
        except Exception:
            return False
        has_source = any(key in mat_data for key in _SOURCE_ONLY_X_KEYS)
        has_target = any(key in mat_data for key in _TARGET_ONLY_X_KEYS)
        return has_source and has_target

    def load(self, path: str, is_source: bool = True) -> Dataset:
        file_suffix = Path(path).suffix.lower()

        if file_suffix == ".csv":
            dataframe = pd.read_csv(path)
        elif file_suffix == ".xlsx":
            dataframe = pd.read_excel(path, engine="openpyxl")
        elif file_suffix == ".mat":
            mat_data = sio.loadmat(path)

            if is_source:
                x_keys = [*_SOURCE_ONLY_X_KEYS, 'fea', 'X', 'x']
                y_keys = ['Y_src', 'Ys', 'gnd', 'Y', 'y', 'label', 'labels']
            else:
                x_keys = [*_TARGET_ONLY_X_KEYS, 'X', 'x']
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

            features = features.astype(float)
            return Dataset(
                features=features,
                labels=labels,
                feature_names=[f"feature_{i+1}" for i in range(features.shape[1])],
                label_name="label",
                dtypes=[str(features.dtype)] * features.shape[1],
                missing_value_count=int(np.isnan(features).sum()),
            )
        else:
            raise ValueError("Unsupported dataset format. Use .csv, .xlsx, or .mat files.")

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
