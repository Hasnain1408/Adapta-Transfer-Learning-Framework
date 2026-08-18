from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from domain.models import Dataset
from infrastructure.data.dataset_loader import DatasetFormatLoader

#: Keys that unambiguously mark a domain as source/target within a .mat file,
#: as opposed to the generic fallback keys ('fea', 'X', 'label', ...) that a
#: single-domain file also matches under *either* role. Only these decide
#: whether a file carries both halves — see ``has_combined_domains``.
_SOURCE_ONLY_X_KEYS = ("X_src", "Xs")
_TARGET_ONLY_X_KEYS = ("X_tar", "Xt", "tar_fea")


class MATDatasetLoader(DatasetFormatLoader):
    """Loads a MATLAB ``.mat`` file, sniffing common source/target key names."""

    extensions = (".mat",)

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
        mat_data = sio.loadmat(path)

        if is_source:
            x_keys = [*_SOURCE_ONLY_X_KEYS, "fea", "X", "x"]
            y_keys = ["Y_src", "Ys", "gnd", "Y", "y", "label", "labels"]
        else:
            x_keys = [*_TARGET_ONLY_X_KEYS, "X", "x"]
            y_keys = ["Y_tar", "Yt", "tar_gnd", "Y", "y", "label", "labels"]

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
            arrays = {
                k: v
                for k, v in mat_data.items()
                if not k.startswith("_") and isinstance(v, np.ndarray) and v.ndim == 2
            }
            if bool(arrays):
                features = sorted(arrays.values(), key=lambda x: x.size)[-1]

        if features is None:
            raise ValueError("Could not find feature matrix in the given .mat file.")

        if labels is None:
            arrays = {
                k: v
                for k, v in mat_data.items()
                if not k.startswith("_") and isinstance(v, np.ndarray) and min(v.shape) == 1
            }
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
            feature_names=[f"feature_{i + 1}" for i in range(features.shape[1])],
            label_name="label",
            dtypes=[str(features.dtype)] * features.shape[1],
            missing_value_count=int(np.isnan(features).sum()),
        )


__all__ = ["MATDatasetLoader"]
