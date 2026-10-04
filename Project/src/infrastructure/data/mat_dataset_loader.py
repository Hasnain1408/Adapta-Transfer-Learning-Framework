"""MATLAB matrix dataset format loader implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from domain.models import Dataset
from infrastructure.data.dataset_loader import DatasetFormatLoader

# Keys that unambiguously mark a domain as source or target within a .mat file,
# distinguishing dual-domain files from single-domain files that use generic key names.
_SOURCE_ONLY_X_KEYS = ("X_src", "Xs")
_TARGET_ONLY_X_KEYS = ("X_tar", "Xt", "tar_fea")


class MATDatasetLoader(DatasetFormatLoader):
    """Loads MATLAB matrix files (.mat), detecting domain keys and matrix orientations."""

    extensions = (".mat",)

    def has_combined_domains(self, path: str) -> bool:
        """Check whether the MATLAB file contains both source and target domains.

        Recognizes benchmark datasets (such as COIL20 or MNIST-USPS) packaged with
        source and target feature and label matrices within the same file.

        Args:
            path: Path to the .mat file.

        Returns:
            bool: True if both source-specific and target-specific keys are present.
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
        """Parse a MATLAB file and return a domain Dataset entity.

        Performs key heuristic searching, fallback matrix discovery, dimension alignment,
        and missing value counting.

        Args:
            path: Path to the .mat file.
            is_source: True if extracting the source domain partition, False for target.

        Returns:
            Dataset: Loaded domain dataset entity.

        Raises:
            ValueError: If feature matrices or label vectors cannot be identified or aligned.
        """
        mat_data = sio.loadmat(path)

        if is_source:
            x_keys = [*_SOURCE_ONLY_X_KEYS, "fea", "X", "x"]
            y_keys = ["Y_src", "Ys", "gnd", "Y", "y", "label", "labels"]
        else:
            x_keys = [*_TARGET_ONLY_X_KEYS, "X", "x"]
            y_keys = ["Y_tar", "Yt", "tar_gnd", "Y", "y", "label", "labels"]

        features = None
        labels = None

        # 1. Match known domain-specific or conventional keys.
        for k in x_keys:
            if k in mat_data:
                features = mat_data[k]
                break

        for k in y_keys:
            if k in mat_data:
                labels = mat_data[k]
                break

        # 2. Heuristic fallback for features: select largest 2D numerical array.
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

        # 3. Heuristic fallback for labels: select largest 1D or single-column array.
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

        # 4. Align matrix orientation (samples x features vs features x samples).
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
