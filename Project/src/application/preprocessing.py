"""Feature matrix preprocessing: normalization and standardization transformations."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler


def apply_normalization(features: np.ndarray, normalization_type: str) -> np.ndarray:
    """Apply feature normalization transform according to the specified option."""
    if normalization_type == "Min-Max Normalization":
        return MinMaxScaler().fit_transform(features)
    if normalization_type == "Z-score Standardization":
        return StandardScaler().fit_transform(features)
    if normalization_type == "Robust Standardization":
        return RobustScaler().fit_transform(features)
    if normalization_type == "L1 normalization":
        return Normalizer(norm="l1").fit_transform(features)
    if normalization_type in ("L2 (Unit Vector) Normalization", "Row-wise unit norm"):
        return Normalizer(norm="l2").fit_transform(features)
    if normalization_type == "Mean Centering":
        return StandardScaler(with_std=False).fit_transform(features)
    if normalization_type == "Column-wise unit norm":
        return Normalizer(norm="l2").fit_transform(features.T).T
    if normalization_type == "Frobenius normalization":
        norm = np.linalg.norm(features, ord="fro")
        return features / norm if norm > 0 else features
    if normalization_type == "Spectral normalization":
        norm = np.linalg.norm(features, ord=2)
        return features / norm if norm > 0 else features
    return features


def apply_standardization(features: np.ndarray, standardization_type: str) -> np.ndarray:
    """Apply feature standardization transform according to the specified option."""
    if standardization_type == "Z-score Standardization":
        return StandardScaler().fit_transform(features)
    if standardization_type == "Robust Standardization":
        return RobustScaler().fit_transform(features)
    if standardization_type == "Variance Scaling":
        return StandardScaler(with_mean=False).fit_transform(features)
    if standardization_type == "Column mean centering":
        return StandardScaler(with_std=False).fit_transform(features)
    if standardization_type == "Row mean centering":
        return features - features.mean(axis=1, keepdims=True)
    if standardization_type == "Mahalanobis Standardization":
        covariance = np.cov(features, rowvar=False)
        u, singular_values, _vt = np.linalg.svd(covariance)
        inverse_sqrt = np.diag(1.0 / np.sqrt(singular_values + 1e-10))
        whitening = u @ inverse_sqrt @ u.T
        mean = np.mean(features, axis=0)
        return (features - mean) @ whitening
    return features


__all__ = ["apply_normalization", "apply_standardization"]
