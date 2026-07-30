from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from domain.interfaces import TransferLearningMethod


class TCAMethod(TransferLearningMethod):
    def __init__(self, latent_dim: int = 5, regularization: float = 1.0) -> None:
        self._latent_dim = latent_dim
        self._regularization = regularization

    @property
    def name(self) -> str:
        return "TCA"

    def run(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_size = source_features.shape[0]
        target_size = target_features.shape[0]
        total_size = source_size + target_size

        combined = np.vstack([source_features, target_features])
        combined = StandardScaler().fit_transform(combined)

        e = np.vstack(
            [
                np.full((source_size, 1), 1.0 / source_size),
                np.full((target_size, 1), -1.0 / target_size),
            ]
        )
        m_matrix = e @ e.T
        m_matrix /= np.linalg.norm(m_matrix, ord="fro")

        h_matrix = np.eye(total_size) - (1.0 / total_size) * np.ones((total_size, total_size))
        k_matrix = combined @ combined.T

        left = (k_matrix @ m_matrix @ k_matrix) + self._regularization * np.eye(total_size)
        right = k_matrix @ h_matrix @ k_matrix

        eigen_problem = np.linalg.pinv(left) @ right
        eigenvalues, eigenvectors = np.linalg.eig(eigen_problem)

        sorted_idx = np.argsort(-eigenvalues.real)
        components = min(self._latent_dim, total_size)
        projection = eigenvectors[:, sorted_idx[:components]].real

        transformed_all = k_matrix @ projection
        norms = np.linalg.norm(transformed_all, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        transformed_all = transformed_all / norms

        transformed_source = transformed_all[:source_size, :]
        transformed_target = transformed_all[source_size:, :]
        return transformed_source, transformed_target
