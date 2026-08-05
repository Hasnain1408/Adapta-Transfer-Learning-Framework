from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from domain.interfaces import TransferLearningMethod


class TCAMethod(TransferLearningMethod):
    def __init__(
        self,
        latent_dim: int = 5,
        regularization: float = 1.0,
        kernel_type: str = "linear",
    ) -> None:
        self._latent_dim = latent_dim
        self._regularization = regularization
        self._kernel_type = kernel_type
        self._gamma = 1.0

    @property
    def name(self) -> str:
        return "TCA"

    def _kernel(
        self,
        kernel_type: str,
        X: np.ndarray,
        X2: np.ndarray | None,
        gamma: float,
    ) -> np.ndarray:
        # X is shape (n_samples, n_features); X2 is the second block or None.
        if kernel_type == "linear":
            if X2 is None:
                return X @ X.T
            return X @ X2.T
        elif kernel_type == "rbf":
            n1sq = np.sum(X**2, axis=1)
            if X2 is None:
                distance_sq = n1sq[:, np.newaxis] + n1sq[np.newaxis, :] - 2 * (X @ X.T)
            else:
                n2sq = np.sum(X2**2, axis=1)
                distance_sq = n1sq[:, np.newaxis] + n2sq[np.newaxis, :] - 2 * (X @ X2.T)
            distance_sq = np.maximum(distance_sq, 0.0)
            return np.exp(-gamma * distance_sq)
        elif kernel_type == "sam":
            if X2 is None:
                gram = X @ X.T
                norms = np.linalg.norm(X, axis=1)
                cosine = gram / np.maximum(np.outer(norms, norms), 1e-12)
            else:
                gram = X @ X2.T
                norms = np.linalg.norm(X, axis=1)
                norms2 = np.linalg.norm(X2, axis=1)
                cosine = gram / np.maximum(np.outer(norms, norms2), 1e-12)
            cosine = np.clip(cosine, -1.0, 1.0)
            return np.exp(-gamma * (np.arccos(cosine) ** 2))
        else:
            raise ValueError(f"Unsupported kernel {kernel_type}")

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
        components = min(self._latent_dim, total_size)

        if self._kernel_type == "primal":
            # Feature-space form: project onto the raw feature basis rather than
            # the reproducing-kernel basis.
            left = (combined.T @ m_matrix @ combined) + self._regularization * np.eye(
                combined.shape[1]
            )
            right = combined.T @ h_matrix @ combined

            eigen_problem = np.linalg.pinv(left) @ right
            eigenvalues, eigenvectors = np.linalg.eig(eigen_problem)

            sorted_idx = np.argsort(-eigenvalues.real)
            projection = eigenvectors[:, sorted_idx[:components]].real
            transformed_all = combined @ projection
        else:
            k_matrix = self._kernel(self._kernel_type, combined, None, self._gamma)

            left = (k_matrix @ m_matrix @ k_matrix) + self._regularization * np.eye(total_size)
            right = k_matrix @ h_matrix @ k_matrix

            eigen_problem = np.linalg.pinv(left) @ right
            eigenvalues, eigenvectors = np.linalg.eig(eigen_problem)

            sorted_idx = np.argsort(-eigenvalues.real)
            projection = eigenvectors[:, sorted_idx[:components]].real
            transformed_all = k_matrix @ projection

        norms = np.linalg.norm(transformed_all, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        transformed_all = transformed_all / norms

        transformed_source = transformed_all[:source_size, :]
        transformed_target = transformed_all[source_size:, :]
        return transformed_source, transformed_target
