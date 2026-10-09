from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl


class JDAMethod(TransferLearningMethod):
    def __init__(
        self,
        latent_dim: int = 5,
        regularization: float = 1.0,
        kernel_type: str = "linear",
        iterations: int = 10,
        gamma: float = 1.0,
    ) -> None:
        self._latent_dim = latent_dim
        self._regularization = regularization
        self._kernel_type = kernel_type
        self._iterations = iterations
        self._gamma = gamma

    @property
    def name(self) -> str:
        return "JDA"

    def _kernel(self, kernel_type: str, X: np.ndarray, X2: np.ndarray | None, gamma: float) -> np.ndarray:
        # X shape is (m_features, n_samples)
        # X2 shape is (m_features, n2_samples) or None
        if kernel_type == "linear":
            if X2 is None:
                return X.T @ X
            else:
                return X.T @ X2
        elif kernel_type == "rbf":
            n1sq = np.sum(X**2, axis=0)
            if X2 is None:
                D = n1sq[:, np.newaxis] + n1sq[np.newaxis, :] - 2 * (X.T @ X)
            else:
                n2sq = np.sum(X2**2, axis=0)
                D = n1sq[:, np.newaxis] + n2sq[np.newaxis, :] - 2 * (X.T @ X2)
            return np.exp(-gamma * D)
        elif kernel_type == "sam":
            if X2 is None:
                D = X.T @ X
            else:
                D = X.T @ X2
            D = np.clip(D, -1.0, 1.0)
            return np.exp(-gamma * (np.arccos(D) ** 2))
        else:
            raise ValueError(f"Unsupported kernel {kernel_type}")

    def run(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_labels: np.ndarray | None = None,
        control: RunControl | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if control is not None:
            control.checkpoint()
        ns = source_features.shape[0]
        nt = target_features.shape[0]
        n_samples = ns + nt

        combined = np.vstack([source_features, target_features])
        combined = StandardScaler().fit_transform(combined)

        # Normalize each sample vector to unit L2 norm
        sample_norms = np.linalg.norm(combined, axis=1, keepdims=True)
        sample_norms[sample_norms == 0] = 1.0
        combined = combined / sample_norms
        X = combined.T  # shape (m_features, n_samples)

        m, n = X.shape
        max_dim = min(m, n) - 1
        if max_dim < 1:
            raise ValueError("Input data does not have enough rank/samples for eigendecomposition.")
        dim = min(self._latent_dim, max_dim)

        # Centering matrix H
        H = np.eye(n) - (1.0 / n) * np.ones((n, n))

        # Marginal alignment vector e
        e_m = np.vstack([
            np.full((ns, 1), 1.0 / ns),
            np.full((nt, 1), -1.0 / nt)
        ])

        if source_labels is not None:
            classes = np.unique(source_labels)
            num_classes = len(classes)
            # Marginal matrix M0
            M0 = (e_m @ e_m.T) * num_classes
            y_tar_pseudo = None
            iterations = self._iterations
        else:
            classes = []
            num_classes = 1
            M0 = e_m @ e_m.T
            y_tar_pseudo = None
            iterations = 1

        transformed_source = np.zeros((ns, dim))
        transformed_target = np.zeros((nt, dim))

        for it in range(iterations):
            if control is not None:
                control.checkpoint()
                control.report_iteration(self.name, it + 1, iterations)
            # Compute conditional matrix Mc (N)
            N = np.zeros((n, n))
            if y_tar_pseudo is not None and len(y_tar_pseudo) == nt:
                for c in classes:
                    e_c = np.zeros((n, 1))
                    source_c_idx = np.where(source_labels == c)[0]
                    target_c_idx = np.where(y_tar_pseudo == c)[0]

                    if len(source_c_idx) > 0:
                        e_c[source_c_idx] = 1.0 / len(source_c_idx)
                    if len(target_c_idx) > 0:
                        e_c[ns + target_c_idx] = -1.0 / len(target_c_idx)

                    N += e_c @ e_c.T

            M = M0 + N
            m_norm = np.linalg.norm(M, ord="fro")
            if m_norm > 0:
                M = M / m_norm

            # Solve generalized eigenvalue problem
            if self._kernel_type == "primal":
                left = (X @ M @ X.T) + self._regularization * np.eye(m)
                right = X @ H @ X.T

                eigen_problem = np.linalg.pinv(left) @ right
                eigenvalues, eigenvectors = np.linalg.eig(eigen_problem)

                sorted_idx = np.argsort(-eigenvalues.real)
                projection = eigenvectors[:, sorted_idx[:dim]].real
                transformed_all = combined @ projection  # (n_samples, dim)
            else:
                K = self._kernel(self._kernel_type, X, None, self._gamma)
                left = (K @ M @ K) + self._regularization * np.eye(n)
                right = K @ H @ K

                eigen_problem = np.linalg.pinv(left) @ right
                eigenvalues, eigenvectors = np.linalg.eig(eigen_problem)

                sorted_idx = np.argsort(-eigenvalues.real)
                projection = eigenvectors[:, sorted_idx[:dim]].real
                transformed_all = K @ projection  # (n_samples, dim)

            # Sample-wise normalization
            norms = np.linalg.norm(transformed_all, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            transformed_all = transformed_all / norms

            transformed_source = transformed_all[:ns, :]
            transformed_target = transformed_all[ns:, :]

            # Update pseudo labels if we have class labels and more iterations to go
            if source_labels is not None and it < iterations - 1:
                # Use 1-NN
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(transformed_source, source_labels)
                y_tar_pseudo = knn.predict(transformed_target)

        return transformed_source, transformed_target
