from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl


class BDAMethod(TransferLearningMethod):
    def __init__(
        self,
        mode: str = "W-BDA",
        mu: float = 1.0,
        regularization: float = 0.1,  # lambda
        latent_dim: int = 10,
        kernel_type: str = "primal",
        gamma: float = 1.0,
        iterations: int = 10,
    ) -> None:
        self._mode = mode
        self._mu = mu
        self._regularization = regularization
        self._latent_dim = latent_dim
        self._kernel_type = kernel_type
        self._gamma = gamma
        self._iterations = iterations

    @property
    def name(self) -> str:
        return "BDA"

    def _kernel(self, kernel_type: str, X: np.ndarray, X2: np.ndarray | None, gamma: float) -> np.ndarray:
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
        if source_labels is None:
            raise ValueError("BDA requires source_labels")

        ns = source_features.shape[0]
        nt = target_features.shape[0]
        n = ns + nt

        combined = np.vstack([source_features, target_features])
        # StandardScaler is commonly applied as a best practice before transfer learning
        combined = StandardScaler().fit_transform(combined)

        # L2 column normalization (MATLAB: X = X*diag(sparse(1./sqrt(sum(X.^2)))))
        X = combined.T
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        m = X.shape[0]
        classes = np.unique(source_labels)
        C = len(classes)

        # Centering matrix H
        H = np.eye(n) - (1.0 / n) * np.ones((n, n))

        # M0
        e = np.vstack([np.ones((ns, 1)) / ns, -np.ones((nt, 1)) / nt])
        M0 = (e @ e.T) * C

        Y_tar_pseudo = None
        if self._mode == "W-BDA":
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X[:, :ns].T, source_labels)
            Y_tar_pseudo = knn.predict(X[:, ns:].T)

        dim = min(self._latent_dim, min(m, n) - 1)
        if dim < 1:
            dim = 1
            
        transformed_source = np.zeros((ns, dim))
        transformed_target = np.zeros((nt, dim))

        for it in range(self._iterations):
            if control is not None:
                control.checkpoint()
            N_mat = np.zeros((n, n))
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in classes:
                    e_c = np.zeros((n, 1))
                    ns_c = np.sum(source_labels == c)
                    nt_c = np.sum(Y_tar_pseudo == c)

                    if self._mode == "W-BDA":
                        Ps = ns_c / len(source_labels) if len(source_labels) > 0 else 0
                        Pt = nt_c / len(Y_tar_pseudo) if len(Y_tar_pseudo) > 0 else 0
                        alpha = Pt / Ps if Ps > 0 else 1.0
                        mu_iter = 1.0
                    else:
                        alpha = 1.0
                        mu_iter = self._mu

                    if ns_c > 0:
                        e_c[np.where(source_labels == c)[0]] = 1.0 / ns_c
                    if nt_c > 0:
                        e_c[ns + np.where(Y_tar_pseudo == c)[0]] = -alpha / nt_c

                    N_mat += (e_c @ e_c.T)
            else:
                mu_iter = self._mu

            M = (1 - mu_iter) * M0 + mu_iter * N_mat
            m_norm = np.linalg.norm(M, ord="fro")
            if m_norm > 0:
                M = M / m_norm

            if self._kernel_type == "primal":
                left = (X @ M @ X.T) + self._regularization * np.eye(m)
                right = X @ H @ X.T
                
                try:
                    eigen_problem = np.linalg.pinv(left) @ right
                except np.linalg.LinAlgError:
                    eigen_problem = np.linalg.inv(left) @ right
                
                eigenvalues, eigenvectors = np.linalg.eig(eigen_problem)
                sorted_idx = np.argsort(-eigenvalues.real)
                A = eigenvectors[:, sorted_idx[:dim]].real
                Z = A.T @ X
            else:
                K = self._kernel(self._kernel_type, X, None, self._gamma)
                left = (K @ M @ K.T) + self._regularization * np.eye(n)
                right = K @ H @ K.T
                
                try:
                    eigen_problem = np.linalg.pinv(left) @ right
                except np.linalg.LinAlgError:
                    eigen_problem = np.linalg.inv(left) @ right
                
                eigenvalues, eigenvectors = np.linalg.eig(eigen_problem)
                sorted_idx = np.argsort(-eigenvalues.real)
                A = eigenvectors[:, sorted_idx[:dim]].real
                Z = A.T @ K

            # MATLAB: Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
            z_norms = np.linalg.norm(Z, axis=0, keepdims=True)
            z_norms[z_norms == 0] = 1.0
            Z = Z / z_norms

            transformed_source = Z[:, :ns].T
            transformed_target = Z[:, ns:].T

            if it < self._iterations - 1:
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(transformed_source, source_labels)
                Y_tar_pseudo = knn.predict(transformed_target)

        return transformed_source, transformed_target
