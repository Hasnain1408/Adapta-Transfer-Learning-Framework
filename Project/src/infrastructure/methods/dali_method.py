from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl


class DALIMethod(TransferLearningMethod):
    r"""Visual Domain Adaptation through Locality Information (DALI, 2023).

    Simultaneously minimizes distribution discrepancy across domains, preserves
    intrinsic local geometric structure, and enhances target class discriminability
    within a unified optimization framework.

    The method alternates between:
    1. Estimating target pseudo-labels using a base classifier in the current
       adapted subspace.
    2. Constructing joint Maximum Mean Discrepancy (MMD) matrices combining
       marginal ($M_0$) and conditional ($N$) distribution divergence: $M = M_0 + N$.
    3. Preserving local neighborhood manifolds on the source domain using
       Locality Preserving Projections (LPP) with graph Laplacian regularization ($L + D$).
    4. Enhancing target domain class discriminability by maximizing between-class
       scatter ($S_b$) and minimizing within-class scatter ($S_w$), while preserving
       target total data variance ($S_t$).
    5. Solving a coupled generalized eigenvalue problem for source ($A$) and
       target ($B$) projection bases:
           left @ W = lambda * right @ W
       where the cross-domain divergence penalty $\|A - B\|_F^2$ enforces alignment.
    6. Iterating subspace extraction and pseudo-label refinement until convergence.

    Parameters
    ----------
    latent_dim : int, default=10
        Dimensionality of the learned domain-invariant subspace.
    regularization : float, default=1.0
        Tikhonov regularization and domain divergence penalty enforcing A \approx B.
    beta : float, default=0.4
        Regularization coefficient for source locality-preserving graph structure (LPP).
    delta : float, default=0.2
        Trade-off coefficient for target discriminative scatter (between-class vs within-class).
    epsilon : float, default=0.1
        Regularization coefficient for target total scatter variance preservation.
    iterations : int, default=10
        Number of alternating optimization and pseudo-label refinement iterations.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        regularization: float = 1.0,
        beta: float = 0.4,
        delta: float = 0.2,
        epsilon: float = 0.1,
        iterations: int = 10,
    ) -> None:
        self._latent_dim = latent_dim
        self._regularization = regularization
        self._beta = beta
        self._delta = delta
        self._epsilon = epsilon
        self._iterations = iterations

    @property
    def name(self) -> str:
        return "DALI"

    @staticmethod
    def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    @staticmethod
    def _within_scatter(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        # X: m x n (features x samples)
        m = X.shape[0]
        Sw = np.zeros((m, m))
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            Xc = X[:, idx]
            diff = Xc - Xc.mean(axis=1, keepdims=True)
            Sw += diff @ diff.T
        return Sw

    @staticmethod
    def _between_scatter(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        mean_all = X.mean(axis=1, keepdims=True)
        Sb = np.zeros((m, m))
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            nc = len(idx)
            if nc == 0:
                continue
            diff = X[:, idx].mean(axis=1, keepdims=True) - mean_all
            Sb += nc * (diff @ diff.T)
        return Sb

    @staticmethod
    def _calculate_lpp(samples_by_features: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        # Construct a k-NN adjacency graph W and unnormalized Graph Laplacian L = D - W
        # over the feature space to preserve intrinsic geometric locality.
        X = samples_by_features
        n = X.shape[1]
        W = np.zeros((n, n))
        for i in range(n):
            distances = np.sum((X[:, [i]] - X) ** 2, axis=0)
            nearest = np.argsort(distances)[1 : k + 1]
            W[i, nearest] = 1.0
            W[nearest, i] = 1.0
        D = np.diag(W.sum(axis=1))
        L = D - W
        return L, D

    @staticmethod
    def _marginal_distribution(ns: int, nt: int, num_classes: int) -> np.ndarray:
        e = np.vstack([np.full((ns, 1), 1.0 / ns), np.full((nt, 1), -1.0 / nt)])
        return (e @ e.T) * num_classes

    @staticmethod
    def _conditional_distribution(
        source_labels: np.ndarray, target_pseudo: np.ndarray, ns: int, nt: int, classes: np.ndarray
    ) -> np.ndarray:
        n = ns + nt
        N = np.zeros((n, n))
        for c in classes:
            e_c = np.zeros((n, 1))
            s_idx = np.where(source_labels == c)[0]
            t_idx = np.where(target_pseudo == c)[0]
            if len(s_idx) > 0:
                e_c[s_idx] = 1.0 / len(s_idx)
            if len(t_idx) > 0:
                e_c[ns + t_idx] = -1.0 / len(t_idx)
            N += e_c @ e_c.T
        return N

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
            raise ValueError("DALI requires source_labels")

        ns = source_features.shape[0]
        nt = target_features.shape[0]
        m = source_features.shape[1]
        classes = np.unique(source_labels)
        num_classes = len(classes)

        source_norm = self._l2_normalize_rows(source_features)
        target_norm = self._l2_normalize_rows(target_features)
        Xs = source_norm.T  # m x ns
        Xt = target_norm.T  # m x nt

        Ht = np.eye(nt) - (1.0 / nt) * np.ones((nt, nt))
        St = Xt @ Ht @ Xt.T

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(source_norm, source_labels)
        target_pseudo = knn.predict(target_norm)

        Sw = self._within_scatter(Xt, target_pseudo)
        Sb = self._between_scatter(Xt, target_pseudo)
        sb_norm = np.linalg.norm(Sb, ord="fro")
        if sb_norm > 0:
            Sb = Sb / sb_norm

        M0 = self._marginal_distribution(ns, nt, num_classes)
        L, D = self._calculate_lpp(source_features, k=1)

        dim = max(1, min(self._latent_dim, m))
        identity_m = np.eye(m)

        transformed_source = source_norm[:, :dim] if source_norm.shape[1] >= dim else source_norm
        transformed_target = target_norm[:, :dim] if target_norm.shape[1] >= dim else target_norm

        for it in range(self._iterations):
            if control is not None:
                control.checkpoint()
                control.report_iteration(self.name, it + 1, self._iterations)
            N = self._conditional_distribution(source_labels, target_pseudo, ns, nt, classes)
            M = M0 + N
            m_norm = np.linalg.norm(M, ord="fro")
            if m_norm > 0:
                M = M / m_norm

            Ms = Xs @ M[:ns, :ns] @ Xs.T
            Mt = Xt @ M[ns:, ns:] @ Xt.T
            Mst = Xs @ M[:ns, ns:] @ Xt.T
            Mts = Xt @ M[ns:, :ns] @ Xs.T

            left = np.block(
                [
                    [np.zeros((m, m)), np.zeros((m, m))],
                    [np.zeros((m, m)), self._epsilon * St + self._delta * Sb],
                ]
            )
            right = np.block(
                [
                    [Ms + self._regularization * identity_m + self._beta * (L + D), Mst - self._regularization * identity_m],
                    [Mts - self._regularization * identity_m, Mt + (self._regularization + self._epsilon) * identity_m + self._delta * Sw],
                ]
            )

            try:
                eigen_problem = np.linalg.pinv(right) @ left
            except np.linalg.LinAlgError:
                eigen_problem = np.linalg.inv(right) @ left
            eigenvalues, eigenvectors = np.linalg.eig(eigen_problem)
            sorted_idx = np.argsort(-eigenvalues.real)
            W = eigenvectors[:, sorted_idx[:dim]].real

            A = W[:m, :]
            B = W[m:, :]
            Zs = A.T @ Xs
            Zt = B.T @ Xt

            transformed_source = Zs.T
            transformed_target = Zt.T

            if it < self._iterations - 1:
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(transformed_source, source_labels)
                target_pseudo = knn.predict(transformed_target)

        return transformed_source, transformed_target
