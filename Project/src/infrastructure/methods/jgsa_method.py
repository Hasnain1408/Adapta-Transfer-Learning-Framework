from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl


class JGSAMethod(TransferLearningMethod):
    """Joint Geometrical and Statistical Alignment (Zhang et al., CVPR 2017).

    Ported from ``TL methods/2017-JGSA/JGSA.m``. Like JDA/BDA it alternates
    building marginal+conditional MMD alignment matrices from
    (pseudo-)labels and solving a generalized eigenproblem, but additionally
    learns *separate* source and target projections regularized by the
    source's within/between-class scatter and the target's variance, rather
    than one shared projection.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        regularization: float = 1.0,
        beta: float = 1.0,
        mu: float = 1.0,
        iterations: int = 10,
    ) -> None:
        self._latent_dim = latent_dim
        self._regularization = regularization
        self._beta = beta
        self._mu = mu
        self._iterations = iterations

    @property
    def name(self) -> str:
        return "JGSA"

    @staticmethod
    def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    @staticmethod
    def _within_scatter(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
            raise ValueError("JGSA requires source_labels")

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

        Sw = self._within_scatter(Xs, source_labels)
        Sb = self._between_scatter(Xs, source_labels)
        sb_norm = np.linalg.norm(Sb, ord="fro")
        if sb_norm > 0:
            Sb = Sb / sb_norm

        M0 = self._marginal_distribution(ns, nt, num_classes)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(source_norm, source_labels)
        target_pseudo = knn.predict(target_norm)

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
                    [self._beta * Sb, np.zeros((m, m))],
                    [np.zeros((m, m)), self._mu * St],
                ]
            )
            right = np.block(
                [
                    [Ms + self._regularization * identity_m + self._beta * Sw, Mst - self._regularization * identity_m],
                    [Mts - self._regularization * identity_m, Mt + (self._regularization + self._mu) * identity_m],
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
