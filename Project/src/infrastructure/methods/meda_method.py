from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl
from infrastructure.methods.gfk_method import GFKMethod


class MEDAMethod(TransferLearningMethod):
    r"""Manifold Embedded Distribution Alignment (MEDA, Wang et al., ACM MM 2018).

    Addresses domain shift by embedding features onto the Grassmann manifold via
    Geodesic Flow Kernel (GFK), dynamically estimating the balance between
    marginal and conditional distributions using data-driven A-distance, and
    directly learning the adapted classifier in a unified regularized kernel
    ridge regression framework.

    The algorithm iteratively alternates between:
    1. Mapping source and target features onto the Grassmann manifold via GFK.
    2. Dynamically computing the balance factor mu from marginal and conditional
       A-distances: mu = d_M / (d_M + \\sum d_c).
    3. Constructing the dynamic MMD matrix M = (1 - mu) M_0 + mu \\sum M_c.
    4. Solving the regularized Structural Risk Minimization system in closed form:
           A = [ (E + \\lambda M + \\rho L) K + \\eta I ]^{-1} Y_tilde
    5. Updating target pseudo-labels from soft decision scores until convergence.
    """

    def __init__(
        self,
        latent_dim: int = 20,
        regularization: float = 1.0,
        kernel_type: str = "rbf",
        rho: float = 1.0,
        eta: float = 0.1,
        gamma: float = 1.0,
        iterations: int = 10,
    ) -> None:
        self._latent_dim = latent_dim  # GFK subspace dimension (paper: dim)
        self._regularization = regularization  # weight of the alignment term M (paper: lambda)
        self._kernel_type = kernel_type  # 'linear' | 'rbf' | 'sam'
        self._rho = rho  # weight of the (currently zero) manifold term
        self._eta = eta  # ridge term keeping the system solvable
        self._gamma = gamma  # rbf/sam kernel bandwidth
        self._iterations = iterations  # T

    @property
    def name(self) -> str:
        return "MEDA"

    def _kernel(self, kernel_type: str, X: np.ndarray, X2: np.ndarray | None, gamma: float) -> np.ndarray:
        # X shape is (m_features, n_samples); same convention as JDA/BDA.
        if kernel_type == "linear":
            if X2 is None:
                return X.T @ X
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
            raise ValueError(
                "MEDA solves a kernel ridge regression over samples and has no "
                f"meaningful primal (no-kernel) mode; got {kernel_type!r} -- "
                "pick Linear, RBF, or SAM."
            )

    @staticmethod
    def _proxy_a_distance(source_embed: np.ndarray, target_embed: np.ndarray) -> float:
        """Domain-divergence proxy: how well a linear classifier separates
        source vs. target samples. Near 0 => indistinguishable (well
        aligned); near 2 => trivially separable (poorly aligned)."""
        n_source = source_embed.shape[0]
        n_target = target_embed.shape[0]
        # Standardized purely so LinearSVC's optimizer converges quickly;
        # doesn't change what separability it measures.
        train_X = StandardScaler().fit_transform(np.vstack([source_embed, target_embed]))
        train_y = np.hstack([np.zeros(n_source, dtype=int), np.ones(n_target, dtype=int)])
        clf = LinearSVC(random_state=0, max_iter=2000)
        clf.fit(train_X, train_y)
        predictions = clf.predict(train_X)
        error = mean_absolute_error(train_y, predictions)
        return 2 * (1 - 2 * error)

    def _estimate_mu(
        self,
        source_embed: np.ndarray,
        source_labels: np.ndarray,
        target_embed: np.ndarray,
        target_pseudo: np.ndarray,
    ) -> float:
        overall_distance = self._proxy_a_distance(source_embed, target_embed)
        class_distances = []
        for c in np.unique(source_labels):
            source_idx = np.where(source_labels == c)[0]
            target_idx = np.where(target_pseudo == c)[0]
            if len(source_idx) == 0 or len(target_idx) == 0:
                continue
            class_distances.append(
                self._proxy_a_distance(source_embed[source_idx], target_embed[target_idx])
            )

        if not class_distances:
            return 0.5
        conditional_distance = sum(class_distances) / len(class_distances)

        denom = conditional_distance + overall_distance
        mu = conditional_distance / denom if denom > 0 else 0.5
        mu = min(mu, 1.0)
        return 0.0 if mu < 1e-3 else mu

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
            raise ValueError("MEDA requires source_labels")

        ns = source_features.shape[0]
        nt = target_features.shape[0]
        n_features = source_features.shape[1]
        n_total = ns + nt
        classes = np.unique(source_labels)
        num_classes = len(classes)
        class_index = {c: i for i, c in enumerate(classes)}

        # Step 1: Geodesic Flow Kernel embedding (see class docstring for why
        # this reuses GFKMethod rather than the reference's bob-based GFK).
        dim = max(1, min(self._latent_dim, n_features // 2))
        source_basis = GFKMethod._full_pca_basis(source_features)
        target_basis = GFKMethod._full_pca_basis(target_features)
        g = GFKMethod._gfk_kernel(source_basis, target_basis[:, :dim])
        eigenvalues, eigenvectors = np.linalg.eigh(g)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        sqrt_g = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

        xs_new = source_features @ sqrt_g  # ns x m
        xt_new = target_features @ sqrt_g  # nt x m

        # Step 2: manifold + distribution-regularized kernel ridge
        # regression over the embedded features (features x samples, same
        # convention as JDA/BDA).
        X = np.hstack([xs_new.T, xt_new.T])
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        YY = np.zeros((n_total, num_classes))
        for row, label in enumerate(source_labels):
            YY[row, class_index[label]] = 1.0

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X[:, :ns].T, source_labels)
        target_pseudo = knn.predict(X[:, ns:].T)

        K = self._kernel(self._kernel_type, X, None, self._gamma)
        E = np.diag(np.concatenate([np.ones(ns), np.zeros(nt)]))
        identity = np.eye(n_total)

        F = None
        for it in range(self._iterations):
            if control is not None:
                control.checkpoint()
                control.report_iteration(self.name, it + 1, self._iterations)
            mu = self._estimate_mu(xs_new, source_labels, xt_new, target_pseudo)

            e_marginal = np.vstack([np.full((ns, 1), 1.0 / ns), np.full((nt, 1), -1.0 / nt)])
            M = (e_marginal @ e_marginal.T) * num_classes

            N = np.zeros((n_total, n_total))
            for c in classes:
                e_c = np.zeros((n_total, 1))
                s_idx = np.where(source_labels == c)[0]
                t_idx = np.where(target_pseudo == c)[0]
                if len(s_idx) > 0:
                    e_c[s_idx] = 1.0 / len(s_idx)
                if len(t_idx) > 0:
                    e_c[ns + t_idx] = -1.0 / len(t_idx)
                N += e_c @ e_c.T

            M = (1 - mu) * M + mu * N
            m_norm = np.linalg.norm(M, ord="fro")
            if m_norm > 0:
                M = M / m_norm

            # rho * L term omitted: L is the zero matrix, see class docstring.
            left = (E + self._regularization * M) @ K + self._eta * identity
            try:
                beta = np.linalg.solve(left, E @ YY)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(left) @ (E @ YY)

            F = K @ beta
            target_pseudo = classes[np.argmax(F[ns:], axis=1)]

        transformed_source = F[:ns]
        transformed_target = F[ns:]
        return transformed_source, transformed_target
