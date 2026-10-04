from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl


class GFKMethod(TransferLearningMethod):
    """Geodesic Flow Kernel (GFK, Gong et al., CVPR 2012).

    Models domain adaptation on the Grassmann manifold by parameterizing the
    continuous geodesic flow between source and target PCA subspaces, integrating
    over an infinite number of intermediate subspaces in closed form.

    Constructs a positive semi-definite (PSD) kernel matrix G from principal
    angles and projects both domains into the domain-invariant space via G^{0.5},
    requiring neither target labels nor iterative optimization.
    """

    def __init__(self, latent_dim: int = 10) -> None:
        self._latent_dim = latent_dim

    @property
    def name(self) -> str:
        return "GFK"

    @staticmethod
    def _full_pca_basis(X: np.ndarray) -> np.ndarray:
        # Computes the full m x m orthonormal principal component basis
        # via eigendecomposition of the centered sample covariance.
        centered = X - X.mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(-eigenvalues)
        return eigenvectors[:, order]

    @staticmethod
    def _gfk_kernel(source_basis: np.ndarray, target_subspace: np.ndarray) -> np.ndarray:
        # source_basis (Q): m x m full orthonormal basis of the source.
        # target_subspace (Pt): m x dim orthonormal basis of the target.
        m = source_basis.shape[0]
        dim = target_subspace.shape[1]

        q_pt = source_basis.T @ target_subspace  # m x dim, orthonormal columns
        a_block = q_pt[:dim, :]
        b_block = q_pt[dim:, :]

        # Columns of [a_block; b_block] are orthonormal (a_block' a_block + b_block' b_block = I),
        # matching the setup for Cosine-Sine (CS) decomposition via SVD of a_block.
        u1, cos_vals, vt = np.linalg.svd(a_block)
        cos_vals = np.clip(cos_vals, -1.0, 1.0)
        theta = np.arccos(cos_vals)
        v = vt.T

        sin_vals = np.sqrt(np.clip(1.0 - cos_vals**2, 0.0, 1.0))
        bv = b_block @ v
        u2_partial = np.zeros_like(bv)
        valid = sin_vals > 1e-8
        u2_partial[:, valid] = bv[:, valid] / sin_vals[valid]

        # Complete u2's valid columns to a full (m-dim) x (m-dim) orthogonal
        # matrix. The completion columns only multiply zero blocks below, so
        # any orthonormal choice for them leaves G unchanged.
        base = u2_partial[:, valid]
        if base.shape[1] > 0:
            base, _ = np.linalg.qr(base)
        rows = m - dim
        if base.shape[1] < rows:
            complement = null_space(base.T) if base.shape[1] > 0 else np.eye(rows)
            v2 = np.hstack([base, complement])
        else:
            v2 = base
        if v2.shape[1] != rows:
            v2 = np.linalg.qr(np.hstack([v2, np.eye(rows)]))[0][:, :rows]

        eps = 1e-20
        theta_safe = np.maximum(theta, eps)
        b1 = np.diag(0.5 * (1.0 + np.sin(2 * theta) / (2 * theta_safe)))
        b2 = np.diag(0.5 * (np.cos(2 * theta) - 1.0) / (2 * theta_safe))
        b3 = b2
        b4 = np.diag(0.5 * (1.0 - np.sin(2 * theta) / (2 * theta_safe)))

        v1_full = np.zeros((m, dim))
        v1_full[:dim, :] = u1
        v2_full = np.zeros((m, rows))
        v2_full[dim:, :] = v2
        basis = np.hstack([v1_full, v2_full])  # m x m orthogonal

        pad = m - 2 * dim
        top = np.hstack([b1, b2, np.zeros((dim, pad))])
        mid = np.hstack([b3, b4, np.zeros((dim, pad))])
        bot = np.zeros((pad, m))
        middle = np.vstack([top, mid, bot])

        return source_basis @ basis @ middle @ basis.T @ source_basis.T

    def run(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_labels: np.ndarray | None = None,
        control: RunControl | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if control is not None:
            control.checkpoint()
        n_features = source_features.shape[1]
        # Paper's constraint: dim <= 0.5 * n_feature, and dim must be >= 1.
        dim = max(1, min(self._latent_dim, n_features // 2))

        source_basis = self._full_pca_basis(source_features)
        target_basis = self._full_pca_basis(target_features)
        target_subspace = target_basis[:, :dim]

        g = self._gfk_kernel(source_basis, target_subspace)

        eigenvalues, eigenvectors = np.linalg.eigh(g)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        sqrt_g = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

        transformed_source = source_features @ sqrt_g
        transformed_target = target_features @ sqrt_g
        return transformed_source, transformed_target
