from __future__ import annotations

import numpy as np

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl


class CORALMethod(TransferLearningMethod):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "CORAL"

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

        # Calculate covariance matrices
        # We ensure cov is a 2D array even if there is only one feature dimension
        cov_src = np.atleast_2d(np.cov(source_features, rowvar=False)) + np.eye(n_features)
        cov_tar = np.atleast_2d(np.cov(target_features, rowvar=False)) + np.eye(n_features)

        # Eigendecomposition for matrix square roots (symmetric positive definite matrices)
        evals_src, evecs_src = np.linalg.eigh(cov_src)
        evals_tar, evecs_tar = np.linalg.eigh(cov_tar)

        # Clip eigenvalues to ensure numerical stability and strictly positive values
        evals_src = np.maximum(evals_src, 1e-9)
        evals_tar = np.maximum(evals_tar, 1e-9)

        # Compute cov_src^(-1/2) and cov_tar^(1/2)
        cov_src_inv_sqrt = evecs_src @ np.diag(1.0 / np.sqrt(evals_src)) @ evecs_src.T
        cov_tar_sqrt = evecs_tar @ np.diag(np.sqrt(evals_tar)) @ evecs_tar.T

        # CORAL alignment transformation matrix
        a_coral = cov_src_inv_sqrt @ cov_tar_sqrt

        # Project source features to target covariance space
        transformed_source = source_features @ a_coral
        transformed_target = target_features

        return transformed_source, transformed_target
