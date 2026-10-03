from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl


class OCMTLMethod(TransferLearningMethod):
    def __init__(self, latent_dim: int = 2, max_iterations: int = 20) -> None:
        self._latent_dim = latent_dim
        self._max_iterations = max_iterations

    @property
    def name(self) -> str:
        return "OCMTL"

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
            raise ValueError("OCMTL requires source_labels to compute class balancing.")

        n_features = source_features.shape[1]
        
        # Calculate class weights safely for variable number of classes
        classes, counts = np.unique(source_labels, return_counts=True)
        hsc_diag = np.zeros(source_features.shape[0])
        for c, count in zip(classes, counts):
            mask = (source_labels == c)
            if count > 0:
                hsc_diag[mask] = 1.0 / np.sqrt(count)
                
        Hsc = np.diag(hsc_diag)

        target = target_features
        source = source_features

        original_tensor_mean = np.mean(target, axis=0)
        original_distance = np.linalg.norm(target - original_tensor_mean, axis=1)

        prevct = None
        finalct = None

        dim = min(self._latent_dim, n_features)
        
        # Initialize randomly like the original MATLAB: Ct = rand(source_size(2), i);
        np.random.seed(42)  # For consistent testing
        Ct = np.random.rand(n_features, dim)
        
        temp_angle = np.inf
        
        for it in range(self._max_iterations):
            if control is not None:
                control.checkpoint()
                control.report_iteration(self.name, it + 1, self._max_iterations)
            # Ass = temp_s * Ct (m x k)
            Ass = source @ Ct
            U_As, _, _ = np.linalg.svd(Ass, full_matrices=False)
            As_mat = U_As
            
            # Css = temp_s' * Hsc * As (k x k, wait actually n_features x k)
            Css = source.T @ Hsc @ As_mat
            U_Cs, _, _ = np.linalg.svd(Css, full_matrices=False)
            Cs_mat = U_Cs
            
            # Att = temp_t * Cs (p x k)
            Att = target @ Cs_mat
            U_At, _, _ = np.linalg.svd(Att, full_matrices=False)
            At_mat = U_At
            
            # Ctt = temp_t' * At (n_features x k)
            Ctt = target.T @ At_mat
            U_Ct, _, _ = np.linalg.svd(Ctt, full_matrices=False)
            Ct_new = U_Ct
            
            ang = np.max(subspace_angles(Cs_mat, Ct_new))
            
            if it > 0 and ang > temp_angle:
                break
            else:
                prevct = Ct_new
                temp_angle = ang
                Ct = Ct_new
                
        if prevct is not None:
            Ct = prevct
            
        finalct = Ct
        
        transformed_source = source @ finalct
        transformed_target = target @ finalct
        
        return transformed_source, transformed_target
