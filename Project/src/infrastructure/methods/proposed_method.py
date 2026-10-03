"""Proposed method – Python port of SimpleExps (Proposed_method.m).

Two-phase domain adaptation:

1. **Covariance-aligned subspace** – whiten the source covariance and colour
   with the target covariance, build an MMD + conditional-alignment matrix
   *M*, and iteratively factorise source/target into a shared low-rank
   subspace *Ct_tar_src*.  Pseudo-labels are refreshed each iteration with
   a 1-NN classifier.

2. **OCMTL subspace** – same iterative factorisation on the original
   (non-covariance-aligned) normalised features, sweeping rank from 2
   upward and stopping when the singular-value ratio exceeds a threshold.
   Produces *Ct_ocmtl*.

The transformed source/target are the horizontal concatenation of both
projections (or just phase-1 if OCMTL yields nothing).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.neighbors import KNeighborsClassifier

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl


class ProposedMethod(TransferLearningMethod):
    """Port of ``SimpleExps`` from ``Proposed_method.m``."""

    def __init__(
        self,
        regularization: float = 0.1,
        max_iterations: int = 20,
        convergence_tol: float = 0.001,
        sv_ratio_threshold: float = 2.0,
        covariance_reg: float = 0.1,
        percentile_cutoff: float = 90.0,
        combined_sig_threshold: float = 3.0,
    ) -> None:
        self._regularization = regularization         # gamma in MATLAB
        self._max_iterations = max_iterations         # inner iteration cap
        self._convergence_tol = convergence_tol       # Frobenius convergence
        self._sv_ratio_threshold = sv_ratio_threshold # OCMTL rank stop
        self._covariance_reg = covariance_reg         # diagonal regulariser
        self._percentile_cutoff = percentile_cutoff   # rank selection percentile
        self._combined_sig_threshold = combined_sig_threshold  # threshold on combsig

    @property
    def name(self) -> str:
        return "Proposed"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalise_rows(X: np.ndarray) -> np.ndarray:
        """Per-sample (row) L2 normalisation, NaN/Inf safe."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out = X / norms
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    @staticmethod
    def _l2_normalise_cols(X: np.ndarray) -> np.ndarray:
        """Per-feature (column) L2 normalisation, NaN/Inf safe."""
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        out = X / norms
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    @staticmethod
    def _pseudo_labels(
        source: np.ndarray,
        source_labels: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """1-NN pseudo-label assignment (mirrors ``get_prediction`` in MATLAB)."""
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(source, source_labels)
        return knn.predict(target)

    # ------------------------------------------------------------------
    # Phase 1 – covariance-aligned subspace
    # ------------------------------------------------------------------

    def _phase1(
        self,
        temp_s: np.ndarray,
        temp_t: np.ndarray,
        source_labels: np.ndarray,
        A_col_norm: np.ndarray,
        ns: int,
        nt: int,
        n: int,
        m: int,
        M0: np.ndarray,
        C: int,
        control: RunControl | None,
    ) -> np.ndarray | None:
        """Covariance-aligned iterative factorisation."""
        # Covariance matrices with Tikhonov regularisation
        d = temp_s.shape[1]
        cov_src = (temp_s.T @ temp_s + self._covariance_reg * np.eye(d)) / ns
        cov_tar = (temp_t.T @ temp_t + self._covariance_reg * np.eye(d)) / nt

        # Covariance whitening/colouring transform
        cov_src_neg_half = np.real(fractional_matrix_power(cov_src, -0.5))
        cov_tar_pos_half = np.real(fractional_matrix_power(cov_tar, 0.5))

        temp_s_tar_src = temp_s @ cov_src_neg_half @ cov_tar_pos_half
        temp_t_tar_src = temp_t @ cov_src_neg_half @ cov_tar_pos_half

        # Automatic rank selection via combined singular values
        _, ss_diag, _ = np.linalg.svd(cov_src_neg_half, full_matrices=False)
        _, st_diag, _ = np.linalg.svd(cov_tar_pos_half, full_matrices=False)
        combsig = np.outer(ss_diag, st_diag)

        meds = np.floor(np.percentile(ss_diag, self._percentile_cutoff))

        rank_tar_src = 0
        for xx in range(1, len(ss_diag)):
            if combsig[xx, xx] < self._combined_sig_threshold or ss_diag[xx] < meds:
                rank_tar_src = xx
                break
        rank_tar_src = max(1, rank_tar_src)

        # Initialise random factors
        rng = np.random.RandomState(10)
        Cs = rng.rand(d, rank_tar_src)
        Ct = Cs.copy()

        # Initial pseudo-labels
        Y_tar_pseudo = self._pseudo_labels(temp_s, source_labels, temp_t)
        M = M0.copy()

        for it in range(self._max_iterations):
            if control is not None:
                control.checkpoint()
                control.report_iteration(
                    f"{self.name} (phase 1)", it + 1, self._max_iterations,
                )

            # Conditional alignment matrix N
            N = self._conditional_matrix(
                source_labels, Y_tar_pseudo, ns, nt, n, C,
            )
            mu = 1.0
            M = (1.0 - mu) * M + mu * N
            m_norm = np.linalg.norm(M, ord="fro")
            if m_norm > 0:
                M = M / m_norm

            deno = A_col_norm @ M @ A_col_norm.T + self._regularization * np.eye(m)

            temp_s1 = np.linalg.solve(deno.T, temp_s_tar_src.T).T
            temp_t1 = np.linalg.solve(deno.T, temp_t_tar_src.T).T

            if it > 0:
                cs_old = Cs.copy()
                ct_old = Ct.copy()

            # Alternating SVD factorisation
            Ass = temp_s1 @ Cs
            As, _, _ = np.linalg.svd(Ass, full_matrices=False)

            Css = temp_s1.T @ As
            Cs, _, _ = np.linalg.svd(Css, full_matrices=False)

            Att = temp_t1 @ Cs
            At, _, _ = np.linalg.svd(Att, full_matrices=False)

            Ctt = temp_t1.T @ At
            Ct, _, _ = np.linalg.svd(Ctt, full_matrices=False)

            # Convergence check
            if it > 0:
                cs_norm = np.linalg.norm(cs_old, ord="fro")
                ct_norm = np.linalg.norm(ct_old, ord="fro")
                cs_change = np.linalg.norm(cs_old - Cs, ord="fro") / cs_norm if cs_norm > 0 else 0
                ct_change = np.linalg.norm(ct_old - Ct, ord="fro") / ct_norm if ct_norm > 0 else 0
                if cs_change < self._convergence_tol or ct_change < self._convergence_tol:
                    break

            Cs = (Cs + Ct) / 2.0
            Y_tar_pseudo = self._pseudo_labels(
                temp_s @ Ct, source_labels, temp_t @ Ct,
            )

        return Ct  # Ct_tar_src

    # ------------------------------------------------------------------
    # Phase 2 – OCMTL subspace
    # ------------------------------------------------------------------

    def _phase2(
        self,
        temp_s_ocmtl: np.ndarray,
        temp_t_ocmtl: np.ndarray,
        source_labels: np.ndarray,
        A_col_norm: np.ndarray,
        ns: int,
        nt: int,
        n: int,
        m: int,
        M0: np.ndarray,
        C: int,
        control: RunControl | None,
    ) -> np.ndarray | None:
        """OCMTL-style factorisation with automatic rank selection."""
        d = temp_s_ocmtl.shape[1]
        finalct = None
        rng = np.random.RandomState(10)

        for rank_ocmtl in range(2, d + 1):
            if control is not None:
                control.checkpoint()

            Cs = rng.rand(d, rank_ocmtl)
            Ct = Cs.copy()

            Y_tar_pseudo = self._pseudo_labels(
                temp_s_ocmtl @ Ct, source_labels, temp_t_ocmtl @ Ct,
            )
            M = M0.copy()

            last_s = None

            for it in range(self._max_iterations):
                if control is not None:
                    control.checkpoint()
                    control.report_iteration(
                        f"{self.name} (phase 2, rank {rank_ocmtl})",
                        it + 1,
                        self._max_iterations,
                    )

                N = self._conditional_matrix(
                    source_labels, Y_tar_pseudo, ns, nt, n, C,
                )
                mu = 1.0
                M = (1.0 - mu) * M + mu * N
                m_norm = np.linalg.norm(M, ord="fro")
                if m_norm > 0:
                    M = M / m_norm

                deno = A_col_norm @ M @ A_col_norm.T + self._regularization * np.eye(m)
                temp_s1 = np.linalg.solve(deno.T, temp_s_ocmtl.T).T
                temp_t1 = np.linalg.solve(deno.T, temp_t_ocmtl.T).T

                if it > 0:
                    cs_old = Cs.copy()
                    ct_old = Ct.copy()

                Ass = temp_s1 @ Cs
                As, _, _ = np.linalg.svd(Ass, full_matrices=False)

                Css = temp_s1.T @ As
                Cs, _, _ = np.linalg.svd(Css, full_matrices=False)

                Att = temp_t1 @ Cs
                At, sigma1, _ = np.linalg.svd(Att, full_matrices=False)

                Ctt = temp_t1.T @ At
                Ct, s_vals, _ = np.linalg.svd(Ctt, full_matrices=False)
                last_s = s_vals

                if it > 0:
                    cs_norm = np.linalg.norm(cs_old, ord="fro")
                    ct_norm = np.linalg.norm(ct_old, ord="fro")
                    cs_change = np.linalg.norm(cs_old - Cs, ord="fro") / cs_norm if cs_norm > 0 else 0
                    ct_change = np.linalg.norm(ct_old - Ct, ord="fro") / ct_norm if ct_norm > 0 else 0
                    if cs_change < self._convergence_tol or ct_change < self._convergence_tol:
                        break

                Cs = (Cs + Ct) / 2.0
                Y_tar_pseudo = self._pseudo_labels(
                    temp_s_ocmtl @ Ct, source_labels, temp_t_ocmtl @ Ct,
                )

            if rank_ocmtl == 2:
                finalct = Ct

            # Singular-value ratio stopping criterion
            if last_s is not None and len(last_s) > 0 and last_s[-1] > 0:
                if last_s[0] / last_s[-1] > self._sv_ratio_threshold:
                    break

            finalct = Ct

        return finalct  # Ct_ocmtl (may be None if d < 2)

    # ------------------------------------------------------------------
    # Conditional alignment matrix (shared by both phases)
    # ------------------------------------------------------------------

    @staticmethod
    def _conditional_matrix(
        source_labels: np.ndarray,
        Y_tar_pseudo: np.ndarray | None,
        ns: int,
        nt: int,
        n: int,
        C: int,
    ) -> np.ndarray:
        """Build the class-conditional alignment matrix *N*."""
        N = np.zeros((n, n))
        if Y_tar_pseudo is None or len(Y_tar_pseudo) != nt:
            return N
        classes = np.unique(source_labels)
        for c in classes:
            e = np.zeros((n, 1))
            src_c = np.where(source_labels == c)[0]
            tar_c = np.where(Y_tar_pseudo == c)[0]
            if len(src_c) > 0:
                e[src_c] = 1.0 / len(src_c)
            if len(tar_c) > 0:
                e[ns + tar_c] = -1.0 / len(tar_c)
            e[np.isinf(e)] = 0.0
            N += e @ e.T
        return N

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

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
            raise ValueError("Proposed method requires source_labels.")

        ns = source_features.shape[0]
        nt = target_features.shape[0]

        # Column-normalised combined matrix A = [source'; target']
        A = np.vstack([source_features, target_features]).T  # (d, ns+nt)
        A = self._l2_normalise_cols(A)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        # Row-normalised copies of source/target
        temp_s = self._l2_normalise_rows(source_features)
        temp_t = self._l2_normalise_rows(target_features)

        m, n = A.shape  # m = features, n = ns + nt
        C = len(np.unique(source_labels))

        # Marginal alignment matrix M0
        e = np.vstack([
            np.full((ns, 1), 1.0 / ns),
            np.full((nt, 1), -1.0 / nt),
        ])
        M0 = (e @ e.T) * C

        # --- Phase 1: covariance-aligned subspace ---
        Ct_tar_src = self._phase1(
            temp_s, temp_t, source_labels,
            A, ns, nt, n, m, M0, C, control,
        )

        # --- Phase 2: OCMTL subspace ---
        Ct_ocmtl = self._phase2(
            temp_s, temp_t, source_labels,
            A, ns, nt, n, m, M0, C, control,
        )

        # --- Combine projections ---
        if Ct_tar_src is not None:
            src_part1 = temp_s @ Ct_tar_src
            tar_part1 = temp_t @ Ct_tar_src
        else:
            src_part1 = temp_s
            tar_part1 = temp_t

        if Ct_ocmtl is not None:
            src_part2 = temp_s @ Ct_ocmtl
            tar_part2 = temp_t @ Ct_ocmtl
            transformed_source = np.hstack([src_part1, src_part2])
            transformed_target = np.hstack([tar_part1, tar_part2])
        else:
            transformed_source = src_part1
            transformed_target = tar_part1

        return transformed_source, transformed_target
