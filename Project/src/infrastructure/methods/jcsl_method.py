from __future__ import annotations

import numpy as np

from domain.interfaces import TransferLearningMethod
from domain.run_control import RunControl


class JCSLMethod(TransferLearningMethod):
    """Joint Cross-domain Classification and Subspace Learning (Fernando et al., 2015).

    Ported from ``TL methods/JCSL.m``. Learns a source-side projection U via
    mini-batch SGD on a hinge loss, regularized to stay close to the
    target's fixed PCA subspace V. The reference implementation is
    binary-hinge only (``Ys`` in {-1,+1}); this port generalizes to
    multi-class datasets with a one-vs-rest hinge term per class (shared U),
    which reduces to the original formulation exactly when there are only
    two classes.

    Note: MATLAB's inner loop reassigns ``gradient_U`` on every violated
    sample instead of accumulating it (``for jj = 1:ne gradient_U = ...``),
    so only the last sample survives before the ``/ne`` averaging that
    follows -- that average would be meaningless applied to a single
    sample's raw gradient, so this port accumulates over violated samples
    before averaging, as the surrounding code clearly intends.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        regularization: float = 0.01,
        subspace_weight: float = 0.5,
        iterations: int = 200,
        batch_size: int = 25,
        learning_rate: float = 0.1,
    ) -> None:
        self._latent_dim = latent_dim
        self._regularization = regularization  # hinge weight (MATLAB Params.L)
        self._subspace_weight = subspace_weight  # MATLAB Params.B
        self._iterations = iterations
        self._batch_size = batch_size
        self._learning_rate = learning_rate

    @property
    def name(self) -> str:
        return "JCSL"

    @staticmethod
    def _pca_top(X: np.ndarray, dim: int) -> np.ndarray:
        centered = X - X.mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(-eigenvalues)
        return eigenvectors[:, order[:dim]]

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
            raise ValueError("JCSL requires source_labels")

        ns, m = source_features.shape
        dim = max(1, min(self._latent_dim, m, target_features.shape[0] - 1 or m))
        V = self._pca_top(target_features, dim)

        classes = np.unique(source_labels)
        rng = np.random.default_rng(42)
        U = rng.standard_normal((m, dim))
        weights = [rng.standard_normal(dim) for _ in classes]

        hinge_weight = self._regularization
        subspace_weight = self._subspace_weight
        lr = self._learning_rate
        batch_size = self._batch_size

        def objective(U: np.ndarray, weights: list[np.ndarray]) -> float:
            reg = sum(float(w @ w) for w in weights)
            hinge = 0.0
            for c, w in zip(classes, weights):
                y = np.where(source_labels == c, 1.0, -1.0)
                margins = y * (source_features @ U @ w)
                hinge += float(np.mean(np.maximum(0.0, 1.0 - margins)))
            penalty = subspace_weight * np.linalg.norm(U - V, ord="fro")
            return reg + hinge_weight * hinge + penalty

        obj_old = objective(U, weights)
        n_batches = int(np.ceil(ns / batch_size))

        for _ in range(self._iterations):
            if control is not None:
                control.checkpoint()
            order = rng.permutation(ns)
            for b in range(n_batches):
                batch_idx = order[b * batch_size : min((b + 1) * batch_size, ns)]
                grad_U = subspace_weight * 2 * (U - V)
                new_weights = []
                for c, w in zip(classes, weights):
                    y_batch = np.where(source_labels[batch_idx] == c, 1.0, -1.0)
                    scores = y_batch * (source_features[batch_idx] @ U @ w)
                    violated = batch_idx[scores < 1.0]
                    grad_w = 2 * w
                    if len(violated) > 0:
                        y_violated = np.where(source_labels[violated] == c, 1.0, -1.0)
                        grad_w = 2 * w + hinge_weight * np.mean(
                            -(source_features[violated] @ U) * y_violated[:, np.newaxis], axis=0
                        )
                        # sum_j y_j * outer(x_j, w) == outer(X_violated.T @ y_violated, w)
                        grad_U += hinge_weight * np.outer(
                            source_features[violated].T @ y_violated, w
                        ) / len(violated)
                    new_weights.append(w - lr * grad_w)
                weights = new_weights
                U = U - lr * grad_U

            obj_new = objective(U, weights)
            if obj_old - obj_new < 1e-6:
                obj_old = obj_new
                break
            obj_old = obj_new

        transformed_source = source_features @ U
        transformed_target = target_features @ V
        return transformed_source, transformed_target
