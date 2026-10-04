"""Statistical measurement of domain distribution overlap in feature space.

Computes the percentage overlap between source and target sample distributions
using normalized nearest-neighbour mixing analysis.
"""

from __future__ import annotations

import numpy as np

# Number of nearest neighbours inspected per sample.
NEIGHBOURS = 10

# Maximum sample cap per domain before fixed-seed subsampling is applied.
MAX_SAMPLES_PER_DOMAIN = 1500
SUBSAMPLE_SEED = 42

# Block chunk size for memory-efficient distance calculations.
_CHUNK = 256

# Standard user-facing explanation disclaimer.
OVERLAP_DISCLAIMER = (
    "A visual and statistical measure of how much the two point clouds intermix — "
    "not proof that transfer succeeded. Judge that by target accuracy and the "
    "negative-transfer check."
)


def _subsample(matrix: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if matrix.shape[0] <= cap:
        return matrix
    rng = np.random.default_rng(seed)
    indices = rng.choice(matrix.shape[0], size=cap, replace=False)
    return matrix[np.sort(indices)]


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values.reshape(-1, 1) if values.ndim == 1 else values


def compute_overlap(
    source: np.ndarray,
    target: np.ndarray,
    neighbours: int = NEIGHBOURS,
    cap: int = MAX_SAMPLES_PER_DOMAIN,
    seed: int = SUBSAMPLE_SEED,
) -> float:
    """Calculate percentage overlap between source and target point distributions (0 to 100).

    Args:
        source: Source domain feature matrix.
        target: Target domain feature matrix.
        neighbours: Number of nearest neighbours to evaluate per sample.
        cap: Maximum sample count per domain before subsampling.
        seed: Random seed for deterministic subsampling.

    Returns:
        float: Normalized overlap percentage (0.0 to 100.0).
    """
    source = _as_2d(source)
    target = _as_2d(target)
    if source.size == 0 or target.size == 0:
        return 0.0
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            "Domain overlap needs both domains in the same feature space; got "
            f"{source.shape[1]} and {target.shape[1]} columns."
        )

    source = _subsample(source, cap, seed)
    target = _subsample(target, cap, seed)

    n_source, n_target = source.shape[0], target.shape[0]
    total = n_source + n_target
    if total < 2:
        return 0.0

    pooled = np.vstack([source, target])
    is_target = np.concatenate(
        [np.zeros(n_source, dtype=np.int8), np.ones(n_target, dtype=np.int8)]
    )

    k = min(neighbours, total - 1)
    if k < 1:
        return 0.0

    pooled_sq = np.einsum("ij,ij->i", pooled, pooled)
    cross_neighbours = 0

    for start in range(0, total, _CHUNK):
        stop = min(start + _CHUNK, total)
        block = pooled[start:stop]
        distances = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + pooled_sq[None, :]
            - 2.0 * (block @ pooled.T)
        )
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf

        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        cross_neighbours += int(
            np.count_nonzero(is_target[nearest] != is_target[start:stop, None])
        )

    observed = cross_neighbours / (total * k)
    expected = 2.0 * n_source * n_target / (total * (total - 1))
    if expected <= 0.0:
        return 0.0
    return float(min(observed / expected, 1.0) * 100.0)


def overlap_change(before: float, after: float) -> float:
    """Calculate the change in overlap in percentage points (after - before)."""
    if not np.isfinite(before) or not np.isfinite(after):
        return 0.0
    return float(after - before)


__all__ = [
    "MAX_SAMPLES_PER_DOMAIN",
    "NEIGHBOURS",
    "OVERLAP_DISCLAIMER",
    "SUBSAMPLE_SEED",
    "compute_overlap",
    "overlap_change",
]
