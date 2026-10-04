"""Maximum Mean Discrepancy (MMD) distribution divergence measurement."""

from __future__ import annotations

import numpy as np

# Percentage-change classification thresholds.
IMPROVED_THRESHOLD = 10.0
SLIGHT_THRESHOLD = 2.0

STATUS_IMPROVED = "Improved"
STATUS_SLIGHTLY_IMPROVED = "Slightly Improved"
STATUS_NO_CHANGE = "No Significant Change"
STATUS_WORSENED = "Worsened"

# Maximum samples per domain before subsampling for distance computation.
MAX_SAMPLES_PER_DOMAIN = 2000
SUBSAMPLE_SEED = 42


def _subsample(matrix: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if matrix.shape[0] <= cap:
        return matrix
    rng = np.random.default_rng(seed)
    indices = rng.choice(matrix.shape[0], size=cap, replace=False)
    return matrix[np.sort(indices)]


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between row vectors."""
    left_sq = np.einsum("ij,ij->i", left, left)[:, None]
    right_sq = np.einsum("ij,ij->i", right, right)[None, :]
    distances = left_sq + right_sq - 2.0 * (left @ right.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def _median_bandwidth(source: np.ndarray, target: np.ndarray) -> float:
    """Compute median heuristic RBF kernel bandwidth across pooled samples."""
    pooled = np.vstack([source, target])
    distances = _squared_distances(pooled, pooled)
    upper = distances[np.triu_indices_from(distances, k=1)]
    if upper.size == 0:
        return 1.0
    median = float(np.median(upper))
    return median if median > 1e-12 else 1.0


def compute_mmd(
    source: np.ndarray,
    target: np.ndarray,
    kernel: str = "rbf",
    cap: int = MAX_SAMPLES_PER_DOMAIN,
    seed: int = SUBSAMPLE_SEED,
) -> float:
    """Calculate Maximum Mean Discrepancy distance between source and target sample distributions.

    Args:
        source: Source domain feature matrix.
        target: Target domain feature matrix.
        kernel: Kernel type (rbf with median bandwidth or linear).
        cap: Maximum sample count per domain before subsampling.
        seed: Random seed for deterministic subsampling.

    Returns:
        float: Non-negative MMD distance metric.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim == 1:
        source = source.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    if source.size == 0 or target.size == 0:
        return 0.0
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            "MMD needs both domains in the same feature space; got "
            f"{source.shape[1]} and {target.shape[1]} columns."
        )

    source = _subsample(source, cap, seed)
    target = _subsample(target, cap, seed)

    if kernel == "linear":
        k_ss = source @ source.T
        k_tt = target @ target.T
        k_st = source @ target.T
    else:
        gamma = 1.0 / _median_bandwidth(source, target)
        k_ss = np.exp(-gamma * _squared_distances(source, source))
        k_tt = np.exp(-gamma * _squared_distances(target, target))
        k_st = np.exp(-gamma * _squared_distances(source, target))

    mmd_squared = float(k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean())
    return float(np.sqrt(max(mmd_squared, 0.0)))


def percentage_change(before: float, after: float) -> float:
    """Calculate percentage change in MMD: ((before - after) / before) * 100."""
    if not np.isfinite(before) or abs(before) < 1e-12:
        return 0.0
    return float((before - after) / before * 100.0)


def classify_alignment(change_percent: float) -> str:
    """Classify domain alignment status based on percentage change in MMD."""
    if change_percent >= IMPROVED_THRESHOLD:
        return STATUS_IMPROVED
    if change_percent >= SLIGHT_THRESHOLD:
        return STATUS_SLIGHTLY_IMPROVED
    if change_percent > -SLIGHT_THRESHOLD:
        return STATUS_NO_CHANGE
    return STATUS_WORSENED


__all__ = [
    "IMPROVED_THRESHOLD",
    "MAX_SAMPLES_PER_DOMAIN",
    "SLIGHT_THRESHOLD",
    "STATUS_IMPROVED",
    "STATUS_NO_CHANGE",
    "STATUS_SLIGHTLY_IMPROVED",
    "STATUS_WORSENED",
    "SUBSAMPLE_SEED",
    "classify_alignment",
    "compute_mmd",
    "percentage_change",
]
