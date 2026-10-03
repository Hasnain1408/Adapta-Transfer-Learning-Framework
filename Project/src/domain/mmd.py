"""Maximum Mean Discrepancy between a source and a target sample.

MMD is the yardstick every method in ``infrastructure/methods/`` is
minimising (TCA/JDA/BDA/DALI/JGSA all build MMD alignment matrices), so
measuring it before and after adaptation says directly whether the method
did what it set out to do.

Pure NumPy and no imports from the rest of the app, so this stays in
``domain``: the use case computes it, the visualisation and the exported
report only read the result.

The estimator here is the **biased** (V-statistic) one::

    MMD^2 = mean(K_ss) + mean(K_tt) - 2 * mean(K_st)

which is always >= 0 — the unbiased U-statistic can come out slightly
negative on well-aligned domains, which would render as a nonsensical
negative distance in the UI and break the percentage-change formula. The
square root is returned so the number reads as a distance.

Note that "before" is measured in the preprocessed input space and "after"
in the method's output space, which are generally different dimensions.
That is the standard comparison in the transfer-learning literature (it is
what "this method reduced the domain gap" means), but it does mean the two
numbers are only meaningful relative to each other, not as absolute
quantities — hence the percentage change rather than a raw difference.
"""

from __future__ import annotations

import numpy as np

#: Percentage-change bands for :func:`classify_alignment`. A drop in MMD is
#: a *positive* percentage change (the domains got closer together).
IMPROVED_THRESHOLD = 10.0
SLIGHT_THRESHOLD = 2.0

STATUS_IMPROVED = "Improved"
STATUS_SLIGHTLY_IMPROVED = "Slightly Improved"
STATUS_NO_CHANGE = "No Significant Change"
STATUS_WORSENED = "Worsened"

#: Kernel matrices are O(n^2); above this many samples per domain the MMD is
#: estimated from a fixed-seed subsample rather than the full set. 2000
#: samples is a 2000x2000 float64 matrix (~32 MB), which is the most we are
#: willing to allocate inside a run.
MAX_SAMPLES_PER_DOMAIN = 2000
SUBSAMPLE_SEED = 42


def _subsample(matrix: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if matrix.shape[0] <= cap:
        return matrix
    rng = np.random.default_rng(seed)
    indices = rng.choice(matrix.shape[0], size=cap, replace=False)
    return matrix[np.sort(indices)]


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, ``|left_i - right_j|^2``."""
    left_sq = np.einsum("ij,ij->i", left, left)[:, None]
    right_sq = np.einsum("ij,ij->i", right, right)[None, :]
    distances = left_sq + right_sq - 2.0 * (left @ right.T)
    # Catastrophic cancellation can push near-zero entries slightly negative.
    np.maximum(distances, 0.0, out=distances)
    return distances


def _median_bandwidth(source: np.ndarray, target: np.ndarray) -> float:
    """Median-heuristic RBF bandwidth over the pooled sample.

    Without this the kernel width would have to be guessed, and MMD is very
    sensitive to it — a fixed gamma that suits the input space would
    saturate to a constant in the method's (differently scaled) output
    space, making before/after incomparable.
    """
    pooled = np.vstack([source, target])
    distances = _squared_distances(pooled, pooled)
    # Off-diagonal only: the zero self-distances would drag the median down.
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
    """MMD between *source* and *target* rows, as a non-negative distance.

    *kernel* is ``"rbf"`` (median-heuristic bandwidth) or ``"linear"``.
    Returns ``0.0`` for degenerate input (either side empty) rather than
    raising — a caller drawing a chart should not have to guard this.
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
    """``((before - after) / before) * 100`` — positive means MMD shrank.

    Returns ``0.0`` when *before* is ~0, where the ratio is undefined and
    there was no domain gap left to close in the first place.
    """
    if not np.isfinite(before) or abs(before) < 1e-12:
        return 0.0
    return float((before - after) / before * 100.0)


def classify_alignment(change_percent: float) -> str:
    """Map a percentage change onto one of the four ``STATUS_*`` labels."""
    if change_percent >= IMPROVED_THRESHOLD:
        return STATUS_IMPROVED
    if change_percent >= SLIGHT_THRESHOLD:
        return STATUS_SLIGHTLY_IMPROVED
    if change_percent > -SLIGHT_THRESHOLD:
        return STATUS_NO_CHANGE
    return STATUS_WORSENED


__all__ = [
    "compute_mmd",
    "percentage_change",
    "classify_alignment",
    "IMPROVED_THRESHOLD",
    "SLIGHT_THRESHOLD",
    "STATUS_IMPROVED",
    "STATUS_SLIGHTLY_IMPROVED",
    "STATUS_NO_CHANGE",
    "STATUS_WORSENED",
    "MAX_SAMPLES_PER_DOMAIN",
    "SUBSAMPLE_SEED",
]
