"""How much the source and target point clouds intermix, as a percentage.

This is the number behind the Domain Overlap indicator. It answers the
question a reader actually asks of the Feature Space scatter plot — *"are
the two colours sitting on top of each other, or in separate clumps?"* —
with a figure instead of an impression.

**It is a descriptive statistic, not a verdict.** High overlap means the
two distributions occupy the same region of feature space; it does not
mean the transfer helped, and it cannot. A method can shred the class
structure while mixing the domains perfectly, scoring 100% here and
failing on the task. Whether adaptation actually *helped* is the
baseline-vs-adapted comparison in ``NegativeTransferAnalysis``, which is
computed from labels and evaluation scores. Every surface that shows this
number is required to say so alongside it.

Method — normalised nearest-neighbour mixing
--------------------------------------------
Pool both domains, and for every point look at its ``k`` nearest
neighbours. Count what fraction of those neighbours come from the *other*
domain::

    observed = mean over all points of (cross-domain neighbours / k)

Perfectly intermixed domains do not give ``observed == 0.5`` unless the
two sides happen to be the same size. For a source point the chance that
a random other point is a target point is ``nt / (n - 1)``, so the rate
expected under perfect mixing is::

    expected = 2 * ns * nt / (n * (n - 1))

and the reported figure is ``observed / expected``, capped at 100%. That
normalisation is what makes the measure usable here at all: this app is
routinely pointed at very unbalanced pairs (PC3 has 1073 rows, CM1 has
327), where a raw cross-neighbour rate would top out near 34% no matter
how well mixed the domains were, and would read as a failure.

Chosen over a distributional overlap integral because it needs no density
estimate and no Gaussian assumption in a space that may have 800+
dimensions, and over a domain-classifier proxy (A-distance) because that
introduces a second classifier whose own hyper-parameters would end up
confounding the number.

Pure NumPy with no imports from the rest of the app, so this stays in
``domain`` alongside ``mmd.py``: the use case computes it, the
visualisation and the exported report only read the result.
"""

from __future__ import annotations

import numpy as np

#: Neighbours inspected per point. Large enough that a single oddly-placed
#: neighbour cannot swing a point's contribution (k=1 is close to binary
#: noise), small enough to stay a *local* measure — at k comparable to the
#: sample size every point's neighbourhood is the whole cloud and the
#: statistic collapses towards its expected value for any input.
NEIGHBOURS = 10

#: Distances are O(n^2) in the pooled sample, so above this many rows per
#: domain the estimate comes from a fixed-seed subsample. Lower than
#: ``mmd.MAX_SAMPLES_PER_DOMAIN`` because this metric pools the two domains
#: before measuring: 1500 each is a 3000-row pooled set, whose distance
#: rows are computed in chunks (see ``_CHUNK``) rather than as one matrix.
MAX_SAMPLES_PER_DOMAIN = 1500
SUBSAMPLE_SEED = 42

#: Rows of the distance matrix computed at once. Caps peak memory at
#: ``_CHUNK x n`` floats instead of the ``n x n`` a single call would need
#: (~72 MB at the sample cap above, versus ~1.5 MB here).
_CHUNK = 256

#: Shown wherever this figure is displayed. It lives beside the metric
#: rather than in a UI module because it is a statement about what the
#: measure *is*, and because both the Feature Space panel and the exported
#: report need it — a report is read away from the app, often by someone
#: who never saw the scatter plot, so the qualifier has to travel with the
#: number rather than being re-typed at each surface and drifting.
OVERLAP_DISCLAIMER = (
    "A visual/statistical measure of how much the two point clouds intermix — "
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
    """Percentage overlap between *source* and *target* rows, 0-100.

    100 means the two domains are as intermixed as randomly relabelling
    the pooled points would make them; 0 means every point's neighbours
    all come from its own domain (two entirely separate clumps).

    Returns ``0.0`` for degenerate input (either side empty, or fewer than
    two rows in total) rather than raising — a caller drawing a chart
    should not have to guard this.
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
    # 0 = source, 1 = target.
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
        # |a - b|^2 = |a|^2 + |b|^2 - 2a.b, same identity mmd.py uses.
        distances = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + pooled_sq[None, :]
            - 2.0 * (block @ pooled.T)
        )
        # A point is always its own nearest neighbour; exclude it so the
        # count is over genuine neighbours only.
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf

        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        cross_neighbours += int(
            np.count_nonzero(is_target[nearest] != is_target[start:stop, None])
        )

    observed = cross_neighbours / (total * k)
    expected = 2.0 * n_source * n_target / (total * (total - 1))
    if expected <= 0.0:
        return 0.0
    # Capped at 100: a ratio above 1 means the domains interleave *more*
    # than chance (a regular alternating lattice would), which is still
    # just "fully mixed" as far as a reader is concerned.
    return float(min(observed / expected, 1.0) * 100.0)


def overlap_change(before: float, after: float) -> float:
    """After minus before, in **percentage points**.

    Deliberately not the relative percentage change ``mmd.percentage_change``
    uses: MMD is an unbounded distance, where only a ratio is meaningful,
    whereas overlap is already a percentage. Going 40% -> 60% is "+20
    points", and calling that "+50%" would be true but reliably misread.
    """
    if not np.isfinite(before) or not np.isfinite(after):
        return 0.0
    return float(after - before)


__all__ = [
    "MAX_SAMPLES_PER_DOMAIN",
    "OVERLAP_DISCLAIMER",
    "NEIGHBOURS",
    "SUBSAMPLE_SEED",
    "compute_overlap",
    "overlap_change",
]
