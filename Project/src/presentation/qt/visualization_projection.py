"""Projecting a run's feature matrices down to 2D for the scatter plot.

Pure numpy/sklearn — no Qt, no widget state — so the projection maths can be
read and tested apart from the tab that draws it. Both domains are always
fitted together: separate fits would put source and target in different
coordinate systems, and the whole point of the plot is where they sit
relative to each other.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECTION_PCA = "PCA"
PROJECTION_TSNE = "t-SNE"

#: Every projection is seeded with this so the same run always draws the
#: same picture — a scatter that reshuffles on each redraw is not evidence.
RANDOM_SEED = 42

#: Scatter plots stop being readable long before they stop being drawable,
#: and t-SNE is O(n log n) with a large constant. Past these caps a
#: fixed-seed subsample is drawn; the indices depend only on (n, cap, seed),
#: so Before and After keep showing the *same* samples.
MAX_PLOT_SAMPLES = 3000
MAX_TSNE_SAMPLES = 1500

TSNE_DEFAULT_PERPLEXITY = 30
TSNE_DEFAULT_ITERATIONS = 1000


def subsample_indices(count: int, cap: int) -> np.ndarray:
    """Deterministic row indices, so Before and After plot the same samples."""
    if count <= cap:
        return np.arange(count)
    rng = np.random.default_rng(RANDOM_SEED)
    return np.sort(rng.choice(count, size=cap, replace=False))


def as_2d(matrix: np.ndarray) -> np.ndarray:
    """Widen a 1-column feature matrix so a 2-component projection is defined."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[1] == 1:
        return np.column_stack([matrix[:, 0], np.zeros(matrix.shape[0])])
    return matrix


def run_tsne(combined: np.ndarray, perplexity: float, iterations: int) -> np.ndarray:
    # sklearn requires perplexity < n_samples, and the usual guidance is
    # to stay under (n - 1) / 3; a small run would otherwise raise.
    n_samples = combined.shape[0]
    perplexity = min(perplexity, max(2.0, (n_samples - 1) / 3.0))
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    )
    try:
        return TSNE(max_iter=iterations, **kwargs).fit_transform(combined)
    except TypeError:
        # sklearn < 1.5 spells it n_iter.
        return TSNE(n_iter=iterations, **kwargs).fit_transform(combined)


def project_pair(
    source: np.ndarray,
    target: np.ndarray,
    projection: str,
    perplexity: float,
    iterations: int,
    cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a (source, target) pair to 2D, subsampled to *cap* rows each.

    Returns the two halves already split back apart, in the same row order
    ``subsample_indices`` produced — so a caller can line the coordinates up
    against the matching label slices.
    """
    source, target = (as_2d(np.asarray(m, dtype=np.float64)) for m in (source, target))
    src_idx = subsample_indices(source.shape[0], cap)
    tgt_idx = subsample_indices(target.shape[0], cap)
    source, target = source[src_idx], target[tgt_idx]

    # Fitted on both domains at once: separate fits would put the two
    # domains in different coordinate systems, and the whole point is to
    # see where they sit relative to each other.
    combined = np.vstack([source, target])
    if projection == PROJECTION_TSNE:
        coords = run_tsne(combined, perplexity, iterations)
    else:
        coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(combined)

    split = source.shape[0]
    return coords[:split], coords[split:]


__all__ = [
    "MAX_PLOT_SAMPLES",
    "MAX_TSNE_SAMPLES",
    "PROJECTION_PCA",
    "PROJECTION_TSNE",
    "RANDOM_SEED",
    "TSNE_DEFAULT_ITERATIONS",
    "TSNE_DEFAULT_PERPLEXITY",
    "as_2d",
    "project_pair",
    "run_tsne",
    "subsample_indices",
]
