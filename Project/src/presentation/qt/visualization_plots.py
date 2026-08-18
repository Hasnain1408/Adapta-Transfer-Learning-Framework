"""Drawing the projected feature space onto a matplotlib axes.

Takes coordinates and a palette and nothing else — no widget state — so the
two colouring modes stay readable side by side. Shape always carries the
domain (circle = source, triangle = target) in both modes, so "did the
domains merge?" survives whatever the colour is being used for.
"""

from __future__ import annotations

import matplotlib
import numpy as np

COLOR_BY_DOMAIN = "Domain"
COLOR_BY_CLASS = "Class"
COLOR_BY_BOTH = "Domain + Class"


def plot_by_domain(
    axes, source_2d: np.ndarray, target_2d: np.ndarray, palette: dict[str, str]
) -> None:
    axes.scatter(
        source_2d[:, 0],
        source_2d[:, 1],
        c=palette.get("accent", "#1e6fd9"),
        alpha=0.6,
        s=34,
        edgecolors="none",
        label=f"Source ({source_2d.shape[0]})",
    )
    axes.scatter(
        target_2d[:, 0],
        target_2d[:, 1],
        c=palette.get("metric_amber", "#c77700"),
        alpha=0.6,
        s=34,
        marker="^",
        edgecolors="none",
        label=f"Target ({target_2d.shape[0]})",
    )


def plot_by_class(
    axes,
    source_2d,
    target_2d,
    source_labels,
    target_labels,
    color_mode: str,
    palette: dict[str, str],
) -> None:
    """Colour by class, keeping the domains distinguishable by marker.

    Shape always carries the domain (circle = source, triangle = target)
    so that "are the class clusters preserved?" and "did the domains
    merge?" stay separately readable in the same plot.
    """
    present = [labels for labels in (source_labels, target_labels) if labels is not None]
    classes = np.unique(np.concatenate(present))
    colormap = matplotlib.colormaps["tab10" if len(classes) <= 10 else "tab20"]

    for position, class_value in enumerate(classes):
        color = colormap(position % colormap.N)
        for labels, coords, marker, domain in (
            (source_labels, source_2d, "o", "Source"),
            (target_labels, target_2d, "^", "Target"),
        ):
            if labels is None:
                continue
            mask = labels == class_value
            if not np.any(mask):
                continue
            # "Domain + Class" additionally rings the target points, so
            # the two domains stay apart even when printed in greyscale.
            edge = "none"
            if color_mode == COLOR_BY_BOTH and domain == "Target":
                edge = palette.get("text", "#e6edf3")
            axes.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=[color],
                marker=marker,
                alpha=0.7,
                s=34,
                linewidths=0.5,
                edgecolors=edge,
                label=f"{domain} · class {class_value} ({int(np.sum(mask))})",
            )


__all__ = [
    "COLOR_BY_BOTH",
    "COLOR_BY_CLASS",
    "COLOR_BY_DOMAIN",
    "plot_by_class",
    "plot_by_domain",
]
