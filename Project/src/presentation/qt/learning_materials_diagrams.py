"""Teaching diagrams for the Learning Materials dialog.

Each figure is a *schematic*: small synthetic point clouds drawn with a fixed
seed to show what a method does to the geometry of two domains. They are
illustrations, never measurements — every one is captioned as such, because
this app also renders charts of real experiment output a few tabs away and
the two must never be confused.

Referenced from the content HTML as ``{{diagram:key}}`` and substituted at
render time by ``learning_materials_dialog``, so the content stays plain
data and the figures are only built when a topic is actually opened.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np

#: Fixed so a diagram looks identical every time it is opened.
SEED = 7

#: key -> (base64 <img> markup), keyed also by theme so a palette switch
#: redraws rather than serving a figure with the wrong text colour.
_CACHE: dict[tuple[str, str], str] = {}


def _clouds(offset: float = 3.0, spread: float = 0.8, count: int = 60):
    """Two 2-D class-labelled domains, the target shifted away from source."""
    rng = np.random.default_rng(SEED)
    src_a = rng.normal([0, 0], spread, (count, 2))
    src_b = rng.normal([3, 2.4], spread, (count, 2))
    tgt_a = rng.normal([0 + offset, 0 + offset * 0.5], spread, (count, 2))
    tgt_b = rng.normal([3 + offset, 2.4 + offset * 0.5], spread, (count, 2))
    return src_a, src_b, tgt_a, tgt_b


def _style(axes, palette: dict[str, str], title: str = "") -> None:
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color(palette["border"])
    axes.set_facecolor(palette["surface_alt"])
    if title:
        axes.set_title(title, fontsize=8.5, fontweight="bold", color=palette["text"], pad=5)


def _scatter_domains(axes, palette, src_a, src_b, tgt_a, tgt_b, show_classes=True):
    src_colour = palette["accent"]
    tgt_colour = palette["metric_amber"]
    if show_classes:
        axes.scatter(*src_a.T, s=9, c=src_colour, marker="o", alpha=0.75)
        axes.scatter(*src_b.T, s=9, c=src_colour, marker="s", alpha=0.75)
        axes.scatter(*tgt_a.T, s=9, c=tgt_colour, marker="^", alpha=0.75)
        axes.scatter(*tgt_b.T, s=9, c=tgt_colour, marker="D", alpha=0.75)
    else:
        axes.scatter(*np.vstack([src_a, src_b]).T, s=9, c=src_colour, alpha=0.75)
        axes.scatter(*np.vstack([tgt_a, tgt_b]).T, s=9, c=tgt_colour, alpha=0.75)


def _new_figure(palette: dict[str, str], width: float = 6.4, height: float = 2.5) -> Figure:
    figure = Figure(figsize=(width, height), dpi=110)
    figure.patch.set_facecolor(palette["surface"])
    try:
        figure.set_layout_engine("constrained")
    except AttributeError:  # matplotlib < 3.6
        figure.set_constrained_layout(True)
    return figure


def _suptitle(figure, palette, text: str) -> None:
    """One headline across the whole figure.

    Two side-by-side panels cannot each carry a full sentence — the titles
    run into each other at the boundary — so the explanation lives here and
    the panels get two-word labels.
    """
    figure.suptitle(text, fontsize=9, fontweight="bold", color=palette["text"])


def _legend(figure, palette, entries: list[tuple[str, str]]) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5, color=colour, label=label)
        for label, colour in entries
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(entries),
        fontsize=7,
        frameon=False,
        labelcolor=palette["text_secondary"],
    )


# ----------------------------------------------------------------------
# Core concept diagrams
# ----------------------------------------------------------------------
def _domain_shift(palette):
    figure = _new_figure(palette, height=2.4)
    axes = figure.add_subplot(111)
    src_a, src_b, tgt_a, tgt_b = _clouds()
    _scatter_domains(axes, palette, src_a, src_b, tgt_a, tgt_b, show_classes=False)
    axes.annotate(
        "", xy=(4.6, 2.4), xytext=(1.5, 1.2),
        arrowprops=dict(arrowstyle="->", color=palette["danger"], lw=1.6),
    )
    axes.text(3.0, 3.4, "domain gap", fontsize=8, color=palette["danger"], fontweight="bold")
    _style(axes, palette, "The problem: a classifier fitted on blue is tested on orange")
    _legend(figure, palette, [("Source domain", palette["accent"]), ("Target domain", palette["metric_amber"])])
    return figure


def _mmd(palette):
    figure = _new_figure(palette)
    src_a, src_b, tgt_a, tgt_b = _clouds()
    for index, (offset, label) in enumerate(((3.0, "High MMD"), (0.35, "Low MMD"))):
        axes = figure.add_subplot(1, 2, index + 1)
        a, b, c, d = _clouds(offset=offset)
        _scatter_domains(axes, palette, a, b, c, d, show_classes=False)
        source_mean = np.vstack([a, b]).mean(axis=0)
        target_mean = np.vstack([c, d]).mean(axis=0)
        axes.plot(*zip(source_mean, target_mean), color=palette["danger"], lw=1.5, ls="--")
        axes.scatter(*source_mean, s=60, marker="X", c=palette["danger"], zorder=5)
        axes.scatter(*target_mean, s=60, marker="X", c=palette["danger"], zorder=5)
        _style(axes, palette, label)
    _suptitle(figure, palette, "MMD measures how far apart the two distributions sit")
    _legend(figure, palette, [("Source", palette["accent"]), ("Target", palette["metric_amber"]),
                              ("Distribution mean", palette["danger"])])
    return figure


def _negative_transfer(palette):
    figure = _new_figure(palette, height=2.3)
    axes = figure.add_subplot(111)
    methods = ["Method A", "Method B", "Method C"]
    baseline = [0.70, 0.70, 0.70]
    adapted = [0.78, 0.71, 0.61]
    positions = np.arange(len(methods))
    axes.bar(positions - 0.2, baseline, 0.4, color=palette["text_muted"], label="No adaptation")
    axes.bar(positions + 0.2, adapted, 0.4,
             color=[palette["success"], palette["success"], palette["danger"]], label="After transfer")
    for position, (base, new) in enumerate(zip(baseline, adapted)):
        axes.text(position + 0.2, new + 0.02, f"{new - base:+.2f}", ha="center", fontsize=7.5,
                  fontweight="bold", color=palette["success"] if new >= base else palette["danger"])
    axes.text(2.2, 0.30, "negative\ntransfer", fontsize=8, color=palette["danger"], fontweight="bold")
    axes.set_xticks(positions)
    axes.set_xticklabels(methods, fontsize=8, color=palette["text_secondary"])
    axes.set_ylim(0, 0.95)
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color(palette["border"])
    axes.set_facecolor(palette["surface_alt"])
    axes.set_title("Transfer can hurt: Method C scores below the baseline",
                   fontsize=8.5, fontweight="bold", color=palette["text"], pad=5)
    axes.legend(fontsize=7, frameon=False, labelcolor=palette["text_secondary"], loc="upper left")
    return figure


def _pca_vs_tsne(palette):
    figure = _new_figure(palette)
    rng = np.random.default_rng(SEED)
    for index, title in enumerate(("PCA — global, linear", "t-SNE — local clusters")):
        axes = figure.add_subplot(1, 2, index + 1)
        if index == 0:
            blob = rng.normal(0, 1, (140, 2)) @ np.array([[2.4, 0.9], [0.0, 0.5]])
            axes.scatter(*blob.T, s=9, c=palette["accent"], alpha=0.7)
            axes.annotate("", xy=(4.2, 1.7), xytext=(-4.2, -1.7),
                          arrowprops=dict(arrowstyle="->", color=palette["danger"], lw=1.5))
            axes.text(1.4, -1.9, "PC1", fontsize=8, color=palette["danger"], fontweight="bold")
        else:
            centres = [(-2.2, 1.6), (2.2, 1.6), (0, -2.0)]
            for centre, colour in zip(centres, ("metric_teal", "metric_purple", "metric_pink")):
                axes.scatter(*rng.normal(centre, 0.42, (46, 2)).T, s=9,
                             c=palette[colour], alpha=0.8)
        _style(axes, palette, title)
    _suptitle(figure, palette, "Two ways to flatten many dimensions down to two")
    return figure


# ----------------------------------------------------------------------
# Method diagrams
# ----------------------------------------------------------------------
def _before_after(palette, after_offset, headline, after_label):
    """Shared 'two panels: shifted, then aligned' layout most methods share."""
    figure = _new_figure(palette, height=2.6)
    for index, (offset, title) in enumerate(((3.0, "Before"), (after_offset, after_label))):
        axes = figure.add_subplot(1, 2, index + 1)
        _scatter_domains(axes, palette, *_clouds(offset=offset))
        _style(axes, palette, title)
    _suptitle(figure, palette, headline)
    _legend(figure, palette, [("Source class 1/2", palette["accent"]),
                              ("Target class 1/2", palette["metric_amber"])])
    return figure


def _tca(palette):
    return _before_after(
        palette, 0.4,
        "TCA finds a subspace where the two domains overlap",
        "After TCA",
    )


def _jda(palette):
    figure = _new_figure(palette, height=2.6)
    axes = figure.add_subplot(1, 2, 1)
    _scatter_domains(axes, palette, *_clouds(offset=3.0))
    _style(axes, palette, "Before")
    axes = figure.add_subplot(1, 2, 2)
    src_a, src_b, tgt_a, tgt_b = _clouds(offset=0.25)
    _scatter_domains(axes, palette, src_a, src_b, tgt_a, tgt_b)
    for cloud_pair, colour in (((src_a, tgt_a), "metric_teal"), ((src_b, tgt_b), "metric_pink")):
        centre = np.vstack(cloud_pair).mean(axis=0)
        axes.add_patch(Ellipse(centre, 3.0, 2.4, facecolor="none",
                               edgecolor=palette[colour], lw=1.3, ls="--"))
    _style(axes, palette, "After JDA")
    _suptitle(figure, palette, "JDA also matches each class across the domains")
    _legend(figure, palette, [("Source", palette["accent"]), ("Target", palette["metric_amber"]),
                              ("Matched class", palette["metric_teal"])])
    return figure


def _bda(palette):
    figure = _new_figure(palette, height=2.4)
    axes = figure.add_subplot(111)
    mu = np.linspace(0, 1, 100)
    axes.plot(mu, 1 - mu, color=palette["accent"], lw=2, label="Marginal weight (1 − μ)")
    axes.plot(mu, mu, color=palette["metric_pink"], lw=2, label="Conditional weight (μ)")
    axes.axvline(0.5, color=palette["text_muted"], ls=":", lw=1.2)
    axes.text(0.5, 1.03, "balanced", ha="center", fontsize=7.5, color=palette["text_secondary"])
    axes.text(0.02, 0.9, "μ=0\nTCA-like", fontsize=7.5, color=palette["accent"])
    axes.text(0.86, 0.9, "μ=1\nclass-focused", fontsize=7.5, color=palette["metric_pink"])
    axes.set_xlabel("μ (balance factor)", fontsize=8, color=palette["text_secondary"])
    axes.set_ylim(0, 1.15)
    axes.set_yticks([])
    axes.tick_params(colors=palette["text_secondary"], labelsize=7)
    for spine in axes.spines.values():
        spine.set_color(palette["border"])
    axes.set_facecolor(palette["surface_alt"])
    axes.set_title("BDA's μ trades marginal alignment against class alignment",
                   fontsize=8.5, fontweight="bold", color=palette["text"], pad=5)
    axes.legend(fontsize=7, frameon=False, labelcolor=palette["text_secondary"], loc="center right")
    return figure


def _coral(palette):
    figure = _new_figure(palette)
    rng = np.random.default_rng(SEED)
    source = rng.normal(0, 1, (110, 2)) @ np.array([[1.6, 0.8], [0.0, 0.6]])
    target_raw = rng.normal(0, 1, (110, 2)) @ np.array([[0.6, -0.9], [0.0, 1.5]]) + [3.0, 0.5]
    for index, (data, title) in enumerate((
        (target_raw, "Before"),
        (source + [3.0, 0.5] + rng.normal(0, 0.12, source.shape), "After CORAL"),
    )):
        axes = figure.add_subplot(1, 2, index + 1)
        axes.scatter(*source.T, s=9, c=palette["accent"], alpha=0.7)
        axes.scatter(*data.T, s=9, c=palette["metric_amber"], alpha=0.7)
        for cloud, colour in ((source, "accent"), (data, "metric_amber")):
            centre = cloud.mean(axis=0)
            values, vectors = np.linalg.eigh(np.cov(cloud.T))
            angle = np.degrees(np.arctan2(*vectors[:, 1][::-1]))
            axes.add_patch(Ellipse(centre, 2 * 2 * np.sqrt(values[1]), 2 * 2 * np.sqrt(values[0]),
                                   angle=angle, facecolor="none", edgecolor=palette[colour], lw=1.4))
        _style(axes, palette, title)
    _suptitle(figure, palette, "CORAL re-shapes the target's covariance to match the source")
    _legend(figure, palette, [("Source", palette["accent"]), ("Target", palette["metric_amber"])])
    return figure


def _gfk(palette):
    figure = _new_figure(palette, height=2.5)
    axes = figure.add_subplot(111)
    angles = np.linspace(np.pi * 0.85, np.pi * 0.25, 6)
    for index, angle in enumerate(angles):
        colour = palette["accent"] if index == 0 else (
            palette["metric_amber"] if index == len(angles) - 1 else palette["text_muted"]
        )
        width = 2.0 if index in (0, len(angles) - 1) else 1.0
        axes.plot([0, np.cos(angle) * 2], [0, np.sin(angle) * 2], color=colour, lw=width,
                  alpha=1.0 if index in (0, len(angles) - 1) else 0.55)
    arc = np.linspace(np.pi * 0.85, np.pi * 0.25, 60)
    axes.plot(np.cos(arc) * 2.25, np.sin(arc) * 2.25, color=palette["metric_teal"], ls="--", lw=1.3)
    axes.text(-1.9, 1.75, "source\nsubspace", fontsize=7.5, color=palette["accent"])
    axes.text(1.35, 1.65, "target\nsubspace", fontsize=7.5, color=palette["metric_amber"])
    axes.text(-0.35, 2.45, "geodesic path — GFK integrates over every step",
              fontsize=7.5, color=palette["metric_teal"], ha="center")
    axes.set_xlim(-2.6, 2.6)
    axes.set_ylim(-0.2, 2.8)
    _style(axes, palette, "GFK walks from the source subspace to the target one")
    return figure


def _jgsa(palette):
    figure = _new_figure(palette, height=2.6)
    axes = figure.add_subplot(1, 2, 1)
    _scatter_domains(axes, palette, *_clouds(offset=3.0))
    _style(axes, palette, "Before")
    axes = figure.add_subplot(1, 2, 2)
    src_a, src_b, tgt_a, tgt_b = _clouds(offset=0.2, spread=0.45)
    _scatter_domains(axes, palette, src_a, src_b, tgt_a, tgt_b)
    axes.annotate("", xy=(3.0, 2.4), xytext=(0.0, 0.0),
                  arrowprops=dict(arrowstyle="<->", color=palette["success"], lw=1.5, ls="--"))
    axes.text(1.0, 1.6, "wider class\nmargin", fontsize=7.5, color=palette["success"], fontweight="bold")
    _style(axes, palette, "After JGSA")
    _suptitle(figure, palette, "JGSA aligns the domains and separates the classes at once")
    _legend(figure, palette, [("Source", palette["accent"]), ("Target", palette["metric_amber"])])
    return figure


def _jcsl(palette):
    figure = _new_figure(palette, height=2.6)
    for index, (spread, title) in enumerate(((0.85, "Before"), (0.4, "After JCSL"))):
        axes = figure.add_subplot(1, 2, index + 1)
        offset = 3.0 if index == 0 else 0.2
        src_a, src_b, tgt_a, tgt_b = _clouds(offset=offset, spread=spread)
        _scatter_domains(axes, palette, src_a, src_b, tgt_a, tgt_b)
        if index == 1:
            for cloud_pair, colour in (((src_a, tgt_a), "metric_teal"), ((src_b, tgt_b), "metric_pink")):
                centre = np.vstack(cloud_pair).mean(axis=0)
                axes.add_patch(Ellipse(centre, 1.9, 1.6, facecolor="none",
                                       edgecolor=palette[colour], lw=1.3))
        _style(axes, palette, title)
    _suptitle(figure, palette, "JCSL keeps each class compact while closing the domain gap")
    _legend(figure, palette, [("Source", palette["accent"]), ("Target", palette["metric_amber"])])
    return figure


def _ocmtl(palette):
    figure = _new_figure(palette, height=2.5)
    axes = figure.add_subplot(111)
    rng = np.random.default_rng(SEED)
    class_a = rng.normal([-1.5, 0], 0.5, (70, 2))
    class_b = rng.normal([1.5, 0], 0.5, (70, 2))
    axes.scatter(*class_a.T, s=10, c=palette["accent"], alpha=0.8)
    axes.scatter(*class_b.T, s=10, c=palette["metric_pink"], alpha=0.8)
    axes.axvline(0, color=palette["text"], lw=1.6)
    axes.axvspan(-0.75, 0.75, color=palette["success"], alpha=0.18)
    axes.annotate("", xy=(0.75, 1.5), xytext=(-0.75, 1.5),
                  arrowprops=dict(arrowstyle="<->", color=palette["success"], lw=1.5))
    axes.text(0, 1.72, "margin OCMTL maximises", ha="center", fontsize=8,
              color=palette["success"], fontweight="bold")
    axes.set_ylim(-1.8, 2.1)
    _style(axes, palette, "OCMTL widens the gap between classes while transferring")
    return figure


def _dali(palette):
    figure = _new_figure(palette, width=6.6, height=2.6)
    axes = figure.add_subplot(111)
    axes.axis("off")
    steps = ["Train on\nsource", "Pseudo-label\ntarget", "Re-align\ndomains", "Refine\nlabels"]
    for index, text in enumerate(steps):
        x = 0.09 + index * 0.24
        axes.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, 0.42), 0.16, 0.30, boxstyle="round,pad=0.02",
            facecolor=palette["surface_alt"], edgecolor=palette["accent"], lw=1.3,
            transform=axes.transAxes))
        axes.text(x + 0.08, 0.57, text, ha="center", va="center", fontsize=7.5,
                  color=palette["text"], transform=axes.transAxes)
        if index < len(steps) - 1:
            axes.add_patch(FancyArrowPatch(
                (x + 0.17, 0.57), (x + 0.235, 0.57), transform=axes.transAxes,
                arrowstyle="->", color=palette["text_secondary"], lw=1.3, mutation_scale=11))
    # Negative rad bows the return path downward; a positive one arcs it
    # straight back through the four boxes.
    axes.add_patch(FancyArrowPatch(
        (0.81, 0.40), (0.17, 0.40), transform=axes.transAxes, arrowstyle="->",
        color=palette["metric_teal"], lw=1.4, mutation_scale=11,
        connectionstyle="arc3,rad=-0.22"))
    axes.text(0.5, 0.02, "repeat until the pseudo-labels stop changing",
              ha="center", fontsize=7.5, color=palette["metric_teal"], transform=axes.transAxes)
    axes.set_title("DALI refines target labels and alignment together, in a loop",
                   fontsize=8.5, fontweight="bold", color=palette["text"])
    return figure


def _meda(palette):
    figure = _new_figure(palette, height=2.6)
    axes = figure.add_subplot(1, 2, 1)
    rng = np.random.default_rng(SEED)
    theta = np.linspace(0, np.pi, 70)
    axes.plot(np.cos(theta) * 2, np.sin(theta) * 1.2, color=palette["metric_teal"], lw=1.6, ls="--")
    axes.scatter(np.cos(theta[::6]) * 2 + rng.normal(0, .07, 12),
                 np.sin(theta[::6]) * 1.2 + rng.normal(0, .07, 12),
                 s=14, c=palette["accent"])
    axes.scatter(np.cos(theta[3::6]) * 2 + rng.normal(0, .07, 12),
                 np.sin(theta[3::6]) * 1.2 + rng.normal(0, .07, 12),
                 s=14, c=palette["metric_amber"], marker="^")
    axes.text(0, -0.45, "features live on a curved manifold", ha="center",
              fontsize=7.5, color=palette["metric_teal"])
    axes.set_ylim(-0.7, 1.6)
    _style(axes, palette, "1. Manifold feature learning")
    axes = figure.add_subplot(1, 2, 2)
    mu = np.linspace(0, 1, 100)
    axes.plot(mu, 1 - mu, color=palette["accent"], lw=2)
    axes.plot(mu, mu, color=palette["metric_pink"], lw=2)
    axes.scatter([0.35], [0.65], s=45, c=palette["success"], zorder=5)
    axes.text(0.37, 0.72, "μ estimated\nfrom the data", fontsize=7.5, color=palette["success"])
    axes.set_xlabel("μ", fontsize=8, color=palette["text_secondary"])
    axes.tick_params(colors=palette["text_secondary"], labelsize=7)
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color(palette["border"])
    axes.set_facecolor(palette["surface_alt"])
    axes.set_title("2. Dynamic distribution alignment", fontsize=8.5,
                   fontweight="bold", color=palette["text"], pad=5)
    _suptitle(figure, palette, "MEDA learns manifold features, then picks μ itself")
    return figure


# ----------------------------------------------------------------------
# Classifier / metric diagrams
# ----------------------------------------------------------------------
def _knn(palette):
    figure = _new_figure(palette, height=2.5)
    axes = figure.add_subplot(111)
    rng = np.random.default_rng(SEED)
    class_a = rng.normal([-1.3, 0.2], 0.65, (28, 2))
    class_b = rng.normal([1.4, 0.1], 0.65, (28, 2))
    axes.scatter(*class_a.T, s=22, c=palette["accent"], alpha=0.85, label="Class 1")
    axes.scatter(*class_b.T, s=22, c=palette["metric_pink"], alpha=0.85, label="Class 2")
    query = np.array([0.25, 0.35])
    axes.scatter(*query, s=90, marker="*", c=palette["success"], zorder=6, label="New sample")
    everything = np.vstack([class_a, class_b])
    distances = np.linalg.norm(everything - query, axis=1)
    radius = np.sort(distances)[4]
    axes.add_patch(matplotlib.patches.Circle(query, radius, facecolor="none",
                                             edgecolor=palette["success"], ls="--", lw=1.4))
    axes.text(query[0], query[1] + radius + 0.16, "5 nearest neighbours vote",
              ha="center", fontsize=7.5, color=palette["success"], fontweight="bold")
    _style(axes, palette, "k-NN labels a sample by asking its closest neighbours")
    axes.legend(fontsize=7, frameon=False, labelcolor=palette["text_secondary"], loc="lower right")
    return figure


def _confusion(palette):
    figure = _new_figure(palette, width=5.6, height=2.5)
    axes = figure.add_subplot(111)
    axes.axis("off")
    cells = [("True Positive", "predicted yes,\nreally yes", "success"),
             ("False Positive", "predicted yes,\nreally no", "danger"),
             ("False Negative", "predicted no,\nreally yes", "danger"),
             ("True Negative", "predicted no,\nreally no", "success")]
    for index, (title, subtitle, colour) in enumerate(cells):
        row, column = divmod(index, 2)
        x, y = 0.24 + column * 0.34, 0.54 - row * 0.34
        axes.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, y), 0.30, 0.28, boxstyle="round,pad=0.012", transform=axes.transAxes,
            facecolor=palette["surface_alt"], edgecolor=palette[colour], lw=1.4))
        axes.text(x + 0.15, y + 0.19, title, ha="center", fontsize=8,
                  fontweight="bold", color=palette[colour], transform=axes.transAxes)
        axes.text(x + 0.15, y + 0.08, subtitle, ha="center", fontsize=7,
                  color=palette["text_secondary"], transform=axes.transAxes)
    axes.text(0.58, 0.90, "Predicted", ha="center", fontsize=8,
              color=palette["text"], fontweight="bold", transform=axes.transAxes)
    axes.text(0.16, 0.47, "Actual", rotation=90, va="center", fontsize=8,
              color=palette["text"], fontweight="bold", transform=axes.transAxes)
    return figure


def _precision_recall(palette):
    figure = _new_figure(palette, width=6.0, height=2.4)
    axes = figure.add_subplot(111)
    axes.axis("off")
    rng = np.random.default_rng(SEED)
    positives = rng.normal([0.34, 0.5], [0.09, 0.16], (34, 2))
    negatives = rng.normal([0.66, 0.5], [0.09, 0.16], (34, 2))
    axes.scatter(*positives.T, s=16, c=palette["success"], transform=axes.transAxes)
    axes.scatter(*negatives.T, s=16, c=palette["text_muted"], transform=axes.transAxes)
    axes.add_patch(matplotlib.patches.Ellipse((0.44, 0.5), 0.34, 0.66, transform=axes.transAxes,
                                              facecolor="none", edgecolor=palette["accent"], lw=1.8))
    axes.text(0.44, 0.90, "what the model predicted positive", ha="center", fontsize=7.5,
              color=palette["accent"], transform=axes.transAxes)
    axes.text(0.02, 0.5, "Precision =\nright ÷ predicted", fontsize=7.5,
              color=palette["accent"], transform=axes.transAxes, va="center")
    axes.text(0.80, 0.5, "Recall =\nfound ÷ all real", fontsize=7.5,
              color=palette["success"], transform=axes.transAxes, va="center")
    return figure


def _roc(palette):
    figure = _new_figure(palette, width=5.2, height=2.7)
    axes = figure.add_subplot(111)
    fpr = np.linspace(0, 1, 100)
    for power, label, colour in ((0.28, "Strong (AUC ≈ 0.93)", "success"),
                                 (0.62, "Fair (AUC ≈ 0.74)", "metric_amber")):
        axes.plot(fpr, fpr ** power, color=palette[colour], lw=1.8, label=label)
    axes.plot(fpr, fpr, ls="--", lw=1.3, color=palette["text_muted"], label="Random (AUC = 0.50)")
    axes.set_xlabel("False positive rate", fontsize=8, color=palette["text_secondary"])
    axes.set_ylabel("True positive rate", fontsize=8, color=palette["text_secondary"])
    axes.tick_params(colors=palette["text_secondary"], labelsize=7)
    for spine in axes.spines.values():
        spine.set_color(palette["border"])
    axes.set_facecolor(palette["surface_alt"])
    axes.set_title("AUC is the area under this curve", fontsize=8.5,
                   fontweight="bold", color=palette["text"], pad=5)
    axes.legend(fontsize=7, frameon=False, labelcolor=palette["text_secondary"], loc="lower right")
    return figure


#: Diagram key -> builder. Keys are what content HTML references.
BUILDERS = {
    "domain_shift": _domain_shift,
    "mmd": _mmd,
    "negative_transfer": _negative_transfer,
    "pca_vs_tsne": _pca_vs_tsne,
    "tca": _tca,
    "jda": _jda,
    "bda": _bda,
    "coral": _coral,
    "gfk": _gfk,
    "jgsa": _jgsa,
    "jcsl": _jcsl,
    "ocmtl": _ocmtl,
    "dali": _dali,
    "meda": _meda,
    "knn": _knn,
    "confusion": _confusion,
    "precision_recall": _precision_recall,
    "roc": _roc,
}


def diagram_html(key: str, palette: dict[str, str]) -> str:
    """``<img>`` markup for *key*, built once per key and theme.

    Unknown keys render as nothing rather than raising — a typo in the
    content should not take the whole dialog down.
    """
    builder = BUILDERS.get(key)
    if builder is None:
        return ""
    cache_key = (key, palette.get("surface", ""))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    figure = builder(palette)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor=palette["surface"], bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    markup = (
        f"<p><img src='data:image/png;base64,{encoded}' width='620'/></p>"
        "<p class='diagram-note'>Schematic illustration — not output from your experiments.</p>"
    )
    _CACHE[cache_key] = markup
    return markup


__all__ = ["diagram_html", "BUILDERS"]
