"""The three analytics charts, drawn onto a caller-supplied axes.

Pure matplotlib over a ``TransferLearningResult`` and a palette — no widget
state — so each chart can be read on its own and reused wherever a figure is
available (the tab draws them live; the PDF export re-renders the same
figure).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from domain.models import TransferLearningResult


def plot_domain_alignment(ax, result: TransferLearningResult, palette: dict[str, str]) -> None:
    """Source vs. target in the transformed space, target split by correctness."""
    src = result.transformed_source
    tgt = result.transformed_target

    if src.shape[1] > 2:
        pca = PCA(n_components=2)
        all_feats = np.vstack([src, tgt])
        pca.fit(all_feats)
        src_2d = pca.transform(src)
        tgt_2d = pca.transform(tgt)
        xlabel, ylabel = "PCA Component 1", "PCA Component 2"
    elif src.shape[1] == 2:
        src_2d, tgt_2d = src, tgt
        xlabel, ylabel = "Component 1", "Component 2"
    else:
        # 1D feature space
        src_2d = np.column_stack([src[:, 0], np.zeros(src.shape[0])])
        tgt_2d = np.column_stack([tgt[:, 0], np.zeros(tgt.shape[0])])
        xlabel, ylabel = "Component 1", ""

    accent_col = palette.get("accent", "#1e6fd9")
    success_col = palette.get("success", "#14804a")
    danger_col = palette.get("danger", "#b3261e")

    ax.scatter(
        src_2d[:, 0],
        src_2d[:, 1],
        c=accent_col,
        alpha=0.6,
        edgecolors="none",
        s=40,
        label=f"Source ({src.shape[0]} samples)",
    )

    if result.prediction_correctness is not None:
        correct_mask = np.array([c == "Right" for c in result.prediction_correctness])
        ax.scatter(
            tgt_2d[correct_mask, 0],
            tgt_2d[correct_mask, 1],
            c=success_col,
            marker="^",
            alpha=0.8,
            s=45,
            label=f"Target Correct ({np.sum(correct_mask)})",
        )
        ax.scatter(
            tgt_2d[~correct_mask, 0],
            tgt_2d[~correct_mask, 1],
            c=danger_col,
            marker="x",
            alpha=0.9,
            s=55,
            linewidths=1.5,
            label=f"Target Incorrect ({np.sum(~correct_mask)})",
        )
    else:
        ax.scatter(
            tgt_2d[:, 0],
            tgt_2d[:, 1],
            c=success_col,
            marker="^",
            alpha=0.7,
            s=45,
            label=f"Target ({tgt.shape[0]} samples)",
        )

    ax.set_title("Transformed Feature Space Alignment", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def plot_confusion_matrix(figure, ax, result: TransferLearningResult) -> None:
    """Target confusion matrix. *figure* is needed only for the colour bar."""
    if result.target_actuals is None or result.target_predictions is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No target labels available", ha="center", va="center")
        return

    y_true = result.target_actuals
    y_pred = result.target_predictions
    labels = np.unique(np.concatenate([y_true, y_pred]))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_title("Target Classification Confusion Matrix", fontsize=11, fontweight="bold", pad=8)
    figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([str(lbl) for lbl in labels], fontsize=8, rotation=45 if len(str(labels[0])) > 5 else 0)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels([str(lbl) for lbl in labels], fontsize=8)

    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("Actual Label", fontsize=9)

    # Annotate cell counts
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            ax.text(
                j,
                i,
                f"{count}",
                ha="center",
                va="center",
                color="white" if count > thresh else "black",
                fontsize=10,
                fontweight="bold",
            )


def plot_metrics(ax, result: TransferLearningResult, palette: dict[str, str]) -> None:
    """The four headline scores as a labelled bar chart."""
    metrics = result.metrics
    if metrics is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Metrics unavailable", ha="center", va="center")
        return

    names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score]
    colors = [
        palette["metric_teal"],
        palette["metric_purple"],
        palette["metric_amber"],
        palette["metric_pink"],
    ]

    bars = ax.bar(names, values, color=colors, width=0.55, edgecolor="none", alpha=0.85)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Target Domain Evaluation Metrics", fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


__all__ = ["plot_confusion_matrix", "plot_domain_alignment", "plot_metrics"]
