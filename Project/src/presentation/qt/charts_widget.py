from __future__ import annotations

import base64
import io
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from domain.models import TransferLearningResult
from presentation.qt.style import PALETTE


class AnalyticsChartsWidget(QWidget):
    """Widget embedding Matplotlib charts for visual experiment analytics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_result: Optional[TransferLearningResult] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        # ── Control Bar ──────────────────────────────────────────────────
        control_bar = QHBoxLayout()
        control_bar.setSpacing(10)

        view_label = QLabel("View:")
        view_label.setObjectName("fieldLabel")

        self._view_combo = QComboBox()
        self._view_combo.addItems([
            "Overview (All Charts)",
            "Domain Feature Alignment",
            "Confusion Matrix",
            "Performance Metrics",
        ])
        self._view_combo.setToolTip("Select chart layout view")
        self._view_combo.currentTextChanged.connect(self._on_view_changed)

        control_bar.addWidget(view_label)
        control_bar.addWidget(self._view_combo)
        control_bar.addStretch(1)

        # ── Canvas & Toolbar ─────────────────────────────────────────────
        self._figure = Figure(figsize=(9, 6), dpi=100, facecolor="#ffffff")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setStyleSheet("background-color: #ffffff; border-radius: 6px;")

        self._toolbar = NavigationToolbar(self._canvas, self)
        self._toolbar.setStyleSheet("background: transparent; border: none;")

        layout.addLayout(control_bar)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._toolbar)

        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Run an experiment to display visual charts & analytics",
            ha="center",
            va="center",
            fontsize=13,
            color=PALETTE.get("text_muted", "#8a97a5"),
            transform=ax.transAxes,
        )
        self._canvas.draw()

    def update_result(self, result: TransferLearningResult) -> None:
        self._last_result = result
        self._render_charts()

    def _on_view_changed(self, text: str) -> None:
        if self._last_result is not None:
            self._render_charts()

    def _render_charts(self) -> None:
        result = self._last_result
        if result is None:
            self._show_placeholder()
            return

        self._figure.clear()
        view_mode = self._view_combo.currentText()

        if view_mode == "Overview (All Charts)":
            self._render_overview(result)
        elif view_mode == "Domain Feature Alignment":
            ax = self._figure.add_subplot(111)
            self._plot_domain_alignment(ax, result)
        elif view_mode == "Confusion Matrix":
            ax = self._figure.add_subplot(111)
            self._plot_confusion_matrix(ax, result)
        elif view_mode == "Performance Metrics":
            ax = self._figure.add_subplot(111)
            self._plot_metrics(ax, result)

        self._figure.tight_layout()
        self._canvas.draw()

    def _render_overview(self, result: TransferLearningResult) -> None:
        has_metrics = result.metrics is not None and result.target_predictions is not None
        if has_metrics:
            ax1 = self._figure.add_subplot(2, 2, 1)
            self._plot_domain_alignment(ax1, result)

            ax2 = self._figure.add_subplot(2, 2, 2)
            self._plot_confusion_matrix(ax2, result)

            ax3 = self._figure.add_subplot(2, 2, (3, 4))
            self._plot_metrics(ax3, result)
        else:
            ax = self._figure.add_subplot(111)
            self._plot_domain_alignment(ax, result)

    def _plot_domain_alignment(self, ax, result: TransferLearningResult) -> None:
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

        accent_col = PALETTE.get("accent", "#1e6fd9")
        success_col = PALETTE.get("success", "#14804a")
        danger_col = PALETTE.get("danger", "#b3261e")

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

    def _plot_confusion_matrix(self, ax, result: TransferLearningResult) -> None:
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
        self._figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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

    def _plot_metrics(self, ax, result: TransferLearningResult) -> None:
        metrics = result.metrics
        if metrics is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "Metrics unavailable", ha="center", va="center")
            return

        names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        values = [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score]
        colors = ["#1e6fd9", "#2bb068", "#f59e0b", "#8b5cf6"]

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

    def render_to_base64_html(self) -> str:
        """Render the current figure to a base64 encoded PNG for HTML/PDF reports."""
        if self._last_result is None:
            return ""

        buf = io.BytesIO()
        self._figure.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        return f"<img src='data:image/png;base64,{encoded}' style='width:100%; max-width:800px; margin-top:10px;' />"
