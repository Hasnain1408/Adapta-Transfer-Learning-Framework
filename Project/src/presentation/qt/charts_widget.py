from __future__ import annotations

import base64
import io
from typing import Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
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

from domain.models import TransferLearningResult
from presentation.qt.charts_plots import (
    plot_confusion_matrix,
    plot_domain_alignment,
    plot_metrics,
)
from presentation.qt.report import REPORT_IMAGE_WIDTH
from presentation.qt.style import DARK_PALETTE


class AnalyticsChartsWidget(QWidget):
    """Widget embedding Matplotlib charts for visual experiment analytics.

    Chart colours mirror the app's active theme (dark or light) so the
    embedded Matplotlib figure reads as part of the dashboard instead of a
    pasted-in box. Call set_palette after a theme switch.
    """

    def __init__(self, palette: dict[str, str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._last_result: Optional[TransferLearningResult] = None
        self._build_ui()
        self._apply_rcparams()

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
        self._figure = Figure(figsize=(9, 6), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)

        layout.addLayout(control_bar)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._toolbar)

        self._show_placeholder()

    def _apply_rcparams(self) -> None:
        """Push the active palette into Matplotlib and the surrounding widgets."""
        palette = self._palette
        fig_bg = palette["surface"]
        axes_bg = palette["surface_alt"]
        text = palette["text_secondary"]
        grid = palette["border"]

        matplotlib.rcParams.update(
            {
                "figure.facecolor": fig_bg,
                "savefig.facecolor": fig_bg,
                "axes.facecolor": axes_bg,
                "axes.edgecolor": grid,
                "axes.labelcolor": text,
                "text.color": text,
                "xtick.color": text,
                "ytick.color": text,
                "grid.color": grid,
                "legend.facecolor": axes_bg,
                "legend.edgecolor": grid,
                "legend.labelcolor": text,
            }
        )
        self._figure.set_facecolor(fig_bg)
        self._canvas.setStyleSheet(f"background-color: {fig_bg}; border-radius: 10px;")
        self._toolbar.setStyleSheet(
            f"background-color: {palette['surface_alt']}; border: 1px solid {palette['border']};"
            " border-radius: 7px; padding: 2px;"
        )

    def set_palette(self, palette: dict[str, str]) -> None:
        """Re-theme the embedded figure after a dark/light mode switch."""
        self._palette = palette
        self._apply_rcparams()
        if self._last_result is not None:
            self._render_charts()
        else:
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
            color=self._palette.get("text_muted", "#8a97a5"),
            transform=ax.transAxes,
        )
        self._canvas.draw()

    def update_result(self, result: TransferLearningResult) -> None:
        self._last_result = result
        self._render_charts()

    def clear(self) -> None:
        """Drop the rendered run and go back to the placeholder."""
        self._last_result = None
        self._view_combo.blockSignals(True)
        self._view_combo.setCurrentIndex(0)
        self._view_combo.blockSignals(False)
        self._show_placeholder()

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
            plot_domain_alignment(ax, result, self._palette)
        elif view_mode == "Confusion Matrix":
            ax = self._figure.add_subplot(111)
            plot_confusion_matrix(self._figure, ax, result)
        elif view_mode == "Performance Metrics":
            ax = self._figure.add_subplot(111)
            plot_metrics(ax, result, self._palette)

        self._figure.tight_layout()
        self._canvas.draw()

    def _render_overview(self, result: TransferLearningResult) -> None:
        has_metrics = result.metrics is not None and result.target_predictions is not None
        if has_metrics:
            ax1 = self._figure.add_subplot(2, 2, 1)
            plot_domain_alignment(ax1, result, self._palette)

            ax2 = self._figure.add_subplot(2, 2, 2)
            plot_confusion_matrix(self._figure, ax2, result)

            ax3 = self._figure.add_subplot(2, 2, (3, 4))
            plot_metrics(ax3, result, self._palette)
        else:
            ax = self._figure.add_subplot(111)
            plot_domain_alignment(ax, result, self._palette)

    def render_to_base64_html(self) -> str:
        """Render the current figure to a base64 encoded PNG for HTML/PDF reports.

        See REPORT_IMAGE_WIDTH for why this is a width attribute
        rather than a CSS style.
        """
        if self._last_result is None:
            return ""

        buf = io.BytesIO()
        self._figure.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        return (
            f"<img src='data:image/png;base64,{encoded}' "
            f"width='{REPORT_IMAGE_WIDTH}' style='margin-top:10px;' />"
        )

