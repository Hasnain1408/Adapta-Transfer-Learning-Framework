"""Full detail for one batch pair: predictions, visualizations, dataset summary.

Opened as a closable tab in ``MainWindow``'s results tab strip when a row is
selected in the Batch Results table — mirrors ``ExperimentDetailView``'s
"select a row, get a detail tab" pattern already used for History.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from domain.models import PairTransferLearningResult, TransferLearningResult
from presentation.qt.charts_widget import AnalyticsChartsWidget
from presentation.qt.dataset_summary_view import DatasetSummaryView
from presentation.qt.metric_card import MetricCard
from presentation.qt.result_tables import PREDICTIONS_HEADERS, populate_predictions_table


def _build_table(headers: list[str]) -> QTableWidget:
    table = QTableWidget()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(28)
    table.horizontalHeader().setHighlightSections(False)
    table.horizontalHeader().setStretchLastSection(True)
    return table


class PairDetailView(QWidget):
    """Read-only page combining one batch pair's predictions/charts/summary."""

    export_requested = pyqtSignal()

    def __init__(
        self,
        pair: PairTransferLearningResult,
        palette: dict[str, str],
        parent: QWidget | None = None,
        title: str | None = None,
        subtitle_prefix: str = "",
    ) -> None:
        """*title* and *subtitle_prefix* override the header for a reuse
        where "source → target" is not the interesting identity — a method
        of a comparison run, where every entry shares the same pair and the
        *method* is what tells them apart."""
        super().__init__(parent)
        self._pair = pair
        self._palette = palette
        self._title = title
        self._subtitle_prefix = subtitle_prefix
        self._charts_widget: AnalyticsChartsWidget | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(14)
        outer.addLayout(self._build_header(has_result=pair.result is not None))

        result = pair.result
        if result is None:
            error_label = QLabel(pair.error or "This pair failed and produced no result.")
            error_label.setWordWrap(True)
            error_label.setStyleSheet(f"color: {palette['danger']};")
            outer.addWidget(error_label)
            outer.addStretch(1)
            return

        if result.metrics is not None:
            outer.addLayout(self._build_metrics_row(result))
            outer.addWidget(self._build_counts_label(result))

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_predictions_page(result), "Predictions")
        tabs.addTab(self._build_visualizations_page(result), "Visualizations")
        tabs.addTab(self._build_summary_page(result), "Dataset Summary")
        outer.addWidget(tabs, 1)

    def chart_html(self) -> str:
        """Rendered chart markup for the report export, or "" if this pair
        failed and never built a charts widget."""
        return self._charts_widget.render_to_base64_html() if self._charts_widget else ""

    # ------------------------------------------------------------------
    def _build_header(self, has_result: bool) -> QHBoxLayout:
        pair = self._pair
        source_name = Path(pair.source_path).name
        target_name = Path(pair.target_path).name

        row = QHBoxLayout()
        row.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel(self._title or f"{source_name} → {target_name}")
        title.setObjectName("appTitle")
        subtitle = QLabel(self._status_text())
        subtitle.setObjectName("appSubtitle")
        subtitle.setWordWrap(True)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        row.addLayout(title_box)
        row.addStretch(1)

        if has_result:
            export_button = QPushButton("Export Report…")
            export_button.setCursor(Qt.PointingHandCursor)
            export_button.clicked.connect(self.export_requested.emit)
            row.addWidget(export_button, 0, Qt.AlignTop)
        return row

    def _status_text(self) -> str:
        pair = self._pair
        if pair.result is None:
            status = "Failed"
        elif pair.result.metrics is None:
            status = "Completed — no target labels to evaluate against"
        else:
            status = "Completed"
        return f"{self._subtitle_prefix}  ·  {status}" if self._subtitle_prefix else status

    def _build_metrics_row(self, result: TransferLearningResult) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        metrics = result.metrics
        specs = [
            ("Accuracy", metrics.accuracy, "teal"),
            ("Precision", metrics.precision, "purple"),
            ("Recall", metrics.recall, "amber"),
            ("F1 Score", metrics.f1_score, "pink"),
        ]
        for title, value, accent in specs:
            card = MetricCard(title, accent=accent)
            card.set_value(f"{value:.4f}")
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(card)
        return row

    def _build_counts_label(self, result: TransferLearningResult) -> QLabel:
        metrics = result.metrics
        label = QLabel(
            f"{metrics.total_samples} target samples evaluated  ·  "
            f"{metrics.correct_predictions} correct  ·  "
            f"{metrics.wrong_predictions} incorrect"
        )
        label.setObjectName("captionLabel")
        return label

    def _build_predictions_page(self, result: TransferLearningResult) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        if result.target_predictions is None:
            label = QLabel("No target labels — predictions cannot be evaluated for this pair.")
            label.setObjectName("captionLabel")
            layout.addWidget(label)
            layout.addStretch(1)
            return page

        incorrect_only = QCheckBox("Show incorrect predictions only")
        layout.addWidget(incorrect_only)

        table = _build_table(PREDICTIONS_HEADERS)
        populate_predictions_table(table, result, self._palette)
        layout.addWidget(table, 1)

        def apply_filter(checked: bool) -> None:
            for row, correctness in enumerate(result.prediction_correctness):
                table.setRowHidden(row, checked and correctness == "Right")

        incorrect_only.toggled.connect(apply_filter)
        return page

    def _build_visualizations_page(self, result: TransferLearningResult) -> QWidget:
        charts = AnalyticsChartsWidget(self._palette)
        charts.update_result(result)
        self._charts_widget = charts
        return charts

    def _build_summary_page(self, result: TransferLearningResult) -> QWidget:
        view = DatasetSummaryView(self._palette)
        view.update_result(result)
        return view


__all__ = ["PairDetailView"]
