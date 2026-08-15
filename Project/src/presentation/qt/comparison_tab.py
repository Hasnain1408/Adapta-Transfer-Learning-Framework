"""Cross-experiment comparison built from stored results.

Where the Feature Space tab explains *one* run, this compares many: for a
chosen source→target pair it reads every scored experiment back out of the
database and puts the methods side by side on performance, ranking, domain
alignment, execution cost, and transfer benefit.

Everything here is read from ``evaluations`` rows. Nothing is recomputed,
estimated, or synthesised — a metric that was never recorded (rows written
before a column existed, an AUC the classifier could not produce) is drawn
as a gap and named in the caption, never as a zero. A zero-height bar and a
missing bar mean very different things to someone reading a results chart.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
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

from application.experiment_catalog_use_case import ExperimentCatalogUseCase
from domain.entities import ExperimentHistoryEntry, User
from presentation.qt import message_box
from presentation.qt.duration import format_duration
from presentation.qt.report import build_comparison_report_html
from presentation.qt.style import build_style_sheet
from presentation.qt.title_bar import install_title_bar, install_title_bar_on_grid
from presentation.qt.style import DARK_PALETTE

#: ``(label, EvaluationRecord attribute)`` for every rankable score.
METRICS: list[tuple[str, str]] = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1_score"),
    ("AUC", "auc"),
]

VIEW_PERFORMANCE = "Performance Comparison"
VIEW_RANKING = "Method Ranking"
VIEW_ALIGNMENT = "Domain Alignment (MMD)"
VIEW_EXECUTION = "Execution Time"
VIEW_NEGATIVE = "Baseline vs. Transfer"
VIEW_PAIRS = "Source–Target Pairs"

VIEWS = [
    VIEW_PERFORMANCE,
    VIEW_RANKING,
    VIEW_ALIGNMENT,
    VIEW_EXECUTION,
    VIEW_NEGATIVE,
    VIEW_PAIRS,
]

SORT_METRIC_DESC = "Metric (high → low)"
SORT_METRIC_ASC = "Metric (low → high)"
SORT_NAME = "Method name"
SORT_TIME = "Execution time (fast → slow)"

SORT_OPTIONS = [SORT_METRIC_DESC, SORT_METRIC_ASC, SORT_NAME, SORT_TIME]

#: Sentinel metric key: rank by mean rank across all recorded metrics
#: instead of by any single one. Lower is better, unlike every real metric.
METRIC_OVERALL = "__overall__"
METRIC_OVERALL_LABEL = "Overall (avg. rank)"

#: Label used for the no-adaptation reference in charts and exports.
BASELINE_LABEL = "Baseline"

TABLE_COLUMNS = [
    "Rank",
    "Method",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "AUC",
    "MMD Before",
    "MMD After",
    "MMD Change",
    "Alignment",
    "Exec Time",
    "Baseline Acc",
    "Δ Accuracy",
    "Negative?",
    "Avg. Rank",
]


@dataclass(frozen=True)
class MethodRow:
    """One method's stored numbers for the selected pair.

    Every score is optional: this is a view onto whatever the database
    actually holds, and callers must render ``None`` as a gap.
    """

    method: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc: Optional[float]
    mmd_before: Optional[float]
    mmd_after: Optional[float]
    mmd_change: Optional[float]
    alignment_status: Optional[str]
    execution_seconds: Optional[float]
    baseline_accuracy: Optional[float]
    baseline_f1: Optional[float]
    accuracy_delta: Optional[float]
    experiment_id: Optional[int]

    @property
    def is_negative_transfer(self) -> bool:
        return self.accuracy_delta is not None and self.accuracy_delta < 0

    def value(self, attribute: str) -> Optional[float]:
        return getattr(self, attribute, None)


def rows_from_entries(entries: list[ExperimentHistoryEntry]) -> list[MethodRow]:
    """Flatten history entries into plot-ready rows, dropping unscored runs."""
    rows: list[MethodRow] = []
    for entry in entries:
        evaluation = entry.evaluation
        if evaluation is None:
            continue
        rows.append(
            MethodRow(
                method=entry.experiment.tl_algorithm,
                accuracy=evaluation.accuracy,
                precision=evaluation.precision,
                recall=evaluation.recall,
                f1_score=evaluation.f1_score,
                auc=evaluation.auc,
                mmd_before=evaluation.mmd_before,
                mmd_after=evaluation.mmd_after,
                mmd_change=evaluation.mmd_change_percent,
                alignment_status=evaluation.alignment_status,
                execution_seconds=evaluation.execution_seconds,
                baseline_accuracy=evaluation.baseline_accuracy,
                baseline_f1=evaluation.baseline_f1,
                accuracy_delta=evaluation.accuracy_delta,
                experiment_id=entry.experiment.id,
            )
        )
    return rows


def _fmt(value: Optional[float], places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def competition_ranks(
    values: list[Optional[float]], higher_is_better: bool = True
) -> list[Optional[int]]:
    """Standard competition ranking (1, 2, 2, 4) over *values*.

    Ties share the better rank and consume the ones after it, which is what
    "joint second" means in a results table. ``None`` inputs are unranked
    (``None`` out) rather than ranked last: a method that never recorded a
    metric has not lost on it, and sorting it to the bottom would silently
    invent a comparison the data does not support.
    """
    scored = [(index, value) for index, value in enumerate(values) if value is not None]
    scored.sort(key=lambda pair: pair[1], reverse=higher_is_better)

    ranks: list[Optional[int]] = [None] * len(values)
    previous_value: Optional[float] = None
    previous_rank = 0
    for position, (index, value) in enumerate(scored, start=1):
        if previous_value is not None and value == previous_value:
            ranks[index] = previous_rank
        else:
            ranks[index] = position
            previous_rank = position
            previous_value = value
    return ranks


def average_ranks(rows: list["MethodRow"]) -> dict[str, Optional[float]]:
    """Mean competition rank per method across every recorded metric.

    This is the aggregate the ranking view calls "Overall": a method that
    wins on accuracy but loses on F1 should not be declared best outright,
    and averaging its per-metric ranks is the usual way results tables
    settle that. Each method is averaged only over the metrics it actually
    has, so a missing AUC costs nothing rather than counting as a loss.
    """
    if len(rows) == 1:
        row = rows[0]
        scored = any(row.value(attribute) is not None for _label, attribute in METRICS)
        return {row.method: 1.0 if scored else None}

    per_metric: dict[str, list[Optional[int]]] = {}
    for _label, attribute in METRICS:
        values = [row.value(attribute) for row in rows]
        # A metric only one method recorded is not a contest — ranking it
        # first there would hand it a win nothing else could compete for and
        # drag its average down for free.
        if sum(value is not None for value in values) < 2:
            continue
        per_metric[attribute] = competition_ranks(values)

    overall: dict[str, Optional[float]] = {}
    for index, row in enumerate(rows):
        collected = [
            ranks[index] for ranks in per_metric.values() if ranks[index] is not None
        ]
        overall[row.method] = (sum(collected) / len(collected)) if collected else None
    return overall


class ComparisonExportDialog(QDialog):
    """Pick which comparison views (and the table) go into the PDF.

    Separate from ``ExportOptionsDialog``: that one gates the sections of a
    *single run's* report, while these are the comparison views, which vary
    with what the database actually holds.
    """

    def __init__(self, stylesheet: str, available: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(stylesheet)
        self.setMinimumWidth(380)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        intro = QLabel("Choose which comparisons to include in the exported report.")
        intro.setObjectName("captionLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._boxes: dict[str, QCheckBox] = {}
        for view in VIEWS:
            box = QCheckBox(view)
            box.setChecked(view in available)
            if view not in available:
                box.setEnabled(False)
                box.setToolTip("No stored data supports this view for the current pair.")
            layout.addWidget(box)
            self._boxes[view] = box

        self._table_box = QCheckBox("Exact-values table")
        self._table_box.setChecked(True)
        layout.addWidget(self._table_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export…")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addSpacing(4)
        layout.addWidget(buttons)

        install_title_bar(self, "Export Comparison Report", body)

    def _validate_and_accept(self) -> None:
        if not self.selected_views() and not self.include_table():
            message_box.warning(self, "Nothing Selected", "Select at least one comparison to export.")
            return
        self.accept()

    def selected_views(self) -> list[str]:
        return [view for view, box in self._boxes.items() if box.isChecked()]

    def include_table(self) -> bool:
        return self._table_box.isChecked()


class ComparisonTab(QWidget):
    """Stored-results comparison across methods and domain pairs."""

    def __init__(
        self,
        catalog: ExperimentCatalogUseCase | None,
        palette: dict[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._catalog = catalog
        self._palette = palette or DARK_PALETTE
        self._current_user: User | None = None
        self._rows: list[MethodRow] = []
        #: method name -> mean rank across metrics, recomputed with _rows.
        self._overall: dict[str, Optional[float]] = {}
        self._pairs: list[tuple[str, str]] = []
        self._loading = False
        self._annotation = None
        #: axes -> (bar artist, exact-value label) for hover tooltips.
        self._hover_bars: list[tuple[object, object, str]] = []
        self._build_ui()
        self._apply_rcparams()
        self.refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)
        outer.addLayout(self._build_controls())

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._build_chart_panel())
        splitter.addWidget(self._build_table_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 150])
        outer.addWidget(splitter, 1)

        self._caption = QLabel()
        self._caption.setObjectName("captionLabel")
        self._caption.setWordWrap(True)
        outer.addWidget(self._caption)

    def _build_controls(self) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(6)

        first = QHBoxLayout()
        first.setSpacing(8)

        self._source_combo = QComboBox()
        self._source_combo.setToolTip("Source domain of the stored experiments")
        self._source_combo.currentTextChanged.connect(self._on_source_changed)

        self._target_combo = QComboBox()
        self._target_combo.setToolTip("Target domain of the stored experiments")
        self._target_combo.currentTextChanged.connect(self._reload_rows)

        self._view_combo = QComboBox()
        self._view_combo.addItems(VIEWS)
        self._view_combo.setToolTip("Which comparison to draw")
        self._view_combo.currentTextChanged.connect(self._render)

        self._metric_combo = QComboBox()
        for label, attribute in METRICS:
            self._metric_combo.addItem(label, attribute)
        self._metric_combo.addItem(METRIC_OVERALL_LABEL, METRIC_OVERALL)
        self._metric_combo.setToolTip("Metric used for ranking and sorting")
        self._metric_combo.currentTextChanged.connect(self._render)

        self._sort_combo = QComboBox()
        self._sort_combo.addItems(SORT_OPTIONS)
        self._sort_combo.setToolTip("Order of methods in the chart and table")
        self._sort_combo.currentTextChanged.connect(self._render)

        for label_text, widget in (
            ("Source:", self._source_combo),
            ("Target:", self._target_combo),
            ("View:", self._view_combo),
            ("Metric:", self._metric_combo),
            ("Sort:", self._sort_combo),
        ):
            label = QLabel(label_text)
            label.setObjectName("fieldLabel")
            first.addWidget(label)
            first.addWidget(widget)
        first.addStretch(1)

        second = QHBoxLayout()
        second.setSpacing(8)

        methods_label = QLabel("Methods:")
        methods_label.setObjectName("fieldLabel")
        self._method_list = QListWidget()
        self._method_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._method_list.setFlow(QListWidget.LeftToRight)
        self._method_list.setWrapping(True)
        self._method_list.setMaximumHeight(46)
        self._method_list.setToolTip("Untick a method to leave it out of the comparison")
        self._method_list.itemChanged.connect(self._on_method_toggled)

        self._baseline_check = QCheckBox("Include baseline")
        self._baseline_check.setChecked(True)
        self._baseline_check.setToolTip(
            "Show the no-adaptation reference (the same classifier on the "
            "un-adapted features) alongside the methods"
        )
        self._baseline_check.toggled.connect(self._render)

        self._refresh_button = QPushButton("⟳  Refresh")
        self._refresh_button.setCursor(Qt.PointingHandCursor)
        self._refresh_button.setToolTip("Re-read stored results from the database")
        self._refresh_button.clicked.connect(self.refresh)

        self._export_csv_button = QPushButton("Export CSV…")
        self._export_csv_button.setCursor(Qt.PointingHandCursor)
        self._export_csv_button.setToolTip("Save the comparison table as CSV")
        self._export_csv_button.clicked.connect(self._export_csv)

        second.addWidget(methods_label)
        second.addWidget(self._method_list, 1)
        second.addWidget(self._baseline_check)
        self._export_pdf_button = QPushButton("Export PDF…")
        self._export_pdf_button.setCursor(Qt.PointingHandCursor)
        self._export_pdf_button.setToolTip("Save selected comparison charts and the table as a PDF")
        self._export_pdf_button.clicked.connect(self._export_pdf)

        second.addWidget(self._export_csv_button)
        second.addWidget(self._export_pdf_button)
        second.addWidget(self._refresh_button)

        box.addLayout(first)
        box.addLayout(second)
        return box

    def _build_chart_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._figure = Figure(figsize=(9, 4.6), dpi=100)
        try:
            self._figure.set_layout_engine("constrained")
        except AttributeError:  # matplotlib < 3.6
            self._figure.set_constrained_layout(True)
        self._canvas = FigureCanvas(self._figure)
        # Deliberately low. This tab stacks two control rows, a chart, a
        # table and a caption into the results pane; if the chart panel's
        # minimum exceeds its splitter slot the canvas is clipped by the
        # parent (cutting the x-axis labels) rather than being re-laid out.
        # Constrained layout shrinks the plot area to fit labels at any
        # height, so a low floor degrades gracefully. Use Full Screen for
        # serious comparison work.
        self._canvas.setMinimumHeight(180)
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._toolbar = NavigationToolbar(self._canvas, self)

        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._toolbar)
        return panel

    def _build_table_panel(self) -> QWidget:
        self._table = QTableWidget()
        self._table.setColumnCount(len(TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(26)
        self._table.horizontalHeader().setHighlightSections(False)
        header = self._table.horizontalHeader()
        self._table.setMinimumHeight(56)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for column in range(1, len(TABLE_COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.Stretch)
        return self._table

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------
    def _apply_rcparams(self) -> None:
        palette = self._palette
        matplotlib.rcParams.update(
            {
                "figure.facecolor": palette["surface"],
                "savefig.facecolor": palette["surface"],
                "axes.facecolor": palette["surface_alt"],
                "axes.edgecolor": palette["border"],
                "axes.labelcolor": palette["text_secondary"],
                "text.color": palette["text_secondary"],
                "xtick.color": palette["text_secondary"],
                "ytick.color": palette["text_secondary"],
                "grid.color": palette["border"],
                "legend.facecolor": palette["surface_alt"],
                "legend.edgecolor": palette["border"],
                "legend.labelcolor": palette["text_secondary"],
            }
        )
        self._figure.set_facecolor(palette["surface"])
        self._canvas.setStyleSheet(f"background-color: {palette['surface']}; border-radius: 10px;")
        self._toolbar.setStyleSheet(
            f"background-color: {palette['surface_alt']}; border: 1px solid {palette['border']};"
            " border-radius: 7px; padding: 2px;"
        )

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        self._apply_rcparams()
        self._render()

    def set_catalog(self, catalog: ExperimentCatalogUseCase | None) -> None:
        self._catalog = catalog
        self.refresh()

    def set_current_user(self, user: User | None) -> None:
        self._current_user = user
        self.refresh()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _user_id(self) -> int | None:
        return self._current_user.id if self._current_user else None

    def refresh(self) -> None:
        """Re-read the stored domain pairs, keeping the current selection."""
        if self._catalog is None:
            self._pairs = []
            self._rows = []
            self._overall = {}
            self._reload_method_list()
            self._render()
            return

        previous = (self._source_combo.currentText(), self._target_combo.currentText())
        try:
            self._pairs = self._catalog.domain_pairs(self._user_id())
        except Exception:  # noqa: BLE001 - surfaced in the caption by _render
            self._pairs = []

        self._loading = True
        self._source_combo.clear()
        self._source_combo.addItems(sorted({source for source, _ in self._pairs}))
        if previous[0]:
            index = self._source_combo.findText(previous[0])
            if index >= 0:
                self._source_combo.setCurrentIndex(index)
        self._loading = False

        self._sync_targets(prefer=previous[1])
        self._reload_rows()

    def _on_source_changed(self, _text: str) -> None:
        if self._loading:
            return
        self._sync_targets()
        self._reload_rows()

    def _sync_targets(self, prefer: str = "") -> None:
        """Offer only the targets that actually pair with the chosen source."""
        source = self._source_combo.currentText()
        targets = [target for known_source, target in self._pairs if known_source == source]
        self._loading = True
        self._target_combo.clear()
        self._target_combo.addItems(targets)
        if prefer:
            index = self._target_combo.findText(prefer)
            if index >= 0:
                self._target_combo.setCurrentIndex(index)
        self._loading = False

    def _reload_rows(self, *_args) -> None:
        if self._loading:
            return
        source = self._source_combo.currentText()
        target = self._target_combo.currentText()
        if self._catalog is None or not source or not target:
            self._rows = []
        else:
            try:
                self._rows = rows_from_entries(
                    self._catalog.results_for_pair(source, target, self._user_id())
                )
            except Exception:  # noqa: BLE001 - reported in the caption
                self._rows = []
        self._overall = average_ranks(self._rows)
        self._reload_method_list()
        self._render()

    def _reload_method_list(self) -> None:
        """Rebuild the method checkboxes, preserving anything already unticked."""
        unchecked = {
            self._method_list.item(index).text()
            for index in range(self._method_list.count())
            if self._method_list.item(index).checkState() == Qt.Unchecked
        }
        self._loading = True
        self._method_list.clear()
        for row in self._rows:
            item = QListWidgetItem(row.method)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if row.method in unchecked else Qt.Checked)
            self._method_list.addItem(item)
        self._loading = False

    def _on_method_toggled(self, _item: QListWidgetItem) -> None:
        if not self._loading:
            self._render()

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _selected_methods(self) -> set[str]:
        return {
            self._method_list.item(index).text()
            for index in range(self._method_list.count())
            if self._method_list.item(index).checkState() == Qt.Checked
        }

    def _metric_attribute(self) -> str:
        return self._metric_combo.currentData() or "accuracy"

    def _metric_label(self) -> str:
        return self._metric_combo.currentText()

    def visible_rows(self) -> list[MethodRow]:
        """Ticked methods, in the order the sort control asks for."""
        selected = self._selected_methods()
        rows = [row for row in self._rows if row.method in selected]
        attribute = self._metric_attribute()
        order = self._sort_combo.currentText()

        if order == SORT_NAME:
            rows.sort(key=lambda row: row.method)
        elif order == SORT_TIME:
            # Rows with no stored time go last rather than sorting as 0.
            rows.sort(
                key=lambda row: (
                    row.execution_seconds is None,
                    row.execution_seconds or 0.0,
                )
            )
        else:
            best_first = order != SORT_METRIC_ASC
            if attribute == METRIC_OVERALL:
                # Average rank inverts the usual direction: 1.0 is the best
                # a method can score, so "best first" means ascending here.
                rows.sort(
                    key=lambda row: (
                        self._overall.get(row.method) is None,
                        (self._overall.get(row.method) or 0.0) * (1 if best_first else -1),
                    )
                )
            else:
                rows.sort(
                    key=lambda row: (row.value(attribute) is None, -(row.value(attribute) or 0.0))
                    if best_first
                    else (row.value(attribute) is None, row.value(attribute) or 0.0)
                )
        return rows

    def _metric_value(self, row: MethodRow) -> Optional[float]:
        """The selected metric for *row*, or its average rank for Overall."""
        attribute = self._metric_attribute()
        if attribute == METRIC_OVERALL:
            return self._overall.get(row.method)
        return row.value(attribute)

    def ranked_rows(self) -> list[tuple[Optional[int], MethodRow]]:
        """Visible rows paired with their rank on the selected metric.

        The rank is computed over the *visible* set, so unticking a method
        renumbers the rest — the ranking answers "best of what I am looking
        at", which is what the controls imply.
        """
        rows = self.visible_rows()
        values = [self._metric_value(row) for row in rows]
        higher_is_better = self._metric_attribute() != METRIC_OVERALL
        return list(zip(competition_ranks(values, higher_is_better), rows))

    def _baseline_value(self, rows: list[MethodRow], attribute: str) -> Optional[float]:
        """The no-adaptation reference for *attribute*, if one was stored.

        Only accuracy and F1 have a stored baseline; the runs on a pair share
        preprocessing and classifier, so their baselines agree and the first
        recorded one represents them all.
        """
        field = {"accuracy": "baseline_accuracy", "f1_score": "baseline_f1"}.get(attribute)
        if field is None:
            return None
        for row in rows:
            value = getattr(row, field)
            if value is not None:
                return value
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _placeholder(self, message: str) -> None:
        self._figure.clear()
        self._hover_bars = []
        # figure.clear() destroys whatever axes the annotation (if any) was
        # attached to, so the artist is already gone -- keeping the stale
        # reference makes the next hover event call .remove() on a dead
        # artist, which raises NotImplementedError instead of doing nothing.
        self._annotation = None
        axes = self._figure.add_subplot(111)
        axes.axis("off")
        axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=11,
            wrap=True,
            color=self._palette.get("text_muted", "#8a97a5"),
            transform=axes.transAxes,
        )
        self._canvas.draw()

    def _render(self, *_args) -> None:
        self._populate_table()
        if self._catalog is None:
            self._placeholder("No database connection — stored results are unavailable.")
            self._caption.setText("Connect a database to compare stored experiment results.")
            return
        if not self._pairs:
            self._placeholder(
                "No stored results yet.\nRun and save an experiment, then refresh."
            )
            self._caption.setText("")
            return

        rows = self.visible_rows()
        if not rows:
            self._placeholder("No methods selected for this source–target pair.")
            self._caption.setText("")
            return

        self._figure.clear()
        self._hover_bars = []
        # See _placeholder: the old annotation's axes no longer exists.
        self._annotation = None
        view = self._view_combo.currentText()
        axes = self._figure.add_subplot(111)

        if view == VIEW_PERFORMANCE:
            self._draw_performance(axes, rows)
        elif view == VIEW_RANKING:
            self._draw_ranking(axes, rows)
        elif view == VIEW_ALIGNMENT:
            self._draw_alignment(axes, rows)
        elif view == VIEW_EXECUTION:
            self._draw_execution(axes, rows)
        elif view == VIEW_NEGATIVE:
            self._draw_negative_transfer(axes, rows)
        else:
            self._draw_pairs(axes)

        self._canvas.draw()
        self._update_caption(rows)

    def _bar_colors(self, count: int) -> list:
        keys = ["metric_teal", "metric_purple", "metric_amber", "metric_pink", "accent"]
        return [self._palette.get(keys[i % len(keys)], "#1e6fd9") for i in range(count)]

    def _register_bars(self, axes, bars, labels: list[str]) -> None:
        for bar, label in zip(bars, labels):
            self._hover_bars.append((axes, bar, label))

    # -- individual views ----------------------------------------------
    def _draw_performance(self, axes, rows: list[MethodRow]) -> None:
        """Grouped bars: one cluster per method, one bar per metric."""
        names = [row.method for row in rows]
        include_baseline = self._baseline_check.isChecked()
        baseline_accuracy = self._baseline_value(rows, "accuracy")
        baseline_f1 = self._baseline_value(rows, "f1_score")
        if include_baseline and (baseline_accuracy is not None or baseline_f1 is not None):
            names = names + [BASELINE_LABEL]

        positions = np.arange(len(names))
        width = 0.8 / len(METRICS)
        colors = self._bar_colors(len(METRICS))

        for index, (label, attribute) in enumerate(METRICS):
            values, offsets, texts = [], [], []
            for slot, name in enumerate(names):
                if name == BASELINE_LABEL:
                    # Only accuracy and F1 have a recorded no-adaptation value.
                    value = {"accuracy": baseline_accuracy, "f1_score": baseline_f1}.get(attribute)
                else:
                    value = rows[slot].value(attribute)
                if value is None:
                    continue
                values.append(value)
                offsets.append(positions[slot] + index * width - 0.4 + width / 2)
                texts.append(f"{name} · {label} = {value:.4f}")
            if not values:
                continue
            bars = axes.bar(offsets, values, width=width, color=colors[index], label=label)
            self._register_bars(axes, bars, texts)

        axes.set_xticks(positions)
        axes.set_xticklabels(names, rotation=0, fontsize=8)
        axes.set_ylabel("Score", fontsize=9)
        axes.set_ylim(0, 1.05)
        axes.set_title(self._chart_title("Performance by method"), fontsize=10, fontweight="bold")
        axes.grid(axis="y", linestyle="--", alpha=0.3)
        axes.legend(fontsize=7, ncol=len(METRICS), loc="upper right", framealpha=0.85)

    def _draw_ranking(self, axes, rows: list[MethodRow]) -> None:
        """Horizontal bars, best at the top, for the selected metric."""
        attribute = self._metric_attribute()
        label = self._metric_label()
        overall = attribute == METRIC_OVERALL

        ranked = [(rank, row) for rank, row in self.ranked_rows() if rank is not None]
        if not ranked:
            self._placeholder(f"No stored {label} values for this pair.")
            return

        # Best at the top: barh draws upwards, so plot worst-first.
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        names = [row.method for _rank, row in ranked]
        values = [self._metric_value(row) or 0.0 for _rank, row in ranked]
        positions = np.arange(len(ranked))
        places = 2 if overall else 4

        # The winner is coloured apart so the answer is readable at a glance.
        best = min(rank for rank, _row in ranked)
        colors = [
            self._palette.get("success", "#14804a") if rank == best
            else self._palette.get("metric_teal", "#0f9b8e")
            for rank, _row in ranked
        ]
        bars = axes.barh(positions, values, color=colors)
        self._register_bars(
            axes,
            bars,
            [f"#{rank}  {row.method} · {label} = {self._metric_value(row):.{places}f}"
             for rank, row in ranked],
        )

        span = max(values) or 1.0
        for position, value, (rank, _row) in zip(positions, values, ranked):
            axes.text(
                value + span * 0.015, position,
                f"{value:.{places}f}   #{rank}",
                va="center", fontsize=8,
            )

        if self._baseline_check.isChecked() and not overall:
            # An average rank has no baseline to compare against — the
            # baseline never competed in the ranking.
            baseline = self._baseline_value(rows, attribute)
            if baseline is not None:
                axes.axvline(
                    baseline,
                    color=self._palette.get("danger", "#b3261e"),
                    linestyle="--",
                    linewidth=1.4,
                    label=f"{BASELINE_LABEL} {baseline:.4f}",
                )
                axes.legend(fontsize=7, loc="lower right", framealpha=0.85)

        axes.set_yticks(positions)
        axes.set_yticklabels(names, fontsize=8)
        axes.set_xlim(0, span * 1.28 if overall else 1.18)
        axes.set_xlabel(
            "Average rank across metrics (lower is better)" if overall else label, fontsize=9
        )
        axes.set_title(
            self._chart_title(
                "Methods ranked overall" if overall else f"Methods ranked by {label}"
            ),
            fontsize=10,
            fontweight="bold",
        )
        axes.grid(axis="x", linestyle="--", alpha=0.3)

    def _draw_alignment(self, axes, rows: list[MethodRow]) -> None:
        """MMD before vs. after, per method, with the change annotated."""
        scored = [row for row in rows if row.mmd_before is not None and row.mmd_after is not None]
        if not scored:
            self._placeholder(
                "No stored MMD values for this pair.\n"
                "Re-run these experiments to record domain alignment."
            )
            return

        positions = np.arange(len(scored))
        width = 0.38
        before = [row.mmd_before or 0.0 for row in scored]
        after = [row.mmd_after or 0.0 for row in scored]

        bars_before = axes.bar(
            positions - width / 2, before, width,
            color=self._palette.get("metric_amber", "#c77700"), label="MMD before",
        )
        bars_after = axes.bar(
            positions + width / 2, after, width,
            color=self._palette.get("metric_teal", "#0f9b8e"), label="MMD after",
        )
        self._register_bars(axes, bars_before,
            [f"{r.method} · MMD before = {r.mmd_before:.6f}" for r in scored])
        self._register_bars(axes, bars_after,
            [f"{r.method} · MMD after = {r.mmd_after:.6f}  ({r.alignment_status or '—'})"
             for r in scored])

        top = max(max(before), max(after)) or 1.0
        for position, row in zip(positions, scored):
            if row.mmd_change is None:
                continue
            improved = row.mmd_change >= 0
            axes.text(
                position,
                max(row.mmd_before or 0, row.mmd_after or 0) + top * 0.03,
                f"{row.mmd_change:+.1f}%",
                ha="center",
                fontsize=7,
                fontweight="bold",
                color=self._palette["success"] if improved else self._palette["danger"],
            )

        axes.set_xticks(positions)
        axes.set_xticklabels([row.method for row in scored], rotation=0, fontsize=8)
        axes.set_ylabel("MMD", fontsize=9)
        axes.set_ylim(0, top * 1.2)
        axes.set_title(
            self._chart_title("Domain alignment before vs. after adaptation"),
            fontsize=10, fontweight="bold",
        )
        axes.grid(axis="y", linestyle="--", alpha=0.3)
        axes.legend(fontsize=7, framealpha=0.85)

    def _draw_execution(self, axes, rows: list[MethodRow]) -> None:
        scored = [row for row in rows if row.execution_seconds is not None]
        if not scored:
            self._placeholder(
                "No stored execution times for this pair.\n"
                "Re-run these experiments to record timing."
            )
            return

        scored.sort(key=lambda row: row.execution_seconds or 0.0)
        positions = np.arange(len(scored))
        values = [row.execution_seconds or 0.0 for row in scored]
        bars = axes.bar(positions, values, color=self._palette.get("metric_purple", "#7b5bd6"))
        self._register_bars(
            axes, bars, [f"{r.method} · {format_duration(r.execution_seconds)}" for r in scored]
        )
        for position, value in zip(positions, values):
            axes.text(position, value, f" {format_duration(value)}", ha="center",
                      va="bottom", fontsize=7, rotation=90)

        axes.set_xticks(positions)
        axes.set_xticklabels([row.method for row in scored], rotation=0, fontsize=8)
        axes.set_ylabel("Seconds", fontsize=9)
        axes.set_ylim(0, max(values) * 1.28 if max(values) else 1.0)
        axes.set_title(self._chart_title("Execution time by method"), fontsize=10, fontweight="bold")
        axes.grid(axis="y", linestyle="--", alpha=0.3)

    def _draw_negative_transfer(self, axes, rows: list[MethodRow]) -> None:
        """Baseline vs. adapted accuracy, flagging the methods that lost."""
        scored = [row for row in rows if row.baseline_accuracy is not None]
        if not scored:
            self._placeholder(
                "No stored baseline scores for this pair.\n"
                "Re-run these experiments to record the no-adaptation baseline."
            )
            return

        positions = np.arange(len(scored))
        width = 0.38
        baseline = [row.baseline_accuracy or 0.0 for row in scored]
        adapted = [row.accuracy or 0.0 for row in scored]

        bars_base = axes.bar(
            positions - width / 2, baseline, width,
            color=self._palette.get("text_muted", "#8a97a5"), label="Without adaptation",
        )
        # Losing methods are coloured as the warning they are, so negative
        # transfer is visible without reading the delta labels.
        colors = [
            self._palette["danger"] if row.is_negative_transfer else self._palette["success"]
            for row in scored
        ]
        bars_adapted = axes.bar(
            positions + width / 2, adapted, width, color=colors, label="After adaptation"
        )
        self._register_bars(axes, bars_base,
            [f"{r.method} · baseline accuracy = {r.baseline_accuracy:.4f}" for r in scored])
        self._register_bars(axes, bars_adapted,
            [f"{r.method} · adapted = {r.accuracy:.4f} ({r.accuracy_delta:+.4f})"
             + ("  ⚠ negative transfer" if r.is_negative_transfer else "") for r in scored])

        for position, row in zip(positions, scored):
            if row.accuracy_delta is None:
                continue
            axes.text(
                position,
                max(row.baseline_accuracy or 0, row.accuracy or 0) + 0.02,
                f"{row.accuracy_delta:+.3f}",
                ha="center", fontsize=7, fontweight="bold",
                color=self._palette["danger"] if row.is_negative_transfer
                else self._palette["success"],
            )

        axes.set_xticks(positions)
        axes.set_xticklabels([row.method for row in scored], rotation=0, fontsize=8)
        axes.set_ylabel("Target accuracy", fontsize=9)
        axes.set_ylim(0, 1.12)
        axes.set_title(
            self._chart_title("Baseline vs. transfer accuracy"), fontsize=10, fontweight="bold"
        )
        axes.grid(axis="y", linestyle="--", alpha=0.3)
        axes.legend(fontsize=7, framealpha=0.85)

    def _draw_pairs(self, axes) -> None:
        """The selected metric for one method across every stored pair."""
        attribute = self._metric_attribute()
        label = self._metric_label()
        selected = self._selected_methods()
        if self._catalog is None:
            self._placeholder("No database connection.")
            return

        pair_labels: list[str] = []
        per_method: dict[str, list[Optional[float]]] = {name: [] for name in sorted(selected)}
        for source, target in self._pairs:
            try:
                entries = self._catalog.results_for_pair(source, target, self._user_id())
            except Exception:  # noqa: BLE001
                continue
            by_method = {row.method: row for row in rows_from_entries(entries)}
            if not any(name in by_method for name in per_method):
                continue
            pair_labels.append(f"{source}→{target}")
            for name in per_method:
                row = by_method.get(name)
                per_method[name].append(row.value(attribute) if row else None)

        if not pair_labels:
            self._placeholder("No other stored pairs use the selected methods.")
            return

        positions = np.arange(len(pair_labels))
        active = {name: values for name, values in per_method.items() if any(v is not None for v in values)}
        if not active:
            self._placeholder(f"No stored {label} values across pairs.")
            return
        width = 0.8 / len(active)
        colors = self._bar_colors(len(active))

        for index, (name, values) in enumerate(sorted(active.items())):
            offsets, heights, texts = [], [], []
            for slot, value in enumerate(values):
                if value is None:
                    continue
                offsets.append(positions[slot] + index * width - 0.4 + width / 2)
                heights.append(value)
                texts.append(f"{name} · {pair_labels[slot]} · {label} = {value:.4f}")
            if not heights:
                continue
            bars = axes.bar(offsets, heights, width=width, color=colors[index], label=name)
            self._register_bars(axes, bars, texts)

        axes.set_xticks(positions)
        axes.set_xticklabels(pair_labels, rotation=20, ha="right", fontsize=8)
        axes.set_ylabel(label, fontsize=9)
        axes.set_ylim(0, 1.05)
        axes.set_title(f"{label} across source–target pairs", fontsize=10, fontweight="bold")
        axes.grid(axis="y", linestyle="--", alpha=0.3)
        axes.legend(fontsize=7, ncol=min(len(active), 5), framealpha=0.85)

    def _chart_title(self, prefix: str) -> str:
        return f"{prefix}  ({self._source_combo.currentText()} → {self._target_combo.currentText()})"

    # ------------------------------------------------------------------
    # Table, caption, hover
    # ------------------------------------------------------------------
    def _populate_table(self) -> None:
        ranked = self.ranked_rows() if self._rows else []
        self._table.clearContents()
        self._table.setRowCount(len(ranked))
        for index, (rank, row) in enumerate(ranked):
            values = [
                "—" if rank is None else f"#{rank}",
                row.method,
                _fmt(row.accuracy), _fmt(row.precision), _fmt(row.recall),
                _fmt(row.f1_score), _fmt(row.auc),
                _fmt(row.mmd_before, 6), _fmt(row.mmd_after, 6),
                "—" if row.mmd_change is None else f"{row.mmd_change:+.2f}%",
                row.alignment_status or "—",
                format_duration(row.execution_seconds) or "—",
                _fmt(row.baseline_accuracy),
                "—" if row.accuracy_delta is None else f"{row.accuracy_delta:+.4f}",
                "Yes" if row.is_negative_transfer else ("—" if row.accuracy_delta is None else "No"),
                _fmt(self._overall.get(row.method), 2),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if value == "—":
                    item.setToolTip("Not recorded for this experiment")
                if column == len(values) - 1 and row.is_negative_transfer:
                    item.setForeground(QTableWidgetItem().foreground())
                self._table.setItem(index, column, item)

    def _update_caption(self, rows: list[MethodRow]) -> None:
        total = len(self._rows)
        shown = len(rows)
        missing = [
            name
            for name, present in (
                ("AUC", any(r.auc is not None for r in rows)),
                ("MMD", any(r.mmd_before is not None for r in rows)),
                ("execution time", any(r.execution_seconds is not None for r in rows)),
                ("baseline", any(r.baseline_accuracy is not None for r in rows)),
            )
            if not present
        ]
        parts = [f"{shown} of {total} stored method(s) shown, latest run of each"]

        # The ranking is the headline finding, so state it in words rather
        # than leaving the reader to decode the chart.
        winners = [(rank, row) for rank, row in self.ranked_rows() if rank == 1]
        if winners:
            overall = self._metric_attribute() == METRIC_OVERALL
            places = 2 if overall else 4
            names = ", ".join(row.method for _rank, row in winners)
            value = self._metric_value(winners[0][1])
            descriptor = "avg. rank" if overall else self._metric_label()
            parts.append(
                f"best by {descriptor}: {names} ({value:.{places}f})"
                + (" — tied" if len(winners) > 1 else "")
            )

        losers = [row.method for row in rows if row.is_negative_transfer]
        if losers:
            parts.append("potential negative transfer: " + ", ".join(sorted(losers)))
        if missing:
            parts.append(
                "not recorded for these runs: " + ", ".join(missing)
                + " — re-run to capture them"
            )
        self._caption.setText("  ·  ".join(parts) + ".")

    def _on_hover(self, event) -> None:
        if event.inaxes is None:
            self._hide_annotation()
            return
        for axes, bar, label in self._hover_bars:
            if axes is not event.inaxes:
                continue
            if bar.contains(event)[0]:
                self._show_annotation(axes, event, label)
                return
        self._hide_annotation()

    def _show_annotation(self, axes, event, text: str) -> None:
        if self._annotation is not None:
            self._annotation.remove()
        self._annotation = axes.annotate(
            text,
            xy=(event.xdata, event.ydata),
            xytext=(12, 14),
            textcoords="offset points",
            fontsize=8,
            color=self._palette.get("tooltip_text", "#ffffff"),
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=self._palette.get("tooltip_bg", "#1f2933"),
                edgecolor=self._palette.get("border", "#39424e"),
            ),
            zorder=30,
        )
        self._canvas.draw_idle()

    def _hide_annotation(self) -> None:
        if self._annotation is not None:
            self._annotation.remove()
            self._annotation = None
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def table_rows(self) -> list[list[str]]:
        """The comparison table as plain strings — header row first."""
        out = [list(TABLE_COLUMNS)]
        for rank, row in self.ranked_rows():
            out.append(
                [
                    "—" if rank is None else f"#{rank}",
                    row.method,
                    _fmt(row.accuracy), _fmt(row.precision), _fmt(row.recall),
                    _fmt(row.f1_score), _fmt(row.auc),
                    _fmt(row.mmd_before, 6), _fmt(row.mmd_after, 6),
                    "—" if row.mmd_change is None else f"{row.mmd_change:+.2f}%",
                    row.alignment_status or "—",
                    format_duration(row.execution_seconds) or "—",
                    _fmt(row.baseline_accuracy),
                    "—" if row.accuracy_delta is None else f"{row.accuracy_delta:+.4f}",
                    "Yes" if row.is_negative_transfer else ("—" if row.accuracy_delta is None else "No"),
                    _fmt(self._overall.get(row.method), 2),
                ]
            )
        return out

    def context(self) -> dict[str, str]:
        """What the current chart is showing, for the report header."""
        return {
            "Source domain": self._source_combo.currentText() or "—",
            "Target domain": self._target_combo.currentText() or "—",
            "View": self._view_combo.currentText(),
            "Metric": self._metric_label(),
            "Sort": self._sort_combo.currentText(),
            "Baseline included": "Yes" if self._baseline_check.isChecked() else "No",
        }

    def render_to_base64_html(self) -> str:
        """Current chart as an inline PNG for the PDF report."""
        import base64
        import io

        if not self._rows:
            return ""
        buffer = io.BytesIO()
        self._figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("ascii")
        return (
            f"<img src='data:image/png;base64,{encoded}' "
            "style='width:100%; max-width:820px; margin-top:10px;' />"
        )

    def available_views(self) -> list[str]:
        """Views the stored data can actually support for the current pair.

        Drives which checkboxes the export dialog offers — exporting a page
        that just says "no stored MMD values" helps nobody.
        """
        rows = self.visible_rows()
        if not rows:
            return []
        available = [VIEW_PERFORMANCE, VIEW_RANKING, VIEW_PAIRS]
        if any(r.mmd_before is not None and r.mmd_after is not None for r in rows):
            available.append(VIEW_ALIGNMENT)
        if any(r.execution_seconds is not None for r in rows):
            available.append(VIEW_EXECUTION)
        if any(r.baseline_accuracy is not None for r in rows):
            available.append(VIEW_NEGATIVE)
        return [view for view in VIEWS if view in available]

    def _render_views_to_html(self, views: list[str]) -> list[tuple[str, str]]:
        """Draw each requested view once and capture it as inline PNG markup.

        The visible selection is restored afterwards: exporting should not
        leave the user looking at a different chart than they started on.
        """
        original = self._view_combo.currentText()
        captured: list[tuple[str, str]] = []
        try:
            for view in views:
                self._view_combo.blockSignals(True)
                self._view_combo.setCurrentText(view)
                self._view_combo.blockSignals(False)
                self._render()
                captured.append((view, self.render_to_base64_html()))
        finally:
            self._view_combo.blockSignals(True)
            self._view_combo.setCurrentText(original)
            self._view_combo.blockSignals(False)
            self._render()
        return captured

    def _export_pdf(self) -> None:
        if not self.visible_rows():
            message_box.warning(
                self, "Nothing to Export", "There are no stored results to compare."
            )
            return

        chooser = ComparisonExportDialog(
            build_style_sheet(self._palette), self.available_views(), parent=self
        )
        if chooser.exec_() != QDialog.Accepted:
            return
        views = chooser.selected_views()
        include_table = chooser.include_table()

        source = self._source_combo.currentText() or "results"
        target = self._target_combo.currentText() or "comparison"
        dialog = QFileDialog(
            self,
            "Export Comparison Report",
            f"comparison_{source}_to_{target}.pdf",
            "PDF Files (*.pdf)",
        )
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("pdf")
        dialog.setStyleSheet(build_style_sheet(self._palette))
        install_title_bar_on_grid(dialog, "Export Comparison Report")
        if dialog.exec_() != QDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        path = selected[0]
        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        html = build_comparison_report_html(
            self.context(),
            self._render_views_to_html(views),
            self.table_rows() if include_table else [],
            self._current_user.name if self._current_user else "Not signed in",
            include_table=include_table,
        )
        try:
            from PyQt5.QtGui import QTextDocument
            from PyQt5.QtPrintSupport import QPrinter

            document = QTextDocument()
            document.setHtml(html)
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(path)
            printer.setPageMargins(15, 15, 15, 15, QPrinter.Millimeter)
            document.print_(printer)
        except Exception as exc:  # noqa: BLE001 - shown verbatim
            message_box.critical(self, "Export Failed", f"The report could not be written:\n\n{exc}")
            return
        message_box.information(
            self, "Report Exported", f"The comparison report was saved to:\n\n{path}"
        )

    def _export_csv(self) -> None:
        rows = self.table_rows()
        if len(rows) < 2:
            return
        source = self._source_combo.currentText() or "results"
        target = self._target_combo.currentText() or "comparison"
        dialog = QFileDialog(
            self, "Export Comparison CSV", f"comparison_{source}_to_{target}.csv",
            "CSV Files (*.csv)",
        )
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("csv")
        from presentation.qt.style import build_style_sheet
        from presentation.qt.title_bar import install_title_bar_on_grid

        dialog.setStyleSheet(build_style_sheet(self._palette))
        install_title_bar_on_grid(dialog, "Export Comparison CSV")
        from PyQt5.QtWidgets import QDialog

        if dialog.exec_() != QDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        path = selected[0]
        if not path.lower().endswith(".csv"):
            path += ".csv"

        from presentation.qt import message_box

        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([f"# {key}: {value}" for key, value in self.context().items()])
                writer.writerows(rows)
        except OSError as exc:
            message_box.critical(self, "Export Failed", f"Could not write the CSV:\n\n{exc}")
            return
        message_box.information(self, "CSV Exported", f"The comparison table was saved to:\n\n{path}")


__all__ = [
    "ComparisonTab",
    "MethodRow",
    "rows_from_entries",
    "METRICS",
    "VIEWS",
    "VIEW_PERFORMANCE",
    "VIEW_RANKING",
    "VIEW_ALIGNMENT",
    "VIEW_EXECUTION",
    "VIEW_NEGATIVE",
    "VIEW_PAIRS",
    "SORT_OPTIONS",
    "SORT_METRIC_ASC",
    "SORT_METRIC_DESC",
    "SORT_NAME",
    "SORT_TIME",
    "TABLE_COLUMNS",
    "BASELINE_LABEL",
]
