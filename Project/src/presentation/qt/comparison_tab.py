"""Cross-experiment comparison built from stored results.

Where the Feature Space tab explains *one* run, this compares many: for a
chosen source→target pair it reads every scored experiment back out of the
database and puts the methods side by side on performance, ranking, domain
alignment, execution cost, and transfer benefit.

Everything here is read from evaluations rows. Nothing is recomputed,
estimated, or synthesised — a metric that was never recorded (rows written
before a column existed, an AUC the classifier could not produce) is drawn
as a gap and named in the caption, never as a zero. A zero-height bar and a
missing bar mean very different things to someone reading a results chart.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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

from application.experiment_catalog_use_case import ExperimentCatalogUseCase
from domain.entities import User
from presentation.qt.duration import format_duration
from presentation.qt.comparison_charts import ComparisonChartsMixin
from presentation.qt.comparison_export import ComparisonExportMixin
from presentation.qt.comparison_export_dialog import ComparisonExportDialog
from presentation.qt.comparison_model import (
    METRICS,
    METRIC_OVERALL,
    METRIC_OVERALL_LABEL,
    SORT_METRIC_ASC,
    SORT_NAME,
    SORT_OPTIONS,
    SORT_TIME,
    TABLE_COLUMNS,
    VIEWS,
    VIEW_ALIGNMENT,
    VIEW_EXECUTION,
    VIEW_NEGATIVE,
    VIEW_PERFORMANCE,
    VIEW_RANKING,
    MethodRow,
    average_ranks,
    competition_ranks,
    rows_from_entries,
    _fmt,
)
from presentation.qt.style import DARK_PALETTE


class ComparisonTab(ComparisonChartsMixin, ComparisonExportMixin, QWidget):
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
                "—" if row.overlap_before is None else f"{row.overlap_before:.1f}%",
                "—" if row.overlap_after is None else f"{row.overlap_after:.1f}%",
                "—" if row.overlap_change is None else f"{row.overlap_change:+.1f}",
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
                ("domain overlap", any(r.overlap_before is not None for r in rows)),
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


__all__ = ["ComparisonExportDialog", "ComparisonTab"]
