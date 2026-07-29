"""Experiment history tab: browse, reload, and delete past runs."""

from __future__ import annotations

from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from application.experiment_catalog_use_case import ExperimentCatalogUseCase
from domain.entities import ExperimentHistoryEntry, ExperimentStatus

#: ``(header, width hint)``. ``None`` means the column stretches.
_COLUMNS: list[tuple[str, int | None]] = [
    ("ID", 56),
    ("Date", 150),
    ("User", None),
    ("Algorithm", 110),
    ("Classifier", 130),
    ("Source → Target", None),
    ("Status", 100),
    ("Accuracy", 90),
    ("F1", 90),
]


def _format_timestamp(value: object) -> str:
    """Render a TIMESTAMPTZ as local time, tolerating a plain string."""
    if isinstance(value, datetime):
        return value.astimezone().strftime("%Y-%m-%d %H:%M")
    return str(value or "")


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


class HistoryTab(QWidget):
    """Table of recorded experiments with reload / delete actions.

    Emits :attr:`configuration_requested` when the user asks to reload a past
    run's settings; ``MainWindow`` owns the form widgets, so applying the
    configuration is its job, not this widget's.
    """

    configuration_requested = pyqtSignal(object)  # ExperimentRecord

    def __init__(
        self,
        catalog: ExperimentCatalogUseCase | None,
        palette: dict[str, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._catalog = catalog
        self._palette = palette
        self._entries: list[ExperimentHistoryEntry] = []
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        # ── Toolbar ──────────────────────────────────────────────────────
        user_label = QLabel("Filter by user:")
        user_label.setObjectName("fieldLabel")

        self._user_filter = QComboBox()
        self._user_filter.setMinimumWidth(200)
        self._user_filter.setToolTip("Show only experiments owned by this user")
        self._user_filter.currentIndexChanged.connect(self.refresh)

        self._refresh_button = QPushButton("⟳  Refresh")
        self._refresh_button.setCursor(Qt.PointingHandCursor)
        self._refresh_button.clicked.connect(self.refresh)

        self._load_button = QPushButton("Load Configuration")
        self._load_button.setCursor(Qt.PointingHandCursor)
        self._load_button.setToolTip(
            "Copy the selected experiment's settings back into the run form"
        )
        self._load_button.setEnabled(False)
        self._load_button.clicked.connect(self._emit_configuration)

        self._delete_button = QPushButton("Delete")
        self._delete_button.setObjectName("dangerButton")
        self._delete_button.setCursor(Qt.PointingHandCursor)
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self._delete_selected)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.addWidget(user_label)
        toolbar.addWidget(self._user_filter)
        toolbar.addStretch(1)
        toolbar.addWidget(self._load_button)
        toolbar.addWidget(self._delete_button)
        toolbar.addWidget(self._refresh_button)

        # ── Table ────────────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([header for header, _ in _COLUMNS])
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(30)
        self._table.horizontalHeader().setHighlightSections(False)
        self._table.itemSelectionChanged.connect(self._sync_buttons)
        self._table.doubleClicked.connect(self._emit_configuration)

        # Per-column resize modes must be set instead of calling
        # resizeColumnsToContents(), which is a no-op once a mode is applied.
        header = self._table.horizontalHeader()
        for index, (_, width) in enumerate(_COLUMNS):
            if width is None:
                header.setSectionResizeMode(index, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(index, QHeaderView.Fixed)
                self._table.setColumnWidth(index, width)

        self._caption = QLabel()
        self._caption.setObjectName("captionLabel")

        layout.addLayout(toolbar)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._caption)

    # ------------------------------------------------------------------
    def set_palette(self, palette: dict[str, str]) -> None:
        """Re-colour status/score cells after a theme switch."""
        self._palette = palette
        self._populate()

    def set_catalog(self, catalog: ExperimentCatalogUseCase | None) -> None:
        self._catalog = catalog
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Reload users and experiments from the database."""
        if self._catalog is None:
            self._entries = []
            self._populate()
            self._caption.setText("History is unavailable — no database connection.")
            return

        self._reload_user_filter()

        try:
            self._entries = self._catalog.history(self._selected_user_id())
        except Exception as exc:  # noqa: BLE001 - surfaced in the caption
            self._entries = []
            self._populate()
            self._caption.setText(f"Could not load history: {exc}")
            return

        self._populate()
        total = len(self._entries)
        self._caption.setText(
            "No experiments recorded yet."
            if total == 0
            else f"{total} experiment{'s' if total != 1 else ''} recorded."
        )

    def _reload_user_filter(self) -> None:
        """Rebuild the filter combo, preserving the current selection.

        Signals are blocked while repopulating: ``currentIndexChanged`` is
        wired to :meth:`refresh`, so mutating the combo inside refresh would
        otherwise recurse.
        """
        assert self._catalog is not None
        previous = self._selected_user_id()

        self._user_filter.blockSignals(True)
        try:
            self._user_filter.clear()
            self._user_filter.addItem("All users", None)
            for user in self._catalog.list_users():
                self._user_filter.addItem(f"{user.name} ({user.role.label})", user.id)
            if previous is not None:
                index = self._user_filter.findData(previous)
                if index >= 0:
                    self._user_filter.setCurrentIndex(index)
        finally:
            self._user_filter.blockSignals(False)

    def _selected_user_id(self) -> int | None:
        return self._user_filter.currentData() if self._user_filter.count() else None

    # ------------------------------------------------------------------
    def _populate(self) -> None:
        table = self._table
        table.clearContents()
        table.setRowCount(len(self._entries))

        status_colours = {
            ExperimentStatus.COMPLETED: QBrush(QColor(self._palette["success"])),
            ExperimentStatus.FAILED: QBrush(QColor(self._palette["danger"])),
            ExperimentStatus.RUNNING: QBrush(QColor(self._palette["warning"])),
            ExperimentStatus.PENDING: QBrush(QColor(self._palette["text_muted"])),
        }

        for row, entry in enumerate(self._entries):
            experiment = entry.experiment
            evaluation = entry.evaluation

            cells = [
                str(experiment.id or ""),
                _format_timestamp(experiment.date),
                entry.user.name if entry.user else "—",
                experiment.tl_algorithm,
                experiment.classifier or "—",
                f"{experiment.source_domain} → {experiment.target_domain}",
                experiment.status.label,
                _format_score(evaluation.accuracy if evaluation else None),
                _format_score(evaluation.f1_score if evaluation else None),
            ]

            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if column in (0, 6, 7, 8):
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 6:
                    item.setForeground(
                        status_colours.get(
                            experiment.status, QBrush(QColor(self._palette["text"]))
                        )
                    )
                # A failed run's message is the single most useful thing to
                # surface, and there is no column wide enough for it.
                if experiment.error_message:
                    item.setToolTip(experiment.error_message)
                table.setItem(row, column, item)

        self._sync_buttons()

    def _sync_buttons(self) -> None:
        has_selection = self._selected_entry() is not None
        self._load_button.setEnabled(has_selection)
        self._delete_button.setEnabled(has_selection)

    def _selected_entry(self) -> ExperimentHistoryEntry | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

    # ------------------------------------------------------------------
    def _emit_configuration(self) -> None:
        entry = self._selected_entry()
        if entry is not None:
            self.configuration_requested.emit(entry.experiment)

    def _delete_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None or self._catalog is None or entry.experiment.id is None:
            return

        confirmed = QMessageBox.question(
            self,
            "Delete Experiment",
            f"Delete experiment #{entry.experiment.id} "
            f"({entry.experiment.tl_algorithm})?\n\n"
            "Its evaluation scores are deleted with it. This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            return

        try:
            self._catalog.delete_experiment(entry.experiment.id)
        except Exception as exc:  # noqa: BLE001 - shown verbatim
            QMessageBox.critical(
                self, "Delete Failed", f"The experiment could not be deleted:\n\n{exc}"
            )
            return
        self.refresh()
