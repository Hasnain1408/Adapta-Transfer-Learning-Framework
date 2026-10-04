"""Browsing the dataset catalogue.

Read-only apart from deletion: every column is derived from a file the
loader read, so editing the numbers by hand would only let them drift out
of sync with the file they describe.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from application.experiment_catalog_use_case import ExperimentCatalogUseCase
from domain.entities import DatasetRecord
from presentation.qt import message_box
from presentation.qt.dialog_utils import format_timestamp as _format_timestamp
from presentation.qt.title_bar import install_title_bar


class DatasetRegistryDialog(QDialog):
    """Browse the datasets catalogue.

    Read-only apart from deletion: every column is derived from a file the
    loader read, so editing the numbers by hand would only let them drift out
    of sync with the file they describe. New rows appear automatically when an
    experiment runs.
    """

    def __init__(
        self,
        catalog: ExperimentCatalogUseCase,
        stylesheet: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._catalog = catalog
        self._datasets: list[DatasetRecord] = []

        self.setWindowTitle("Dataset Registry")
        self.setStyleSheet(stylesheet)
        self.setMinimumSize(860, 440)

        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        self._table = QTableWidget()
        headers = [
            "ID",
            "Name",
            "Format",
            "Samples",
            "Features",
            "Label",
            "Classes",
            "Data Types",
            "Missing Values",
            "Registered",
        ]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(30)
        self._table.horizontalHeader().setHighlightSections(False)
        self._table.itemSelectionChanged.connect(self._sync_buttons)

        header = self._table.horizontalHeader()
        fixed_widths = {0: 56, 2: 80, 3: 90, 4: 90, 6: 90, 8: 120, 9: 140}
        for index in range(len(headers)):
            if index in fixed_widths:
                header.setSectionResizeMode(index, QHeaderView.Fixed)
                self._table.setColumnWidth(index, fixed_widths[index])
            else:
                header.setSectionResizeMode(index, QHeaderView.Stretch)

        self._refresh_button = QPushButton("⟳  Refresh")
        self._refresh_button.setCursor(Qt.PointingHandCursor)
        self._refresh_button.clicked.connect(self.refresh)

        self._delete_button = QPushButton("Remove from Registry")
        self._delete_button.setObjectName("dangerButton")
        self._delete_button.setCursor(Qt.PointingHandCursor)
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self._delete_selected)

        close_button = QPushButton("Close")
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.clicked.connect(self.accept)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self._refresh_button)
        actions.addWidget(self._delete_button)
        actions.addStretch(1)
        actions.addWidget(close_button)

        self._caption = QLabel()
        self._caption.setObjectName("captionLabel")

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._caption)
        layout.addLayout(actions)
        install_title_bar(self, "Dataset Registry", body)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            self._datasets = self._catalog.list_datasets()
        except Exception as exc:  # noqa: BLE001
            self._datasets = []
            self._table.setRowCount(0)
            self._caption.setText(f"Could not load datasets: {exc}")
            return

        self._table.setRowCount(len(self._datasets))
        for row, dataset in enumerate(self._datasets):
            class_names = ", ".join(dataset.class_names)
            feature_names = ", ".join(dataset.feature_names)
            cells = [
                str(dataset.id or ""),
                dataset.name,
                dataset.file_format.value,
                str(dataset.num_samples),
                str(dataset.num_features),
                dataset.label_name or "—",
                str(dataset.num_classes),
                dataset.dtype_summary or "—",
                str(dataset.missing_value_count),
                _format_timestamp(dataset.created_at),
            ]
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if column in (0, 2, 3, 4, 6, 8, 9):
                    item.setTextAlignment(Qt.AlignCenter)
                # The full path, feature list and class list are all too long
                # for a cell but valuable on hover.
                if column == 1:
                    item.setToolTip(dataset.source_path)
                elif column == 5 and feature_names:
                    item.setToolTip(f"Features: {feature_names}")
                elif column == 6 and class_names:
                    item.setToolTip(f"Classes: {class_names}")
                elif column == 8 and dataset.missing_value_count:
                    item.setToolTip(
                        "Missing/non-numeric feature cells were filled with 0.0 at load time."
                    )
                self._table.setItem(row, column, item)

        count = len(self._datasets)
        self._caption.setText(
            f"{count} dataset{'s' if count != 1 else ''} registered. "
            "Entries are added automatically when an experiment runs."
        )
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        self._delete_button.setEnabled(self._selected_dataset() is not None)

    def _selected_dataset(self) -> DatasetRecord | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._datasets):
            return self._datasets[row]
        return None

    def _delete_selected(self) -> None:
        dataset = self._selected_dataset()
        if dataset is None or dataset.id is None:
            return
        confirmed = message_box.confirm(
            self,
            "Remove Dataset",
            f"Remove {dataset.name} from the registry?",
            "The file on disk isn't touched, and experiments that used it are "
            "kept with their domain names. It re-registers next time you run it.",
            confirm_label="Remove",
            destructive=True,
        )
        if not confirmed:
            return
        try:
            self._catalog.delete_dataset(dataset.id)
        except Exception as exc:  # noqa: BLE001
            message_box.error(
                self,
                "Remove Failed",
                f"{dataset.name} wasn't removed.",
                "The registry is unchanged. Try again in a moment.",
                technical=str(exc),
            )
            return
        self.refresh()


__all__ = ["DatasetRegistryDialog"]
