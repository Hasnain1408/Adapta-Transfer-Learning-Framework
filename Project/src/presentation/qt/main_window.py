from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.models import TransferLearningRequest, TransferLearningResult
from presentation.qt.worker import TransferLearningWorker


class MainWindow(QMainWindow):
    def __init__(self, use_case: RunTransferLearningUseCase) -> None:
        super().__init__()
        self._use_case = use_case
        self._worker: TransferLearningWorker | None = None

        self.setWindowTitle("Transfer Learning Desktop App")
        self.resize(1000, 680)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["TCA"])
        self._method_combo.addItems(["JDA"])

        self._source_input = QLineEdit()
        self._source_input.setPlaceholderText("Select source CSV or XLSX dataset")
        self._target_input = QLineEdit()
        self._target_input.setPlaceholderText("Select target CSV or XLSX dataset")

        self._browse_source_button = QPushButton("Browse")
        self._browse_target_button = QPushButton("Browse")
        self._run_button = QPushButton("Run Transfer Learning")
        self._status_label = QLabel("Status: Ready")

        self._dataset_summary_table = QTableWidget()
        self._dataset_summary_table.setColumnCount(5)
        self._dataset_summary_table.setHorizontalHeaderLabels(
            ["Dataset", "Rows", "Feature Size", "Feature Names", "Label Name"]
        )
        self._dataset_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._dataset_summary_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._dataset_summary_table.setFocusPolicy(Qt.NoFocus)

        self._result_table = QTableWidget()
        self._result_table.setColumnCount(0)
        self._result_table.setRowCount(0)

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        central_widget = QWidget(self)
        root_layout = QVBoxLayout(central_widget)

        form_layout = QGridLayout()

        form_layout.addWidget(QLabel("Method"), 0, 0)
        form_layout.addWidget(self._method_combo, 0, 1, 1, 2)

        form_layout.addWidget(QLabel("Source Dataset"), 1, 0)
        form_layout.addWidget(self._source_input, 1, 1)
        form_layout.addWidget(self._browse_source_button, 1, 2)

        form_layout.addWidget(QLabel("Target Dataset"), 2, 0)
        form_layout.addWidget(self._target_input, 2, 1)
        form_layout.addWidget(self._browse_target_button, 2, 2)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self._run_button)
        action_layout.addWidget(self._status_label)
        action_layout.setAlignment(self._status_label, Qt.AlignVCenter)

        root_layout.addLayout(form_layout)
        root_layout.addLayout(action_layout)
        root_layout.addWidget(QLabel("Source and Target Dataset Summary"))
        root_layout.addWidget(self._dataset_summary_table)
        root_layout.addWidget(QLabel("Transformed Target Dataset (preview)"))
        root_layout.addWidget(self._result_table)

        self.setCentralWidget(central_widget)

    def _connect_signals(self) -> None:
        self._browse_source_button.clicked.connect(lambda: self._pick_file(self._source_input))
        self._browse_target_button.clicked.connect(lambda: self._pick_file(self._target_input))
        self._run_button.clicked.connect(self._run_transfer_learning)

    def _pick_file(self, target_input: QLineEdit) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            str(Path.cwd()),
            "Dataset Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)",
        )
        if file_path:
            target_input.setText(file_path)

    def _run_transfer_learning(self) -> None:
        source_path = self._source_input.text().strip()
        target_path = self._target_input.text().strip()

        if not source_path or not target_path:
            QMessageBox.warning(self, "Missing Input", "Please select both source and target datasets.")
            return

        if not Path(source_path).exists() or not Path(target_path).exists():
            QMessageBox.warning(self, "Invalid Path", "One or more selected dataset files do not exist.")
            return

        request = TransferLearningRequest(
            method_name=self._method_combo.currentText(),
            source_path=source_path,
            target_path=target_path,
        )

        self._run_button.setEnabled(False)
        self._status_label.setText("Status: Running in background...")

        self._worker = TransferLearningWorker(self._use_case, request)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(lambda: self._run_button.setEnabled(True))
        self._worker.start()

    def _on_success(self, result: TransferLearningResult) -> None:
        self._status_label.setText("Status: Completed")
        self._show_dataset_summary(result.source_dataset, result.target_dataset)
        self._show_result(result.transformed_target, result.output_feature_names)

    def _on_error(self, message: str) -> None:
        self._status_label.setText("Status: Failed")
        QMessageBox.critical(self, "Transfer Learning Failed", message)

    def _show_result(self, matrix: np.ndarray, headers: list[str]) -> None:
        row_count, col_count = matrix.shape
        preview_rows = min(row_count, 200)

        self._result_table.clear()
        self._result_table.setRowCount(preview_rows)
        self._result_table.setColumnCount(col_count)
        self._result_table.setHorizontalHeaderLabels(headers)

        for row in range(preview_rows):
            for col in range(col_count):
                value = f"{matrix[row, col]:.6f}"
                self._result_table.setItem(row, col, QTableWidgetItem(value))

    def _show_dataset_summary(self, source_dataset, target_dataset) -> None:
        summaries = [
            ("Source", source_dataset),
            ("Target", target_dataset),
        ]

        self._dataset_summary_table.setRowCount(len(summaries))

        for row, (dataset_label, dataset) in enumerate(summaries):
            feature_names = ", ".join(dataset.feature_names)
            values = [
                dataset_label,
                str(dataset.features.shape[0]),
                str(dataset.features.shape[1]),
                feature_names,
                dataset.label_name,
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 3:
                    item.setToolTip(feature_names)
                self._dataset_summary_table.setItem(row, col, item)

        self._dataset_summary_table.resizeColumnsToContents()
