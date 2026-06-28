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
    QHeaderView,
)

from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.models import TransferLearningRequest, TransferLearningResult, EvaluationMetrics
from presentation.qt.worker import TransferLearningWorker


STYLE_SHEET = """
QMainWindow {
    background-color: #0f172a;
}
QLabel {
    color: #94a3b8;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
    font-weight: 500;
}
QLineEdit {
    background-color: #1e293b;
    border: 2px solid #334155;
    border-radius: 6px;
    color: #f8fafc;
    padding: 8px 12px;
    font-size: 13px;
    font-family: 'Segoe UI', sans-serif;
}
QLineEdit:focus {
    border: 2px solid #3b82f6;
    background-color: #0d1527;
}
QComboBox {
    background-color: #1e293b;
    border: 2px solid #334155;
    border-radius: 6px;
    color: #f8fafc;
    padding: 8px 12px;
    font-size: 13px;
    font-family: 'Segoe UI', sans-serif;
}
QComboBox:focus {
    border: 2px solid #3b82f6;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #334155;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
    color: #f8fafc;
}
QPushButton {
    background-color: #2563eb;
    border: none;
    border-radius: 6px;
    color: #ffffff;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    font-family: 'Segoe UI', sans-serif;
}
QPushButton:hover {
    background-color: #3b82f6;
}
QPushButton:pressed {
    background-color: #1d4ed8;
}
QPushButton#runButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #06b6d4);
    color: #ffffff;
    font-size: 14px;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 6px;
}
QPushButton#runButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #0891b2);
}
QPushButton#runButton:disabled {
    background: #334155;
    color: #64748b;
}
QPushButton#browseButton {
    background-color: #334155;
    font-weight: 500;
}
QPushButton#browseButton:hover {
    background-color: #475569;
}
QPushButton#browseButton:pressed {
    background-color: #1e293b;
}
QTableWidget {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    gridline-color: #334155;
    color: #cbd5e1;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
}
QTableWidget::item {
    border-bottom: 1px solid #334155;
    padding: 6px;
}
QTableWidget::item:selected {
    background-color: #1e3a8a;
    color: #f8fafc;
}
QHeaderView::section {
    background-color: #0f172a;
    color: #38bdf8;
    padding: 8px;
    font-weight: bold;
    font-size: 12px;
    border: none;
    border-bottom: 2px solid #334155;
    border-right: 1px solid #334155;
}
"""


class MainWindow(QMainWindow):
    def __init__(self, use_case: RunTransferLearningUseCase) -> None:
        super().__init__()
        self._use_case = use_case
        self._worker: TransferLearningWorker | None = None

        self.setWindowTitle("Transfer Learning Desktop App")
        self.resize(1000, 720)
        self.setStyleSheet(STYLE_SHEET)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["TCA"])
        self._method_combo.addItems(["JDA"])

        self._source_input = QLineEdit()
        self._source_input.setPlaceholderText("Select source CSV or XLSX dataset")
        self._target_input = QLineEdit()
        self._target_input.setPlaceholderText("Select target CSV or XLSX dataset")

        self._browse_source_button = QPushButton("Browse")
        self._browse_source_button.setObjectName("browseButton")
        self._browse_target_button = QPushButton("Browse")
        self._browse_target_button.setObjectName("browseButton")
        
        self._run_button = QPushButton("Run Transfer Learning")
        self._run_button.setObjectName("runButton")
        self._status_label = QLabel()
        self._set_status("Ready", "#64748b")

        self._dataset_summary_table = QTableWidget()
        self._dataset_summary_table.setColumnCount(5)
        self._dataset_summary_table.setHorizontalHeaderLabels(
            ["Dataset", "Rows", "Feature Size", "Feature Names", "Label Name"]
        )
        self._dataset_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._dataset_summary_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._dataset_summary_table.setFocusPolicy(Qt.NoFocus)
        self._dataset_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(4)
        self._metrics_table.setHorizontalHeaderLabels(
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        )
        self._metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._metrics_table.setFocusPolicy(Qt.NoFocus)
        self._metrics_table.setRowCount(1)
        self._metrics_table.setFixedHeight(65)
        self._metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for col in range(4):
            self._metrics_table.setItem(0, col, QTableWidgetItem("N/A"))

        self._result_table = QTableWidget()
        self._result_table.setColumnCount(0)
        self._result_table.setRowCount(0)
        self._result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self._build_layout()
        self._connect_signals()

    def _create_heading(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8; margin-top: 10px; margin-bottom: 2px;")
        return lbl

    def _set_status(self, text: str, color: str) -> None:
        self._status_label.setText(f"Status: {text}")
        self._status_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px; margin-left: 10px;")

    def _build_layout(self) -> None:
        central_widget = QWidget(self)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(14)

        form_layout = QGridLayout()
        form_layout.setSpacing(12)

        lbl_method = QLabel("Method")
        lbl_method.setStyleSheet("font-weight: bold; color: #cbd5e1;")
        form_layout.addWidget(lbl_method, 0, 0)
        form_layout.addWidget(self._method_combo, 0, 1, 1, 2)

        lbl_src = QLabel("Source Dataset")
        lbl_src.setStyleSheet("font-weight: bold; color: #cbd5e1;")
        form_layout.addWidget(lbl_src, 1, 0)
        form_layout.addWidget(self._source_input, 1, 1)
        form_layout.addWidget(self._browse_source_button, 1, 2)

        lbl_target = QLabel("Target Dataset")
        lbl_target.setStyleSheet("font-weight: bold; color: #cbd5e1;")
        form_layout.addWidget(lbl_target, 2, 0)
        form_layout.addWidget(self._target_input, 2, 1)
        form_layout.addWidget(self._browse_target_button, 2, 2)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self._run_button)
        action_layout.addWidget(self._status_label)
        action_layout.setAlignment(self._status_label, Qt.AlignVCenter)
        action_layout.addStretch()

        root_layout.addLayout(form_layout)
        root_layout.addLayout(action_layout)
        root_layout.addWidget(self._create_heading("Source and Target Dataset Summary"))
        root_layout.addWidget(self._dataset_summary_table)
        root_layout.addWidget(self._create_heading("KNN Evaluation Metrics (Trained on Source, Tested on Target)"))
        root_layout.addWidget(self._metrics_table)
        root_layout.addWidget(self._create_heading("Transformed Target Dataset (preview)"))
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
        self._set_status("Running in background...", "#f59e0b")

        self._worker = TransferLearningWorker(self._use_case, request)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(lambda: self._run_button.setEnabled(True))
        self._worker.start()

    def _on_success(self, result: TransferLearningResult) -> None:
        self._set_status("Completed", "#10b981")
        self._show_dataset_summary(result.source_dataset, result.target_dataset)
        self._show_result(result.transformed_target, result.output_feature_names)
        self._show_metrics(result.metrics)

    def _on_error(self, message: str) -> None:
        self._set_status("Failed", "#ef4444")
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
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self._result_table.setItem(row, col, item)

        self._result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

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
                item.setTextAlignment(Qt.AlignCenter)
                if col == 3:
                    item.setToolTip(feature_names)
                self._dataset_summary_table.setItem(row, col, item)

        self._dataset_summary_table.resizeColumnsToContents()

    def _show_metrics(self, metrics: EvaluationMetrics | None) -> None:
        if metrics is None:
            for col in range(4):
                item = QTableWidgetItem("N/A")
                item.setTextAlignment(Qt.AlignCenter)
                self._metrics_table.setItem(0, col, item)
            return

        values = [
            f"{metrics.accuracy:.4f}",
            f"{metrics.precision:.4f}",
            f"{metrics.recall:.4f}",
            f"{metrics.f1_score:.4f}"
        ]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self._metrics_table.setItem(0, col, item)
        self._metrics_table.resizeColumnsToContents()
