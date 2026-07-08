from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QColor, QBrush
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QSizePolicy,
    QSpacerItem,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.models import TransferLearningRequest, TransferLearningResult, EvaluationMetrics
from presentation.qt.worker import TransferLearningWorker
from presentation.qt.style import STYLE_SHEET
class MainWindow(QMainWindow):
    def __init__(self, use_case: RunTransferLearningUseCase) -> None:
        super().__init__()
        self._use_case = use_case
        self._worker: TransferLearningWorker | None = None

        self.setWindowTitle("Transfer Learning Desktop App")
        self.setMinimumSize(980, 700)
        self.resize(1080, 760)
        self.setStyleSheet(STYLE_SHEET)

        style = self.style()
        self.setWindowIcon(style.standardIcon(QStyle.SP_ComputerIcon))

        self._method_combo = QComboBox()
        self._method_combo.addItems(["TCA", "JDA", "CORAL", "BDA", "OCMTL"])
        self._method_combo.setMinimumHeight(38)

        self._source_input = QLineEdit()
        self._source_input.setPlaceholderText("Select source CSV, XLSX, or MAT dataset")
        self._target_input = QLineEdit()
        self._target_input.setPlaceholderText("Select target CSV, XLSX, or MAT dataset")

        self._browse_source_button = QPushButton("Browse\u2026")
        self._browse_source_button.setObjectName("browseButton")
        self._browse_source_button.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        self._browse_target_button = QPushButton("Browse\u2026")
        self._browse_target_button.setObjectName("browseButton")
        self._browse_target_button.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))

        self._run_button = QPushButton("Run Transfer Learning")
        self._run_button.setObjectName("runButton")
        self._run_button.setCursor(Qt.PointingHandCursor)
        self._run_button.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self._run_button.setMinimumHeight(42)

        self._status_dot = QLabel("\u25cf")
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
        self._dataset_summary_table.setAlternatingRowColors(True)
        self._dataset_summary_table.verticalHeader().setVisible(False)
        self._dataset_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._dataset_summary_table.setFixedHeight(110)

        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(7)
        self._metrics_table.setHorizontalHeaderLabels(
            ["Total Samples", "Correct", "Wrong", "Accuracy", "Precision", "Recall", "F1 Score"]
        )
        self._metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._metrics_table.setFocusPolicy(Qt.NoFocus)
        self._metrics_table.setAlternatingRowColors(True)
        self._metrics_table.verticalHeader().setVisible(False)
        self._metrics_table.setRowCount(1)
        self._metrics_table.setFixedHeight(68)
        self._metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for col in range(7):
            item = QTableWidgetItem("N/A")
            item.setTextAlignment(Qt.AlignCenter)
            self._metrics_table.setItem(0, col, item)

        self._result_table = QTableWidget()
        self._result_table.setColumnCount(0)
        self._result_table.setRowCount(0)
        self._result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._result_table.setAlternatingRowColors(True)
        self._result_table.verticalHeader().setVisible(False)

        self._build_layout()
        self._connect_signals()

    def _create_heading(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size: 14px; font-weight: 700; color: #38bdf8; "
            "margin-top: 6px; margin-bottom: 2px; letter-spacing: 0.2px;"
        )
        return lbl

    def _create_field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("fieldLabel")
        return lbl

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setObjectName("divider")
        line.setFrameShape(QFrame.HLine)
        return line

    def _set_status(self, text: str, color: str) -> None:
        self._status_dot.setStyleSheet(f"color: {color}; font-size: 12px;")
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; font-weight: 700; font-size: 13px;"
        )

    def _build_layout(self) -> None:
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(16)

        # --- Header ---
        header_layout = QVBoxLayout()
        header_layout.setSpacing(2)
        title_label = QLabel("Transfer Learning Studio")
        title_label.setObjectName("appTitle")
        subtitle_label = QLabel("Domain adaptation experiments powered by TCA, JDA, CORAL, BDA and OCMTL")
        subtitle_label.setObjectName("appSubtitle")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        root_layout.addLayout(header_layout)
        root_layout.addWidget(self._divider())

        # --- Configuration group ---
        config_group = QGroupBox("Configuration")
        form_layout = QGridLayout(config_group)
        form_layout.setHorizontalSpacing(14)
        form_layout.setVerticalSpacing(10)
        form_layout.setColumnStretch(1, 1)

        form_layout.addWidget(self._create_field_label("Method"), 0, 0)
        form_layout.addWidget(self._method_combo, 0, 1, 1, 2)

        form_layout.addWidget(self._create_field_label("Source Dataset"), 1, 0)
        form_layout.addWidget(self._source_input, 1, 1)
        form_layout.addWidget(self._browse_source_button, 1, 2)

        form_layout.addWidget(self._create_field_label("Target Dataset"), 2, 0)
        form_layout.addWidget(self._target_input, 2, 1)
        form_layout.addWidget(self._browse_target_button, 2, 2)

        root_layout.addWidget(config_group)

        # --- Action row ---
        action_layout = QHBoxLayout()
        action_layout.setSpacing(14)
        action_layout.addWidget(self._run_button)

        status_frame = QFrame()
        status_frame.setObjectName("statusCard")
        status_inner = QHBoxLayout(status_frame)
        status_inner.setContentsMargins(14, 6, 16, 6)
        status_inner.setSpacing(8)
        status_inner.addWidget(self._status_dot)
        status_inner.addWidget(self._status_label)
        action_layout.addWidget(status_frame)
        action_layout.addStretch()

        root_layout.addLayout(action_layout)

        # --- Results ---
        root_layout.addWidget(self._create_heading("Source and Target Dataset Summary"))
        root_layout.addWidget(self._dataset_summary_table)

        root_layout.addWidget(self._create_heading("KNN Evaluation Metrics (Trained on Source, Tested on Target)"))
        root_layout.addWidget(self._metrics_table)

        root_layout.addWidget(self._create_heading("Transformed Target Dataset & Predictions "))
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
            "Dataset Files (*.csv *.xlsx *.mat);;CSV Files (*.csv);;Excel Files (*.xlsx);;MAT Files (*.mat)",
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
        self._set_status("Running in background\u2026", "#f59e0b")

        self._worker = TransferLearningWorker(self._use_case, request)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(lambda: self._run_button.setEnabled(True))
        self._worker.start()

    def _on_success(self, result: TransferLearningResult) -> None:
        self._set_status("Completed", "#10b981")
        self._show_dataset_summary(result.source_dataset, result.target_dataset)
        self._show_metrics(result.metrics)
        self._show_result(result)

    def _on_error(self, message: str) -> None:
        self._set_status("Failed", "#ef4444")
        QMessageBox.critical(self, "Transfer Learning Failed", message)

    def _show_result(self, result: TransferLearningResult) -> None:
        matrix = result.transformed_target
        has_predictions = result.target_predictions is not None

        if has_predictions:
            headers = ["Sample Name", "Actual Label", "Predicted Label", "Prediction"]
        else:
            headers = ["Sample Name"] + result.output_feature_names[:]

        row_count = matrix.shape[0]
        col_count = len(headers)
        preview_rows = row_count

        self._result_table.clear()
        self._result_table.setRowCount(preview_rows)
        self._result_table.setColumnCount(col_count)
        self._result_table.setHorizontalHeaderLabels(headers)

        for row in range(preview_rows):
            sample_name = f"Sample {row + 1}"
            
            if has_predictions:
                actual_val = str(result.target_actuals[row])
                pred_val = str(result.target_predictions[row])
                correctness_val = result.prediction_correctness[row]
                
                items = [sample_name, actual_val, pred_val, correctness_val]
                for col, val in enumerate(items):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    
                    if col == 3:
                        if val == "Right":
                            item.setForeground(QBrush(QColor("#10b981")))
                        elif val == "Wrong":
                            item.setForeground(QBrush(QColor("#ef4444")))
                            
                    self._result_table.setItem(row, col, item)
            else:
                item_name = QTableWidgetItem(sample_name)
                item_name.setTextAlignment(Qt.AlignCenter)
                self._result_table.setItem(row, 0, item_name)
                
                for col in range(matrix.shape[1]):
                    value = f"{matrix[row, col]:.6f}"
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignCenter)
                    self._result_table.setItem(row, col + 1, item)

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
            for col in range(7):
                item = QTableWidgetItem("N/A")
                item.setTextAlignment(Qt.AlignCenter)
                self._metrics_table.setItem(0, col, item)
            return

        values = [
            str(metrics.total_samples),
            str(metrics.correct_predictions),
            str(metrics.wrong_predictions),
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