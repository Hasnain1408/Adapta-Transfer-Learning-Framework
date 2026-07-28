from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QKeySequence, QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.models import EvaluationMetrics, TransferLearningRequest, TransferLearningResult
from presentation.qt.style import PALETTE, STYLE_SHEET
from presentation.qt.worker import TransferLearningWorker

#: ``(display label, value)``. The value is what the use case dispatches on, so
#: labels can be reworded without silently disabling a transform.
NORMALIZATION_OPTIONS: list[tuple[str, str]] = [
    ("None", "None"),
    ("Min-Max Normalization", "Min-Max Normalization"),
    ("Z-score Standardization", "Z-score Standardization"),
    ("Robust Standardization", "Robust Standardization"),
    ("L1 normalization", "L1 normalization"),
    ("L2 (Unit Vector) Normalization", "L2 (Unit Vector) Normalization"),
    ("Mean Centering", "Mean Centering"),
    ("Column-wise unit norm", "Column-wise unit norm"),
    ("Row-wise unit norm", "Row-wise unit norm"),
    ("Frobenius normalization", "Frobenius normalization"),
    ("Spectral normalization", "Spectral normalization"),
]

STANDARDIZATION_OPTIONS: list[tuple[str, str]] = [
    ("None", "None"),
    ("Z-score Standardization", "Z-score Standardization"),
    ("Robust Standardization", "Robust Standardization"),
    ("Mahalanobis Standardization", "Mahalanobis Standardization"),
    ("Variance Scaling", "Variance Scaling"),
    ("Column mean centering", "Column mean centering"),
    ("Row mean centering", "Row mean centering"),
]

KERNEL_OPTIONS: list[tuple[str, str]] = [
    ("Primal (no kernel)", "primal"),
    ("Linear", "linear"),
    ("RBF (Gaussian)", "rbf"),
    ("SAM (spectral angle)", "sam"),
]

CLASSIFIER_OPTIONS: list[tuple[str, str]] = [
    ("k-NN (5-Nearest Neighbour)", "k-NN"),
    ("SVM", "SVM"),
    ("Random Forest", "Random Forest"),
    ("Logistic Regression", "Logistic Regression"),
    ("Decision Tree", "Decision Tree"),
]

DATASET_FILTER = (
    "Dataset Files (*.csv *.xlsx *.mat);;"
    "CSV Files (*.csv);;Excel Files (*.xlsx);;MATLAB Files (*.mat)"
)

#: Rows rendered in the transformed-features preview. Wide matrices make full
#: rendering slow, so the cap is surfaced in the caption rather than hidden.
FEATURE_PREVIEW_ROWS = 1000


class MetricCard(QFrame):
    """A single headline metric, shown as a titled value tile."""

    def __init__(self, title: str, tooltip: str = "") -> None:
        super().__init__()
        self.setObjectName("metricCard")
        if tooltip:
            self.setToolTip(tooltip)

        self._value_label = QLabel("—")
        self._value_label.setObjectName("metricCardValue")
        self._value_label.setProperty("placeholder", True)

        title_label = QLabel(title)
        title_label.setObjectName("metricCardTitle")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 11)
        layout.setSpacing(2)
        layout.addWidget(title_label)
        layout.addWidget(self._value_label)

    def set_value(self, text: str | None) -> None:
        is_placeholder = text is None
        self._value_label.setText("—" if is_placeholder else text)
        self._value_label.setProperty("placeholder", is_placeholder)
        # Dynamic properties only affect QSS after a re-polish.
        self._value_label.style().unpolish(self._value_label)
        self._value_label.style().polish(self._value_label)


class MainWindow(QMainWindow):
    def __init__(self, use_case: RunTransferLearningUseCase) -> None:
        super().__init__()
        self._use_case = use_case
        self._worker: TransferLearningWorker | None = None
        self._last_result: TransferLearningResult | None = None

        self.setWindowTitle("Transfer Learning Studio")
        self.setMinimumSize(1000, 720)
        self.resize(1180, 820)
        self.setStyleSheet(STYLE_SHEET)

        self._build_widgets()
        self._build_menu()
        self._build_layout()
        self._build_status_bar()
        self._connect_signals()
        self._sync_method_parameters()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        self._source_input = QLineEdit()
        self._source_input.setPlaceholderText("No source dataset selected")
        self._source_input.setClearButtonEnabled(True)
        self._target_input = QLineEdit()
        self._target_input.setPlaceholderText("No target dataset selected")
        self._target_input.setClearButtonEnabled(True)

        self._browse_source_button = QPushButton("Browse…")
        self._browse_target_button = QPushButton("Browse…")
        for button in (self._browse_source_button, self._browse_target_button):
            button.setCursor(Qt.PointingHandCursor)

        self._method_combo = QComboBox()
        for name in self._use_case.method_names:
            self._method_combo.addItem(name, name)
        self._method_combo.setToolTip("Domain adaptation algorithm to apply")

        self._kernel_combo = self._build_option_combo(
            KERNEL_OPTIONS, "Kernel used to build the Gram matrix"
        )
        self._kernel_combo.setCurrentIndex(1)  # Linear

        self._classifier_combo = self._build_option_combo(
            CLASSIFIER_OPTIONS, "Classifier trained on the transformed source domain and evaluated on the target"
        )

        self._norm_combo = self._build_option_combo(
            NORMALIZATION_OPTIONS, "Applied to source and target features before adaptation"
        )
        self._std_combo = self._build_option_combo(
            STANDARDIZATION_OPTIONS, "Applied after normalization, before adaptation"
        )

        self._reg_input = QDoubleSpinBox()
        self._reg_input.setDecimals(4)
        self._reg_input.setRange(0.0001, 1_000_000.0)
        self._reg_input.setSingleStep(0.1)
        self._reg_input.setValue(1.0)
        self._reg_input.setToolTip("Trade-off term λ added to the eigenproblem")

        self._dim_input = QSpinBox()
        self._dim_input.setRange(1, 4096)
        self._dim_input.setValue(10)
        self._dim_input.setToolTip("Number of latent components to project onto")

        # Numeric fields read better at their natural size than stretched wide.
        for spin_box in (self._reg_input, self._dim_input):
            spin_box.setMaximumWidth(170)
            spin_box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._run_button = QPushButton("Run Experiment")
        self._run_button.setObjectName("runButton")
        self._run_button.setCursor(Qt.PointingHandCursor)
        self._run_button.setDefault(True)

        self._export_button = QPushButton("Export Report…")
        self._export_button.setCursor(Qt.PointingHandCursor)
        self._export_button.setEnabled(False)

        self._accuracy_card = MetricCard("Accuracy", "Share of target samples predicted correctly")
        self._precision_card = MetricCard("Precision", "True positives over predicted positives")
        self._recall_card = MetricCard("Recall", "True positives over actual positives")
        self._f1_card = MetricCard("F1 Score", "Harmonic mean of precision and recall")
        self._metric_cards = (
            self._accuracy_card,
            self._precision_card,
            self._recall_card,
            self._f1_card,
        )

        self._counts_label = QLabel("Run an experiment to evaluate the target domain.")
        self._counts_label.setObjectName("captionLabel")

        self._incorrect_only_check = QCheckBox("Show incorrect predictions only")
        self._incorrect_only_check.setEnabled(False)

        self._predictions_table = self._build_table(
            ["Sample", "Actual Label", "Predicted Label", "Result"]
        )
        self._align_headers(
            self._predictions_table,
            [Qt.AlignLeft, Qt.AlignCenter, Qt.AlignCenter, Qt.AlignCenter],
        )
        self._features_table = self._build_table([])
        self._summary_table = self._build_table(
            ["Dataset", "Samples", "Features", "Label Column", "Feature Names"]
        )
        self._align_headers(
            self._summary_table,
            [Qt.AlignLeft, Qt.AlignRight, Qt.AlignRight, Qt.AlignLeft, Qt.AlignLeft],
        )
        self._summary_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._features_caption = QLabel()
        self._features_caption.setObjectName("captionLabel")
        self._features_caption.hide()

    def _build_option_combo(self, options: list[tuple[str, str]], tooltip: str) -> QComboBox:
        combo = QComboBox()
        for label, value in options:
            combo.addItem(label, value)
        combo.setToolTip(tooltip)
        return combo

    @staticmethod
    def _align_headers(table: QTableWidget, alignments: list[int]) -> None:
        """Match each header's alignment to the cells beneath it."""
        for column, alignment in enumerate(alignments):
            item = table.horizontalHeaderItem(column)
            if item is not None:
                item.setTextAlignment(alignment | Qt.AlignVCenter)

    def _build_table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(headers))
        if headers:
            table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(30)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _build_menu(self) -> None:
        self._open_source_action = QAction("Open &Source Dataset…", self)
        self._open_source_action.setShortcut(QKeySequence("Ctrl+O"))
        self._open_target_action = QAction("Open &Target Dataset…", self)
        self._open_target_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self._export_action = QAction("&Export Report…", self)
        self._export_action.setShortcut(QKeySequence("Ctrl+E"))
        self._export_action.setEnabled(False)
        self._quit_action = QAction("&Quit", self)
        self._quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self._run_action = QAction("&Run Experiment", self)
        self._run_action.setShortcut(QKeySequence("F5"))
        self._about_action = QAction("&About", self)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._open_source_action)
        file_menu.addAction(self._open_target_action)
        file_menu.addSeparator()
        file_menu.addAction(self._export_action)
        file_menu.addSeparator()
        file_menu.addAction(self._quit_action)

        run_menu = self.menuBar().addMenu("&Run")
        run_menu.addAction(self._run_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self._about_action)

    def _build_layout(self) -> None:
        central = QWidget(self)
        central.setObjectName("centralWidget")
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 10)
        root.setSpacing(11)

        root.addWidget(self._build_dataset_group())
        root.addWidget(self._build_parameter_group())
        root.addLayout(self._build_action_row())
        root.addLayout(self._build_metrics_row())
        root.addWidget(self._counts_label)
        root.addWidget(self._build_results_stack(), 1)

        self.setCentralWidget(central)

    def _build_dataset_group(self) -> QGroupBox:
        group = QGroupBox("Datasets")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        grid.addWidget(self._field_label("Source"), 0, 0)
        grid.addWidget(self._source_input, 0, 1)
        grid.addWidget(self._browse_source_button, 0, 2)

        grid.addWidget(self._field_label("Target"), 1, 0)
        grid.addWidget(self._target_input, 1, 1)
        grid.addWidget(self._browse_target_button, 1, 2)

        hint = QLabel("Accepted formats: CSV, XLSX (last column is the label) and MATLAB .mat")
        hint.setObjectName("captionLabel")
        grid.addWidget(hint, 2, 1, 1, 2)

        return group

    def _build_parameter_group(self) -> QGroupBox:
        group = QGroupBox("Method and Parameters")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        # Columns: 0 label | 1 field | 2 gutter | 3 label | 4 field.
        # Only the two field columns stretch, so both halves stay equal width.
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)
        grid.setColumnMinimumWidth(2, 28)

        # (label, widget, row, column-pair) laid out as a two-column form so the
        # six controls stay aligned instead of drifting in nested box layouts.
        self._method_label = self._field_label("Method")
        self._kernel_label = self._field_label("Kernel")
        self._classifier_label = self._field_label("Classifier")
        self._norm_label = self._field_label("Normalization")
        self._std_label = self._field_label("Standardization")
        self._reg_label = self._field_label("Regularization (λ)")
        self._dim_label = self._field_label("Latent Dimensions")

        rows = [
            (self._method_label, self._method_combo, self._classifier_label, self._classifier_combo),
            (self._kernel_label, self._kernel_combo, self._norm_label, self._norm_combo),
            (self._std_label, self._std_combo, self._reg_label, self._reg_input),
            (self._dim_label, self._dim_input, None, None),
        ]
        for row, (left_label, left_widget, right_label, right_widget) in enumerate(rows):
            grid.addWidget(left_label, row, 0)
            grid.addWidget(left_widget, row, 1)
            if right_label is not None:
                grid.addWidget(right_label, row, 3)
            if right_widget is not None:
                grid.addWidget(right_widget, row, 4)

        grid.setColumnMinimumWidth(0, 130)
        grid.setColumnMinimumWidth(3, 130)

        # Captured before any method may disable them, so the explanatory
        # tooltips can be restored when a method that uses them is selected.
        self._tunable_controls = {
            "kernel_type": (self._kernel_label, self._kernel_combo),
            "regularization": (self._reg_label, self._reg_input),
            "latent_dim": (self._dim_label, self._dim_input),
        }
        self._tunable_tooltips = {
            field: widget.toolTip() for field, (_, widget) in self._tunable_controls.items()
        }
        return group

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self._run_button)
        row.addWidget(self._export_button)
        row.addStretch(1)
        return row

    def _build_metrics_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        for card in self._metric_cards:
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(card)
        return row

    def _build_results_stack(self) -> QStackedWidget:
        self._results_stack = QStackedWidget()
        self._results_stack.addWidget(self._build_empty_state())
        self._results_stack.addWidget(self._build_results_tabs())
        return self._results_stack

    def _build_empty_state(self) -> QWidget:
        page = QFrame()
        page.setObjectName("metricCard")

        title = QLabel("No results yet")
        title.setObjectName("emptyStateTitle")
        title.setAlignment(Qt.AlignCenter)
        body = QLabel(
            "Choose a source and a target dataset, pick a method, then run the experiment."
        )
        body.setObjectName("emptyStateBody")
        body.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(6)
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addStretch(1)
        return page

    def _build_results_tabs(self) -> QTabWidget:
        self._results_tabs = QTabWidget()
        self._results_tabs.setDocumentMode(True)

        predictions_page = QWidget()
        predictions_layout = QVBoxLayout(predictions_page)
        predictions_layout.setContentsMargins(12, 10, 12, 12)
        predictions_layout.setSpacing(8)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self._incorrect_only_check)
        filter_row.addStretch(1)
        predictions_layout.addLayout(filter_row)
        predictions_layout.addWidget(self._predictions_table, 1)

        features_page = QWidget()
        features_layout = QVBoxLayout(features_page)
        features_layout.setContentsMargins(12, 10, 12, 12)
        features_layout.setSpacing(8)
        features_layout.addWidget(self._features_caption)
        features_layout.addWidget(self._features_table, 1)

        summary_page = QWidget()
        summary_layout = QVBoxLayout(summary_page)
        summary_layout.setContentsMargins(12, 10, 12, 12)
        summary_layout.addWidget(self._summary_table)
        summary_layout.addStretch(1)

        self._results_tabs.addTab(predictions_page, "Predictions")
        self._results_tabs.addTab(features_page, "Transformed Features")
        self._results_tabs.addTab(summary_page, "Dataset Summary")
        return self._results_tabs

    def _build_status_bar(self) -> None:
        self._status_dot = QLabel("●")
        self._status_label = QLabel()

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedWidth(150)
        self._progress_bar.hide()

        self._context_label = QLabel()

        bar = self.statusBar()
        bar.addWidget(self._status_dot)
        bar.addWidget(self._status_label)
        bar.addWidget(self._progress_bar)
        bar.addPermanentWidget(self._context_label)
        self._set_status("Ready", PALETTE["neutral"])

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _connect_signals(self) -> None:
        self._browse_source_button.clicked.connect(self._pick_source)
        self._browse_target_button.clicked.connect(self._pick_target)
        self._run_button.clicked.connect(self._run_transfer_learning)
        self._export_button.clicked.connect(self._export_pdf_report)

        self._open_source_action.triggered.connect(self._pick_source)
        self._open_target_action.triggered.connect(self._pick_target)
        self._export_action.triggered.connect(self._export_pdf_report)
        self._run_action.triggered.connect(self._run_transfer_learning)
        self._quit_action.triggered.connect(self.close)
        self._about_action.triggered.connect(self._show_about)

        self._method_combo.currentTextChanged.connect(self._sync_method_parameters)
        self._incorrect_only_check.toggled.connect(self._apply_prediction_filter)

    # ------------------------------------------------------------------
    # Configuration state
    # ------------------------------------------------------------------
    def _sync_method_parameters(self) -> None:
        """Grey out parameters the selected method does not consume."""
        method_name = self._method_combo.currentText()
        supported = self._use_case.supported_parameters(method_name)

        for field, (label, widget) in self._tunable_controls.items():
            enabled = field in supported
            label.setEnabled(enabled)
            widget.setEnabled(enabled)
            widget.setToolTip(
                self._tunable_tooltips[field]
                if enabled
                else f"{method_name} does not use this parameter."
            )

        self._update_context_label()

    def _update_context_label(self) -> None:
        method_name = self._method_combo.currentText()
        parts = [method_name]
        if self._kernel_combo.isEnabled():
            parts.append(f"{self._kernel_combo.currentText()} kernel")
        if self._last_result is not None:
            parts.append(f"{self._last_result.target_dataset.features.shape[0]} target samples")
        self._context_label.setText("  ·  ".join(parts))

    def _set_status(self, text: str, color: str) -> None:
        self._status_dot.setStyleSheet(f"color: {color};")
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pick_source(self) -> None:
        self._pick_file(self._source_input, "Select Source Dataset")

    def _pick_target(self) -> None:
        self._pick_file(self._target_input, "Select Target Dataset")

    def _pick_file(self, target_input: QLineEdit, title: str) -> None:
        start_dir = target_input.text().strip() or str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(self, title, start_dir, DATASET_FILTER)
        if file_path:
            target_input.setText(file_path)

    def _run_transfer_learning(self) -> None:
        # Guard against a stale reference: once the C++ QThread is deleted by
        # deleteLater the Python wrapper raises RuntimeError on any method call.
        try:
            if self._worker is not None and self._worker.isRunning():
                return
        except RuntimeError:
            # C++ object already deleted – treat it as "not running".
            self._worker = None

        source_path = self._source_input.text().strip()
        target_path = self._target_input.text().strip()

        if not source_path or not target_path:
            QMessageBox.warning(
                self,
                "Missing Dataset",
                "Select both a source and a target dataset before running.",
            )
            return

        missing = [p for p in (source_path, target_path) if not Path(p).is_file()]
        if missing:
            QMessageBox.warning(
                self,
                "Dataset Not Found",
                "The following path is not a readable file:\n\n" + "\n".join(missing),
            )
            return

        request = TransferLearningRequest(
            method_name=self._method_combo.currentText(),
            source_path=source_path,
            target_path=target_path,
            normalization=self._norm_combo.currentData(),
            standardization=self._std_combo.currentData(),
            kernel_type=self._kernel_combo.currentData(),
            regularization=self._reg_input.value(),
            latent_dim=self._dim_input.value(),
            classifier=self._classifier_combo.currentData(),
        )

        self._set_running(True)
        self._set_status("Running experiment…", PALETTE["warning"])

        self._worker = TransferLearningWorker(self._use_case, request)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(lambda: self._set_running(False))
        # Null out the Python reference *before* the C++ object is scheduled
        # for deletion, so the next call to _run_transfer_learning finds None
        # instead of a dangling wrapped object.
        self._worker.finished.connect(lambda: setattr(self, "_worker", None))
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._run_action.setEnabled(not running)
        self._run_button.setText("Running…" if running else "Run Experiment")
        self._progress_bar.setVisible(running)

    def _on_success(self, result: TransferLearningResult) -> None:
        self._last_result = result
        self._set_status("Completed", PALETTE["success"])
        self._export_button.setEnabled(True)
        self._export_action.setEnabled(True)

        self._show_metrics(result.metrics)
        self._show_dataset_summary(result)
        self._show_predictions(result)
        self._show_transformed_features(result)

        self._results_stack.setCurrentIndex(1)
        self._update_context_label()

    def _on_error(self, message: str) -> None:
        self._set_status("Failed", PALETTE["danger"])
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Experiment Failed")
        box.setText("The transfer learning run could not be completed.")
        box.setInformativeText(message)
        box.exec_()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Transfer Learning Studio",
            "<b>Transfer Learning Studio</b>"
            "<p>Desktop workbench for domain adaptation experiments.</p>"
            "<p>Methods: "
            + ", ".join(self._use_case.method_names)
            + ".<br>Classifiers: k-NN, SVM, Random Forest, Logistic Regression, Decision Tree.<br>"
            "The selected classifier is fitted on the transformed source domain and "
            "evaluated on the transformed target domain.</p>",
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _show_metrics(self, metrics: EvaluationMetrics | None) -> None:
        if metrics is None:
            for card in self._metric_cards:
                card.set_value(None)
            self._counts_label.setText(
                "Metrics unavailable — the source or target dataset has no labels."
            )
            return

        self._accuracy_card.set_value(f"{metrics.accuracy:.4f}")
        self._precision_card.set_value(f"{metrics.precision:.4f}")
        self._recall_card.set_value(f"{metrics.recall:.4f}")
        self._f1_card.set_value(f"{metrics.f1_score:.4f}")
        self._counts_label.setText(
            f"{metrics.total_samples} target samples evaluated  ·  "
            f"{metrics.correct_predictions} correct  ·  "
            f"{metrics.wrong_predictions} incorrect"
        )

    def _show_dataset_summary(self, result: TransferLearningResult) -> None:
        rows = [("Source", result.source_dataset), ("Target", result.target_dataset)]
        self._summary_table.setRowCount(len(rows))

        for row, (role, dataset) in enumerate(rows):
            feature_names = ", ".join(dataset.feature_names)
            values = [
                role,
                str(dataset.features.shape[0]),
                str(dataset.features.shape[1]),
                dataset.label_name,
                feature_names,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (1, 2):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if column == 4:
                    item.setToolTip(feature_names)
                self._summary_table.setItem(row, column, item)

        header = self._summary_table.horizontalHeader()
        for column in range(4):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

        # Only ever two rows, so size to content instead of leaving blank space.
        row_height = self._summary_table.verticalHeader().defaultSectionSize()
        self._summary_table.setFixedHeight(header.height() + len(rows) * row_height + 2)

    def _show_predictions(self, result: TransferLearningResult) -> None:
        table = self._predictions_table
        table.clearContents()

        if result.target_predictions is None:
            table.setRowCount(0)
            self._incorrect_only_check.setEnabled(False)
            self._results_tabs.setTabEnabled(0, False)
            self._results_tabs.setCurrentIndex(1)
            return

        self._results_tabs.setTabEnabled(0, True)
        self._incorrect_only_check.setEnabled(True)
        self._results_tabs.setCurrentIndex(0)

        correct_brush = QBrush(QColor(PALETTE["success"]))
        wrong_brush = QBrush(QColor(PALETTE["danger"]))

        row_count = len(result.target_predictions)
        table.setRowCount(row_count)

        for row in range(row_count):
            is_correct = result.prediction_correctness[row] == "Right"
            cells = [
                f"Sample {row + 1}",
                str(result.target_actuals[row]),
                str(result.target_predictions[row]),
                "✓ Correct" if is_correct else "✗ Incorrect",
            ]
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if column:
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 3:
                    item.setForeground(correct_brush if is_correct else wrong_brush)
                table.setItem(row, column, item)

        header = table.horizontalHeader()
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

        self._apply_prediction_filter()

    def _apply_prediction_filter(self) -> None:
        """Hide correctly classified rows when the filter is active."""
        result = self._last_result
        if result is None or result.prediction_correctness is None:
            return

        incorrect_only = self._incorrect_only_check.isChecked()
        for row, correctness in enumerate(result.prediction_correctness):
            self._predictions_table.setRowHidden(
                row, incorrect_only and correctness == "Right"
            )

    def _show_transformed_features(self, result: TransferLearningResult) -> None:
        matrix = result.transformed_target
        table = self._features_table
        table.clearContents()

        headers = ["Sample"] + list(result.output_feature_names)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        self._align_headers(table, [Qt.AlignLeft] + [Qt.AlignRight] * (len(headers) - 1))

        total_rows = matrix.shape[0]
        shown_rows = min(total_rows, FEATURE_PREVIEW_ROWS)
        table.setRowCount(shown_rows)

        for row in range(shown_rows):
            name_item = QTableWidgetItem(f"Sample {row + 1}")
            table.setItem(row, 0, name_item)
            for column in range(matrix.shape[1]):
                item = QTableWidgetItem(f"{matrix[row, column]:.6f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row, column + 1, item)

        if shown_rows < total_rows:
            self._features_caption.setText(
                f"Showing the first {shown_rows} of {total_rows} transformed target rows."
            )
        else:
            self._features_caption.setText(
                f"{total_rows} transformed target rows × {matrix.shape[1]} components."
            )
        self._features_caption.show()

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

    # ------------------------------------------------------------------
    # Report export
    # ------------------------------------------------------------------
    def _export_pdf_report(self) -> None:
        result = self._last_result
        if result is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF Report",
            str(Path.cwd() / "Transfer_Learning_Report.pdf"),
            "PDF Files (*.pdf)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".pdf"):
            file_path += ".pdf"

        try:
            document = QTextDocument()
            document.setHtml(self._build_report_html(result))

            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(file_path)
            printer.setPageMargins(15, 15, 15, 15, QPrinter.Millimeter)
            document.print_(printer)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            QMessageBox.critical(
                self, "Export Failed", f"The report could not be written:\n\n{exc}"
            )
            return

        self._set_status("Report exported", PALETTE["success"])
        QMessageBox.information(
            self, "Report Exported", f"The PDF report was saved to:\n\n{file_path}"
        )

    def _build_report_html(self, result: TransferLearningResult) -> str:
        metrics = result.metrics
        generated = datetime.now().strftime("%d %B %Y, %H:%M")

        config_rows = [
            ("Method", self._method_combo.currentText()),
            ("Classifier", self._classifier_combo.currentText()),
            ("Source dataset", Path(self._source_input.text().strip()).name),
            ("Target dataset", Path(self._target_input.text().strip()).name),
            ("Normalization", self._norm_combo.currentText()),
            ("Standardization", self._std_combo.currentText()),
        ]
        if self._kernel_combo.isEnabled():
            config_rows.append(("Kernel", self._kernel_combo.currentText()))
        if self._reg_input.isEnabled():
            config_rows.append(("Regularization (lambda)", f"{self._reg_input.value():g}"))
        if self._dim_input.isEnabled():
            config_rows.append(("Latent dimensions", str(self._dim_input.value())))

        parts = [
            "<html><head><meta charset='utf-8'><style>",
            "body { font-family: 'Helvetica', Arial, sans-serif; color: #1a2229; font-size: 10pt; }",
            "h1 { font-size: 16pt; margin: 0 0 2px 0; }",
            "h2 { font-size: 12pt; margin: 22px 0 6px 0; border-bottom: 1px solid #c3ccd6;"
            " padding-bottom: 3px; }",
            "p.meta { color: #5b6b7c; font-size: 9pt; margin: 0 0 4px 0; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #dde2e8; padding: 5px 7px; text-align: left; }",
            "th { background-color: #eef1f4; font-size: 9pt; }",
            "td.num { text-align: right; }",
            "</style></head><body>",
            "<h1>Transfer Learning Evaluation Report</h1>",
            f"<p class='meta'>Generated {escape(generated)}</p>",
            "<h2>Configuration</h2><table width='100%'>",
        ]
        for label, value in config_rows:
            parts.append(f"<tr><th style='width:35%'>{escape(label)}</th>"
                         f"<td>{escape(str(value))}</td></tr>")
        parts.append("</table>")

        parts.append("<h2>Dataset Summary</h2><table width='100%'>")
        parts.append("<tr><th>Dataset</th><th>Samples</th><th>Features</th>"
                     "<th>Label Column</th></tr>")
        for role, dataset in (("Source", result.source_dataset), ("Target", result.target_dataset)):
            parts.append(
                f"<tr><td>{role}</td><td class='num'>{dataset.features.shape[0]}</td>"
                f"<td class='num'>{dataset.features.shape[1]}</td>"
                f"<td>{escape(str(dataset.label_name))}</td></tr>"
            )
        parts.append("</table>")

        if metrics is not None:
            parts.append("<h2>Evaluation Metrics</h2><table width='100%'>")
            parts.append("<tr><th>Accuracy</th><th>Precision</th><th>Recall</th>"
                         "<th>F1 Score</th></tr>")
            parts.append(
                f"<tr><td class='num'>{metrics.accuracy:.4f}</td>"
                f"<td class='num'>{metrics.precision:.4f}</td>"
                f"<td class='num'>{metrics.recall:.4f}</td>"
                f"<td class='num'>{metrics.f1_score:.4f}</td></tr></table>"
            )
            parts.append("<table width='100%' style='margin-top:8px'>")
            parts.append("<tr><th>Total Samples</th><th>Correct</th><th>Incorrect</th></tr>")
            parts.append(
                f"<tr><td class='num'>{metrics.total_samples}</td>"
                f"<td class='num'>{metrics.correct_predictions}</td>"
                f"<td class='num'>{metrics.wrong_predictions}</td></tr></table>"
            )

        if result.target_predictions is not None:
            parts.append("<h2>Target Predictions</h2><table width='100%'>")
            parts.append("<tr><th>Sample</th><th>Actual Label</th><th>Predicted Label</th>"
                         "<th>Result</th></tr>")
            for row in range(len(result.target_predictions)):
                is_correct = result.prediction_correctness[row] == "Right"
                colour = PALETTE["success"] if is_correct else PALETTE["danger"]
                verdict = "Correct" if is_correct else "Incorrect"
                parts.append(
                    f"<tr><td>Sample {row + 1}</td>"
                    f"<td>{escape(str(result.target_actuals[row]))}</td>"
                    f"<td>{escape(str(result.target_predictions[row]))}</td>"
                    f"<td style='color:{colour}; font-weight:bold;'>{verdict}</td></tr>"
                )
            parts.append("</table>")

        parts.append("</body></html>")
        return "".join(parts)

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(3000)
        super().closeEvent(event)
