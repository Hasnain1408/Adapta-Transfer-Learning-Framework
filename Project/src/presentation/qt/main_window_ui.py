"""Widget, layout, and menu construction for ``MainWindow``.

Every ``_build_*`` method here only builds widgets and wires their signals —
no business logic, no persistence, no run orchestration. This was the single
largest chunk of what used to be a ~2000-line ``main_window.py``, split out
because "build every widget" is one coherent job on its own.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from domain.entities import UserRole
from presentation.qt.charts_widget import AnalyticsChartsWidget
from presentation.qt.comparison_tab import ComparisonTab
from presentation.qt.visualization_tab import VisualizationTab
from presentation.qt.dataset_summary_view import DatasetSummaryView
from presentation.qt.history_tab import HistoryTab
from presentation.qt.metric_card import MetricCard
from presentation.qt.processing_status import ProcessingStatusStrip
from presentation.qt.result_tables import BATCH_RESULTS_HEADERS
from presentation.qt.title_bar import TitleBar

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


class _ClickableLineEdit(QLineEdit):
    """Read-only line edit that emits ``clicked`` on any mouse press.

    Used for the dataset path fields so clicking anywhere in the field opens
    the file browser dialog, not just the adjacent "Browse…" button.
    """

    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.clicked.emit()
        super().mousePressEvent(event)


class FullScreenViewer(QMainWindow):
    """Generic standalone window that borrows the results tab widget.

    Reparents the actual tab widget in place — it keeps its live state
    (chart figure, populated table rows, filter checkboxes) and is handed
    back to its original layout slot when the window closes, so no data has
    to be duplicated or re-synced.
    """

    def __init__(
        self,
        title: str,
        content: QWidget,
        stylesheet: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{title} — Full Screen")
        self.setMinimumSize(800, 540)
        self.setStyleSheet(stylesheet)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        title_bar = TitleBar(self, f"{title} — Full Screen")

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(18, 14, 18, 14)
        body_layout.setSpacing(10)
        body_layout.addWidget(content, 1)

        container = QWidget()
        container.setObjectName("centralWidget")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(title_bar)
        outer.addWidget(body, 1)
        self.setCentralWidget(container)

        self.showMaximized()


class MainWindowUIMixin:
    """Widget/layout/menu construction and signal wiring for ``MainWindow``.

    Composed into ``MainWindow`` alongside the other ``MainWindow*Mixin``
    classes; every method below assumes the attributes ``MainWindow.__init__``
    sets before calling into construction (``self._use_case``, ``self._catalog``,
    ``self._palette``, ``self._current_user``, …).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        self._source_input = _ClickableLineEdit()
        self._source_input.setPlaceholderText("No source dataset selected")
        self._source_input.setClearButtonEnabled(True)
        self._source_input.setReadOnly(True)
        self._source_input.setCursor(Qt.PointingHandCursor)
        self._target_input = _ClickableLineEdit()
        self._target_input.setPlaceholderText("No target dataset selected")
        self._target_input.setClearButtonEnabled(True)
        self._target_input.setReadOnly(True)
        self._target_input.setCursor(Qt.PointingHandCursor)

        self._swap_button = QPushButton("⇅ Swap")
        self._swap_button.setObjectName("swapButton")
        self._swap_button.setToolTip("Swap Source and Target dataset paths (Ctrl+Shift+S)")
        self._swap_button.setCursor(Qt.PointingHandCursor)

        self._combined_file_check = QCheckBox("Source and target are combined in one file")
        self._combined_file_check.setToolTip(
            "For datasets like COIL20 or MNIST↔USPS, where a single .mat file "
            "already carries both halves (X_src/Y_src alongside X_tar/Y_tar). "
            "Pick just that one file; the source and target sections are "
            "detected and split automatically."
        )

        self._batch_mode_check = QCheckBox("Batch mode — run every ordered pair from multiple files")
        self._batch_mode_check.setToolTip(
            "Pick 2+ files; every ordered pair (A→B, A→C, B→A, ...) is run with "
            "the same method and configuration."
        )

        self._batch_files_list = QListWidget()
        self._batch_files_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._batch_files_list.setToolTip("Files included in the batch")
        self._batch_files_list.setMaximumHeight(110)

        self._add_batch_files_button = QPushButton("Add Files…")
        self._remove_batch_file_button = QPushButton("Remove Selected")
        self._clear_batch_files_button = QPushButton("Clear All")
        for button in (
            self._add_batch_files_button,
            self._remove_batch_file_button,
            self._clear_batch_files_button,
        ):
            button.setCursor(Qt.PointingHandCursor)

        self._batch_files_caption = QLabel("Add 2+ files to enable batch mode.")
        self._batch_files_caption.setObjectName("captionLabel")

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

        self._mu_input = QDoubleSpinBox()
        self._mu_input.setDecimals(4)
        self._mu_input.setRange(0.0, 1_000_000.0)
        self._mu_input.setSingleStep(0.1)
        self._mu_input.setValue(1.0)
        self._mu_input.setToolTip(
            "Balance between marginal and conditional distribution alignment "
            "(BDA: 0 = marginal only, 1 = conditional only)"
        )

        for spin_box in (self._reg_input, self._dim_input, self._mu_input):
            spin_box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._run_button = QPushButton("Run Experiment")
        self._run_button.setObjectName("runButton")
        self._run_button.setCursor(Qt.PointingHandCursor)
        self._run_button.setDefault(True)

        self._pause_button = QPushButton("Pause")
        self._pause_button.setObjectName("pauseButton")
        self._pause_button.setCursor(Qt.PointingHandCursor)
        self._pause_button.setEnabled(False)
        self._pause_button.setToolTip(
            "Pauses before the next iteration (or, in batch mode, the next "
            "pair) -- a fast one-shot method may finish before this takes effect."
        )

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setObjectName("dangerButton")
        self._cancel_button.setCursor(Qt.PointingHandCursor)
        self._cancel_button.setEnabled(False)
        self._cancel_button.setToolTip(
            "Stops after the current iteration or pair. A single-pair run has "
            "no partial result to show; a batch keeps whatever pairs already finished."
        )

        self._export_button = QPushButton("Export Report…")
        self._export_button.setObjectName("runButton")
        self._export_button.setCursor(Qt.PointingHandCursor)
        self._export_button.setEnabled(False)

        self._clear_button = QPushButton("Clear")
        self._clear_button.setCursor(Qt.PointingHandCursor)
        self._clear_button.setToolTip(
            "Reset every configuration field and discard the displayed results, "
            "ready for a fresh experiment. Saved experiment history is untouched."
        )

        # Sits apart from Cancel (which stops a *run*) so the two reds don't
        # compete: this one is styled neutral until hovered, because quitting
        # is a deliberate act rather than an escape hatch from a bad run.
        self._exit_button = QPushButton("⏻  Exit System")
        self._exit_button.setObjectName("exitButton")
        self._exit_button.setCursor(Qt.PointingHandCursor)
        self._exit_button.setToolTip("Close Adapta and quit the application (Ctrl+Q)")

        self._theme_toggle_button = QPushButton("☀  Light")
        self._theme_toggle_button.setObjectName("themeToggleButton")
        self._theme_toggle_button.setCursor(Qt.PointingHandCursor)
        self._theme_toggle_button.setToolTip("Switch between dark and light mode")

        self._accuracy_card = MetricCard(
            "Accuracy", "Share of target samples predicted correctly", accent="teal"
        )
        self._precision_card = MetricCard(
            "Precision", "True positives over predicted positives", accent="purple"
        )
        self._recall_card = MetricCard(
            "Recall", "True positives over actual positives", accent="amber"
        )
        self._f1_card = MetricCard(
            "F1 Score", "Harmonic mean of precision and recall", accent="pink"
        )
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
        self._correct_only_check = QCheckBox("Show correct predictions only")
        self._correct_only_check.setEnabled(False)

        self._results_fullscreen_button = self._build_fullscreen_button(
            "Open the results tabs in a dedicated full-screen window"
        )

        self._predictions_table = self._build_table(
            ["Sample", "Actual Label", "Predicted Label", "Result"]
        )
        self._align_headers(
            self._predictions_table,
            [Qt.AlignLeft, Qt.AlignCenter, Qt.AlignCenter, Qt.AlignCenter],
        )
        self._features_table = self._build_table([])
        self._summary_view = DatasetSummaryView(self._palette)

        self._features_caption = QLabel()
        self._features_caption.setObjectName("captionLabel")
        self._features_caption.hide()

        # Built from the shared constant rather than a second copy of the
        # column list — populate_batch_results_table fills by position, so a
        # divergence here would silently shift every value one column over.
        self._batch_results_table = self._build_table(list(BATCH_RESULTS_HEADERS))
        self._align_headers(
            self._batch_results_table,
            [Qt.AlignLeft, Qt.AlignLeft, Qt.AlignCenter]
            + [Qt.AlignRight] * (len(BATCH_RESULTS_HEADERS) - 3),
        )
        self._batch_summary_label = QLabel(
            "Run a batch to see per-pair results and the average across the batch."
        )
        self._batch_summary_label.setObjectName("captionLabel")
        self._batch_summary_label.setWordWrap(True)

        # Lives on the Batch Results tab itself (not just the sidebar) so it
        # stays reachable when that tab is viewed in the full-screen results
        # window, which never carries the sidebar with it.
        self._batch_export_button = QPushButton("Export Report…")
        self._batch_export_button.setCursor(Qt.PointingHandCursor)
        self._batch_export_button.setEnabled(False)
        self._batch_export_button.setToolTip(
            "Export a PDF report for the whole batch — every pair plus the average"
        )

        # Owner of any experiment recorded from now on — the account that
        # authenticated through AuthDialog before this window was built.
        self._user_label = QLabel()
        self._user_label.setObjectName("captionLabel")
        self._update_user_label()

        self._logout_button = QPushButton("Log Out")
        self._logout_button.setObjectName("linkButton")
        self._logout_button.setCursor(Qt.PointingHandCursor)
        self._logout_button.setToolTip("Log in to record and review experiments")
        self._logout_button.setEnabled(True)

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
        self._open_source_action.setShortcut(QKeySequence("Ctrl+S"))
        self._open_target_action = QAction("Open &Target Dataset…", self)
        self._open_target_action.setShortcut(QKeySequence("Ctrl+T"))
        self._swap_datasets_action = QAction("S&wap Source / Target", self)
        self._swap_datasets_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._export_action = QAction("&Export Report…", self)
        self._export_action.setShortcut(QKeySequence("Ctrl+R"))
        self._export_action.setEnabled(False)
        self._quit_action = QAction("E&xit System", self)
        self._quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self._run_action = QAction("&Run Experiment", self)
        self._run_action.setShortcut(QKeySequence("Ctrl+E"))
        self._about_action = QAction("&About", self)

        self._learning_materials_action = QAction("&Learning Materials…", self)
        self._learning_materials_action.setShortcut(QKeySequence("Ctrl+L"))
        self._tl_methods_action = QAction("Transfer Learning &Methods…", self)
        self._classifiers_action = QAction("&Classifiers…", self)
        self._metrics_action = QAction("Evaluation &Metrics…", self)
        self._preprocessing_action = QAction("&Normalization & Scaling…", self)

        # Database actions stay disabled without a connection rather than
        # opening a dialog that can only show an error.
        has_database = self._catalog is not None
        is_admin = self._current_user is not None and self._current_user.role == UserRole.ADMIN
        self._manage_users_action = QAction("Manage &Users…", self)
        self._manage_users_action.setShortcut(QKeySequence("Ctrl+U"))
        self._manage_users_action.setEnabled(has_database and is_admin)
        self._dataset_registry_action = QAction("&Dataset Registry…", self)
        self._dataset_registry_action.setShortcut(QKeySequence("Ctrl+D"))
        self._dataset_registry_action.setEnabled(has_database)
        self._refresh_history_action = QAction("Refresh &History", self)
        self._refresh_history_action.setShortcut(QKeySequence("F5"))
        self._refresh_history_action.setEnabled(has_database)
        self._database_info_action = QAction("Connection &Info", self)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._open_source_action)
        file_menu.addAction(self._open_target_action)
        file_menu.addAction(self._swap_datasets_action)
        file_menu.addSeparator()
        file_menu.addAction(self._export_action)
        file_menu.addSeparator()
        file_menu.addAction(self._quit_action)

        run_menu = self.menuBar().addMenu("&Run")
        run_menu.addAction(self._run_action)

        database_menu = self.menuBar().addMenu("&Database")
        database_menu.addAction(self._manage_users_action)
        database_menu.addAction(self._dataset_registry_action)
        database_menu.addSeparator()
        database_menu.addAction(self._refresh_history_action)
        database_menu.addAction(self._database_info_action)

        learning_menu = self.menuBar().addMenu("&Learning Materials")
        learning_menu.addAction(self._learning_materials_action)
        learning_menu.addSeparator()
        learning_menu.addAction(self._tl_methods_action)
        learning_menu.addAction(self._classifiers_action)
        learning_menu.addAction(self._metrics_action)
        learning_menu.addAction(self._preprocessing_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self._learning_materials_action)
        help_menu.addSeparator()
        help_menu.addAction(self._about_action)

    def _build_title_bar(self) -> None:
        """Stack a custom title bar above the real menu bar.

        The window is frameless, so ``TitleBar`` stands in for the OS
        decoration. ``QMainWindow.setMenuWidget`` accepts any widget as the
        menu-bar area, so the existing ``self.menuBar()`` is nested inside a
        small container alongside it rather than rebuilt.
        """
        self._title_bar = TitleBar(self, "Adapta: Transfer Learning Studio")

        container = QWidget()
        container.setObjectName("titleBarContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._title_bar)
        layout.addWidget(self.menuBar())
        self.setMenuWidget(container)

    def _build_layout(self) -> None:
        central = QWidget(self)
        central.setObjectName("centralWidget")
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 10)
        root.setSpacing(11)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_content(), 1)
        root.addLayout(body, 1)

        self.setCentralWidget(central)

    def _build_header(self) -> QWidget:
        """Small branded header: an accent mark, a title/subtitle pair, and the theme toggle."""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(12)

        mark = QLabel()
        mark.setObjectName("brandMark")
        mark.setFixedSize(34, 34)

        layout.addStretch(1)
        layout.addWidget(self._user_label)
        layout.addWidget(self._logout_button)
        layout.addWidget(self._theme_toggle_button)
        return header

    def _build_sidebar(self) -> QWidget:
        """Left-hand column: method/parameters and run actions."""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(320)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(self._build_parameter_group())
        layout.addLayout(self._build_action_row())
        layout.addStretch(1)
        return sidebar

    def _build_content(self) -> QWidget:
        """Right-hand column: datasets on top, metric cards below, then results."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(11)

        layout.addWidget(self._build_dataset_group())
        layout.addLayout(self._build_metrics_row())
        layout.addWidget(self._counts_label)
        # Hidden until a run starts, so an idle window looks the way it
        # always did and the strip's appearance is itself a signal. Placed
        # directly above the results it is about, and in the content column
        # rather than the sidebar because the sidebar has no spare height.
        self._processing_strip = ProcessingStatusStrip()
        layout.addWidget(self._processing_strip)
        layout.addWidget(self._build_results_stack(), 1)
        return content

    def _build_dataset_group(self) -> QGroupBox:
        group = QGroupBox("Datasets")
        outer = QVBoxLayout(group)
        outer.setSpacing(10)

        outer.addWidget(self._batch_mode_check)

        self._single_dataset_widget = self._build_single_dataset_widget()
        self._batch_dataset_widget = self._build_batch_dataset_widget()
        outer.addWidget(self._single_dataset_widget)
        outer.addWidget(self._batch_dataset_widget)
        self._batch_dataset_widget.setVisible(False)

        hint = QLabel("Accepted formats: CSV, XLSX (last column is the label) and MATLAB .mat")
        hint.setObjectName("captionLabel")
        outer.addWidget(hint)

        return group

    def _build_single_dataset_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._combined_file_check)

        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        self._source_field_label = self._field_label("Source")
        grid.addWidget(self._source_field_label, 0, 0)
        grid.addWidget(self._source_input, 0, 1)

        self._target_field_label = self._field_label("Target")
        grid.addWidget(self._target_field_label, 1, 0)
        grid.addWidget(self._target_input, 1, 1)

        grid.addWidget(self._swap_button, 0, 2, 2, 1, Qt.AlignVCenter)
        layout.addWidget(grid_widget)
        return widget

    def _build_batch_dataset_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        button_row = QHBoxLayout()
        button_row.addWidget(self._add_batch_files_button)
        button_row.addWidget(self._remove_batch_file_button)
        button_row.addWidget(self._clear_batch_files_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        layout.addWidget(self._batch_files_list)
        layout.addWidget(self._batch_files_caption)
        return widget

    def _build_parameter_group(self) -> QGroupBox:
        group = QGroupBox("Method and Parameters")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Single-column form: a field label above its control, stacked in the
        # order they matter most — the method choice first, tunables last.
        self._method_label = self._field_label("Method")
        self._classifier_label = self._field_label("Classifier")
        self._kernel_label = self._field_label("Kernel")
        self._norm_label = self._field_label("Normalization")
        self._std_label = self._field_label("Standardization")
        self._reg_label = self._field_label("Regularization (λ)")
        self._dim_label = self._field_label("Latent Dimensions")
        self._mu_label = self._field_label("Mu (μ)")

        rows = [
            (self._method_label, self._method_combo),
            (self._classifier_label, self._classifier_combo),
            (self._kernel_label, self._kernel_combo),
            (self._norm_label, self._norm_combo),
            (self._std_label, self._std_combo),
            (self._reg_label, self._reg_input),
            (self._dim_label, self._dim_input),
            (self._mu_label, self._mu_input),
        ]
        for label, widget in rows:
            layout.addWidget(label)
            layout.addWidget(widget)

        # Captured before any method may disable them, so the explanatory
        # tooltips can be restored when a method that uses them is selected.
        self._tunable_controls = {
            "kernel_type": (self._kernel_label, self._kernel_combo),
            "regularization": (self._reg_label, self._reg_input),
            "latent_dim": (self._dim_label, self._dim_input),
            "mu": (self._mu_label, self._mu_input),
        }
        self._tunable_tooltips = {
            field: widget.toolTip() for field, (_, widget) in self._tunable_controls.items()
        }
        return group

    def _build_action_row(self) -> QVBoxLayout:
        row = QVBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._run_button)

        pause_cancel_row = QHBoxLayout()
        pause_cancel_row.setSpacing(8)
        pause_cancel_row.addWidget(self._pause_button)
        pause_cancel_row.addWidget(self._cancel_button)
        row.addLayout(pause_cancel_row)

        row.addWidget(self._export_button)
        row.addWidget(self._clear_button)
        # Last, and spaced away from the rest, so it is never the button the
        # hand lands on when reaching for Clear.
        row.addSpacing(6)
        row.addWidget(self._exit_button)
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

        # The results tabs — History among them — are the *other* page of the
        # stack, so on a fresh launch past experiments would be unreachable
        # until a run finished. This is the way in.
        if self._catalog is not None:
            self._browse_history_button = QPushButton("Browse Past Experiments")
            self._browse_history_button.setCursor(Qt.PointingHandCursor)
            self._browse_history_button.clicked.connect(self._show_history)

            button_row = QHBoxLayout()
            button_row.addStretch(1)
            button_row.addWidget(self._browse_history_button)
            button_row.addStretch(1)
            layout.addSpacing(10)
            layout.addLayout(button_row)

        layout.addStretch(1)
        return page

    def _build_results_tabs(self) -> QTabWidget:
        self._results_tabs = QTabWidget()
        self._results_tabs.setDocumentMode(True)
        # Closable so a per-experiment detail tab (opened from History) can be
        # dismissed — the fixed tabs below have their close button hidden
        # individually right after being added, so only those stay closable.
        self._results_tabs.setTabsClosable(True)
        self._results_tabs.tabCloseRequested.connect(self._close_results_tab)
        # Comparison and History are populated from the database, not from a
        # run, so Full Screen has to follow the active tab rather than
        # staying disabled until an experiment finishes.
        self._results_tabs.currentChanged.connect(self._sync_fullscreen_button)

        predictions_page = QWidget()
        predictions_layout = QVBoxLayout(predictions_page)
        predictions_layout.setContentsMargins(12, 10, 12, 12)
        predictions_layout.setSpacing(8)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self._incorrect_only_check)
        filter_row.addWidget(self._correct_only_check)
        filter_row.addStretch(1)
        predictions_layout.addLayout(filter_row)
        predictions_layout.addWidget(self._predictions_table, 1)

        features_page = QWidget()
        features_layout = QVBoxLayout(features_page)
        features_layout.setContentsMargins(12, 10, 12, 12)
        features_layout.setSpacing(8)
        features_layout.addWidget(self._features_caption)
        features_layout.addWidget(self._features_table, 1)

        batch_page = QWidget()
        batch_layout = QVBoxLayout(batch_page)
        batch_layout.setContentsMargins(12, 10, 12, 12)
        batch_layout.setSpacing(8)
        batch_header = QHBoxLayout()
        batch_header.addWidget(self._batch_summary_label, 1)
        batch_header.addWidget(self._batch_export_button, 0, Qt.AlignTop)
        batch_layout.addLayout(batch_header)
        batch_layout.addWidget(self._batch_results_table, 1)

        self._charts_widget = AnalyticsChartsWidget(self._palette)
        self._visualization_tab = VisualizationTab(self._palette)
        self._comparison_tab = ComparisonTab(self._catalog, self._palette)
        self._history_tab = HistoryTab(self._catalog, self._palette)

        self._results_tabs.addTab(predictions_page, "Predictions")
        self._results_tabs.addTab(self._charts_widget, "Visualizations")
        self._results_tabs.addTab(self._visualization_tab, "Feature Space")
        self._results_tabs.addTab(features_page, "Transformed Features")
        self._results_tabs.addTab(self._summary_view, "Dataset Summary")
        self._results_tabs.addTab(batch_page, "Batch Results")
        self._results_tabs.addTab(self._comparison_tab, "Comparison")
        self._results_tabs.addTab(self._history_tab, "History")
        self._batch_results_tab_index = self._results_tabs.indexOf(batch_page)
        self._history_tab_index = self._results_tabs.indexOf(self._history_tab)

        # These eight are permanent — only dynamically opened experiment-detail
        # tabs (see _open_experiment_detail_tab) are meant to be closable.
        for index in range(self._results_tabs.count()):
            self._results_tabs.tabBar().setTabButton(index, QTabBar.RightSide, None)
        self._set_history_visible(self._current_user is not None)

        self._results_tabs.setCornerWidget(self._results_fullscreen_button, Qt.TopRightCorner)
        return self._results_tabs

    def _build_status_bar(self) -> None:
        self._status_dot = QLabel("●")
        self._status_label = QLabel()

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("runProgressBar")
        self._progress_bar.setRange(0, 0)  # indeterminate (animated) while running
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedWidth(260)
        self._progress_bar.hide()

        # Steps the chunk through cycleColor 0-3 once a second while running;
        # started/stopped alongside the run itself, see _set_running and
        # _toggle_pause in MainWindowRunMixin.
        self._progress_color_timer = QTimer(self)
        self._progress_color_timer.setInterval(1000)
        self._progress_color_timer.timeout.connect(self._advance_progress_bar_color)

        self._context_label = QLabel()

        bar = self.statusBar()
        bar.addWidget(self._status_dot)
        bar.addWidget(self._status_label)
        bar.addWidget(self._progress_bar)
        bar.addPermanentWidget(self._context_label)
        self._set_status("Ready", "neutral")

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _build_fullscreen_button(self, tooltip: str) -> QPushButton:
        button = QPushButton("⛶  Full Screen")
        button.setObjectName("fullscreenButton")
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(tooltip)
        button.setEnabled(False)
        return button

    def _connect_signals(self) -> None:
        self._source_input.clicked.connect(self._pick_source)
        self._target_input.clicked.connect(self._pick_target)
        self._swap_button.clicked.connect(self._swap_datasets)
        self._combined_file_check.toggled.connect(self._on_combined_file_toggled)
        self._batch_mode_check.toggled.connect(self._on_batch_mode_toggled)
        self._add_batch_files_button.clicked.connect(self._add_batch_files)
        self._remove_batch_file_button.clicked.connect(self._remove_selected_batch_files)
        self._clear_batch_files_button.clicked.connect(self._clear_batch_files)
        self._run_button.clicked.connect(self._run_transfer_learning)
        self._pause_button.clicked.connect(self._toggle_pause)
        self._cancel_button.clicked.connect(self._cancel_run)
        self._export_button.clicked.connect(self._export_pdf_report)
        self._batch_export_button.clicked.connect(self._export_batch_report)
        self._clear_button.clicked.connect(self._clear_all)
        self._exit_button.clicked.connect(self._exit_application)

        self._open_source_action.triggered.connect(self._pick_source)
        self._open_target_action.triggered.connect(self._pick_target)
        self._swap_datasets_action.triggered.connect(self._swap_datasets)
        self._export_action.triggered.connect(self._export_pdf_report)
        self._run_action.triggered.connect(self._run_transfer_learning)
        # Same path as the sidebar's Exit System button, so Ctrl+Q confirms
        # and shuts down identically instead of merely hiding the window.
        self._quit_action.triggered.connect(self._exit_application)
        self._about_action.triggered.connect(self._show_about)

        self._learning_materials_action.triggered.connect(lambda: self._open_learning_materials(0))
        self._tl_methods_action.triggered.connect(lambda: self._open_learning_materials(0))
        self._classifiers_action.triggered.connect(lambda: self._open_learning_materials(1))
        self._metrics_action.triggered.connect(lambda: self._open_learning_materials(2))
        self._preprocessing_action.triggered.connect(lambda: self._open_learning_materials(3))

        self._method_combo.currentTextChanged.connect(self._sync_method_parameters)
        self._incorrect_only_check.toggled.connect(self._on_prediction_filter_toggled)
        self._correct_only_check.toggled.connect(self._on_prediction_filter_toggled)
        self._results_fullscreen_button.clicked.connect(self._open_results_fullscreen)
        self._theme_toggle_button.clicked.connect(self._toggle_theme)

        self._logout_button.clicked.connect(self._handle_auth_button_clicked)
        self._manage_users_action.triggered.connect(self._open_user_manager)
        self._dataset_registry_action.triggered.connect(self._open_dataset_registry)
        self._refresh_history_action.triggered.connect(self._show_history)
        self._database_info_action.triggered.connect(self._show_database_info)
        self._history_tab.configuration_requested.connect(self._apply_configuration)
        self._history_tab.detail_requested.connect(self._open_experiment_detail_tab)
        self._history_tab.export_requested.connect(self._export_history_entry)
        self._batch_results_table.itemSelectionChanged.connect(self._open_pair_detail_tab)


__all__ = [
    "CLASSIFIER_OPTIONS",
    "DATASET_FILTER",
    "FEATURE_PREVIEW_ROWS",
    "KERNEL_OPTIONS",
    "NORMALIZATION_OPTIONS",
    "STANDARDIZATION_OPTIONS",
    "FullScreenViewer",
    "MainWindowUIMixin",
    "_ClickableLineEdit",
]
