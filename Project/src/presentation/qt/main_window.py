from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QIcon, QKeySequence, QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from application.authentication_use_case import AuthenticationUseCase
from application.experiment_catalog_use_case import ExperimentCatalogUseCase
from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.entities import ExperimentHistoryEntry, ExperimentRecord, User, UserRole
from domain.models import (
    BatchTransferLearningRequest,
    BatchTransferLearningResult,
    EvaluationMetrics,
    TransferLearningRequest,
    TransferLearningResult,
)
from presentation.qt.auth_dialogs import AuthDialog
from presentation.qt.charts_widget import AnalyticsChartsWidget
from presentation.qt.database_dialogs import DatasetRegistryDialog, UserManagerDialog
from presentation.qt.dataset_summary_view import DatasetSummaryView
from presentation.qt.experiment_detail_view import ExperimentDetailView
from presentation.qt.history_tab import HistoryTab
from presentation.qt.metric_card import MetricCard
from presentation.qt import message_box
from presentation.qt.pair_detail_view import PairDetailView
from presentation.qt.result_tables import populate_predictions_table
from presentation.qt.style import DARK_PALETTE, LIGHT_PALETTE, build_style_sheet
from presentation.qt.title_bar import TitleBar, install_title_bar_on_grid
from presentation.qt.worker import BatchTransferLearningWorker, TransferLearningWorker

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


class MainWindow(QMainWindow):
    def __init__(
        self,
        use_case: RunTransferLearningUseCase,
        catalog: ExperimentCatalogUseCase | None = None,
        auth: AuthenticationUseCase | None = None,
        current_user: User | None = None,
        database_error: str | None = None,
    ) -> None:
        """*catalog* is ``None`` when the database could not be reached.

        The app deliberately stays usable in that state — experiments still
        run, they just are not recorded — so a Neon outage or a missing
        ``DATABASE_URL`` degrades the feature rather than blocking start-up.
        ``database_error`` carries the reason, shown once in the status bar.

        *current_user* is the account currently signed in when the window is
        created. It is usually ``None`` because sign-in is now optional and is
        only prompted from the header button when the user asks for it.
        """
        super().__init__()
        self._use_case = use_case
        self._catalog = catalog
        self._auth = auth
        self._database_error = database_error
        self._current_user = current_user
        self._worker: TransferLearningWorker | BatchTransferLearningWorker | None = None
        self._last_result: TransferLearningResult | None = None
        self._last_request: TransferLearningRequest | None = None
        self._last_batch_request: BatchTransferLearningRequest | None = None
        self._last_batch_result: BatchTransferLearningResult | None = None
        self._batch_results_tab_index = -1
        self._theme = "dark"
        self._palette = DARK_PALETTE
        self._status_key = "neutral"
        self._history_tab_index = -1
        #: experiment id -> its open detail tab widget, so a second click on
        #: the same row raises the existing tab instead of duplicating it.
        self._experiment_detail_tabs: dict[int, ExperimentDetailView] = {}
        #: id(pair) -> its open detail tab widget, same idea as above but for
        #: batch pairs, which have no persisted id to key on.
        self._pair_detail_tabs: dict[int, PairDetailView] = {}

        self.setWindowTitle("Adapta: Transfer Learning Studio")
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "app_icon.svg")))
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setMinimumSize(1000, 720)
        self.resize(1180, 820)
        self._apply_theme_stylesheet()

        self._build_widgets()
        self._build_menu()
        self._build_title_bar()
        self._build_layout()
        self._build_status_bar()
        self._connect_signals()
        self._sync_method_parameters()
        self._update_auth_ui()

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

        for spin_box in (self._reg_input, self._dim_input):
            spin_box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._run_button = QPushButton("Run Experiment")
        self._run_button.setObjectName("runButton")
        self._run_button.setCursor(Qt.PointingHandCursor)
        self._run_button.setDefault(True)

        self._export_button = QPushButton("Export Report…")
        self._export_button.setObjectName("runButton")
        self._export_button.setCursor(Qt.PointingHandCursor)
        self._export_button.setEnabled(False)

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

        self._batch_results_table = self._build_table(
            [
                "Source",
                "Target",
                "Status",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "Samples",
                "Correct",
                "Wrong",
            ]
        )
        self._align_headers(
            self._batch_results_table,
            [
                Qt.AlignLeft,
                Qt.AlignLeft,
                Qt.AlignCenter,
                Qt.AlignRight,
                Qt.AlignRight,
                Qt.AlignRight,
                Qt.AlignRight,
                Qt.AlignRight,
                Qt.AlignRight,
                Qt.AlignRight,
            ],
        )
        self._batch_summary_label = QLabel(
            "Run a batch to see per-pair results and the average across the batch."
        )
        self._batch_summary_label.setObjectName("captionLabel")
        self._batch_summary_label.setWordWrap(True)

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
        self._quit_action = QAction("&Quit", self)
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

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            self._title_bar.refresh_maximize_glyph()

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

        # title_box = QVBoxLayout()
        # title_box.setContentsMargins(0, 0, 0, 0)
        # title_box.setSpacing(0)
        # title = QLabel("Adapta: Transfer Learning Studio")
        # title.setObjectName("appTitle")
        # subtitle = QLabel("Domain adaptation experiment workbench")
        # subtitle.setObjectName("appSubtitle")
        # title_box.addWidget(title)
        # title_box.addWidget(subtitle)

        # layout.addWidget(mark)
        # layout.addLayout(title_box)
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
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        grid.addWidget(self._field_label("Source"), 0, 0)
        grid.addWidget(self._source_input, 0, 1)

        grid.addWidget(self._field_label("Target"), 1, 0)
        grid.addWidget(self._target_input, 1, 1)

        grid.addWidget(self._swap_button, 0, 2, 2, 1, Qt.AlignVCenter)
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

        rows = [
            (self._method_label, self._method_combo),
            (self._classifier_label, self._classifier_combo),
            (self._kernel_label, self._kernel_combo),
            (self._norm_label, self._norm_combo),
            (self._std_label, self._std_combo),
            (self._reg_label, self._reg_input),
            (self._dim_label, self._dim_input),
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
        }
        self._tunable_tooltips = {
            field: widget.toolTip() for field, (_, widget) in self._tunable_controls.items()
        }
        return group

    def _build_action_row(self) -> QVBoxLayout:
        row = QVBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._run_button)
        row.addWidget(self._export_button)
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
        batch_layout.addWidget(self._batch_summary_label)
        batch_layout.addWidget(self._batch_results_table, 1)

        self._charts_widget = AnalyticsChartsWidget(self._palette)
        self._history_tab = HistoryTab(self._catalog, self._palette)

        self._results_tabs.addTab(predictions_page, "Predictions")
        self._results_tabs.addTab(self._charts_widget, "Visualizations")
        self._results_tabs.addTab(features_page, "Transformed Features")
        self._results_tabs.addTab(self._summary_view, "Dataset Summary")
        self._results_tabs.addTab(batch_page, "Batch Results")
        self._results_tabs.addTab(self._history_tab, "History")
        self._batch_results_tab_index = self._results_tabs.indexOf(batch_page)
        self._history_tab_index = self._results_tabs.indexOf(self._history_tab)

        # These six are permanent — only dynamically opened experiment-detail
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
        self._batch_mode_check.toggled.connect(self._on_batch_mode_toggled)
        self._add_batch_files_button.clicked.connect(self._add_batch_files)
        self._remove_batch_file_button.clicked.connect(self._remove_selected_batch_files)
        self._clear_batch_files_button.clicked.connect(self._clear_batch_files)
        self._run_button.clicked.connect(self._run_transfer_learning)
        self._export_button.clicked.connect(self._export_pdf_report)

        self._open_source_action.triggered.connect(self._pick_source)
        self._open_target_action.triggered.connect(self._pick_target)
        self._swap_datasets_action.triggered.connect(self._swap_datasets)
        self._export_action.triggered.connect(self._export_pdf_report)
        self._run_action.triggered.connect(self._run_transfer_learning)
        self._quit_action.triggered.connect(self.close)
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
        self._batch_results_table.itemSelectionChanged.connect(self._open_pair_detail_tab)

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
        if self._current_user is not None:
            parts.append(self._current_user.name)
        self._context_label.setText("  ·  ".join(parts))

    def _set_status(self, text: str, color_key: str) -> None:
        """Show a status message. ``color_key`` names a semantic palette entry
        (e.g. "success", "warning") rather than a literal colour, so a theme
        switch can recolour the currently displayed status."""
        self._status_key = color_key
        color = self._palette[color_key]
        self._status_dot.setStyleSheet(f"color: {color};")
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    def _update_user_label(self) -> None:
        if self._current_user is not None:
            self._user_label.setText(f"Signed in as {self._current_user.display_name}")
        elif self._catalog is None:
            self._user_label.setText("No database — experiments are not recorded")
        else:
            self._user_label.setText("Signed out")

    def _update_auth_ui(self) -> None:
        signed_in = self._current_user is not None
        can_authenticate = self._auth is not None

        if signed_in:
            self._logout_button.setText("Log Out")
            self._logout_button.setToolTip("Sign out and hide experiment history")
        else:
            self._logout_button.setText("Log In")
            self._logout_button.setToolTip("Log in to record and review experiments")

        self._logout_button.setEnabled(can_authenticate)
        self._manage_users_action.setEnabled(
            self._catalog is not None and signed_in and self._current_user is not None and self._current_user.role == UserRole.ADMIN
        )
        self._refresh_history_action.setEnabled(self._catalog is not None and signed_in)
        if hasattr(self, "_browse_history_button"):
            self._browse_history_button.setVisible(self._catalog is not None and signed_in)

        self._history_tab.set_current_user(self._current_user)
        self._set_history_visible(signed_in)

    def _set_history_visible(self, visible: bool) -> None:
        if not hasattr(self, "_results_tabs"):
            return

        index = self._results_tabs.indexOf(self._history_tab)
        if visible:
            if index < 0:
                insert_at = min(self._history_tab_index, self._results_tabs.count())
                self._results_tabs.insertTab(insert_at, self._history_tab, "History")
                # insertTab always gives a fresh tab a close button when the
                # widget is tabsClosable — History is permanent like the other
                # five fixed tabs, so strip it (as _build_results_tabs does
                # for the initial six) or it would show a close "x" that,
                # if clicked, deletes the singleton HistoryTab widget.
                self._results_tabs.tabBar().setTabButton(
                    insert_at, QTabBar.RightSide, None
                )
        elif index >= 0:
            if self._results_tabs.currentIndex() == index:
                self._results_tabs.setCurrentIndex(0)
            self._results_tabs.removeTab(index)

    def _handle_auth_button_clicked(self) -> None:
        if self._current_user is None:
            self._show_login_dialog()
        else:
            self._sign_out()

    def _show_login_dialog(self) -> None:
        if self._auth is None:
            message_box.information(
                self,
                "Log In Unavailable",
                "Sign-in is only available when the database connection is active.",
            )
            return

        dialog = AuthDialog(self._auth, build_style_sheet(self._palette), parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        self._current_user = dialog.authenticated_user()
        self._update_user_label()
        self._update_context_label()
        self._update_auth_ui()
        if self._current_user is not None:
            self._set_status(f"Signed in as {self._current_user.display_name}", "success")

    def _sign_out(self) -> None:
        self._current_user = None
        self._update_user_label()
        self._update_context_label()
        self._update_auth_ui()
        self._set_status("You have been logged out.", "neutral")
        message_box.information(
            self,
            "Logged Out",
            "You have been logged out. You can keep running experiments, but history stays hidden until you log in again.",
        )

    def _current_user_id(self) -> int | None:
        return self._current_user.id if self._current_user else None

    def _persist_run(
        self,
        request: TransferLearningRequest | None,
        result: TransferLearningResult | None,
        error_message: str | None,
    ) -> None:
        """Record one run's datasets and experiment. No history refresh.

        Unauthenticated runs (without log in) are NOT recorded into the database.
        Shared by both the single-run and batch completion handlers, so every
        pair in a batch is catalogued exactly like a standalone run.
        """
        if (
            self._catalog is None
            or request is None
            or self._current_user is None
            or self._current_user.id is None
        ):
            return

        supported = self._use_case.supported_parameters(request.method_name)

        try:
            if error_message is None and result is not None:
                self._catalog.record_success(
                    request, result, self._current_user_id(), supported
                )
            else:
                self._catalog.record_failure(
                    request, self._current_user_id(), error_message or "Unknown error", supported
                )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Run not recorded: {exc}", "warning")

    def _record_run(
        self, result: TransferLearningResult | None, error_message: str | None
    ) -> None:
        """Persist the single run that just finished and refresh History."""
        self._persist_run(self._last_request, result, error_message)
        self._history_tab.refresh()

    def _record_batch_run(self, batch_result: BatchTransferLearningResult) -> None:
        """Persist every pair of a batch run as its own experiment/dataset pair."""
        shared = self._last_batch_request
        if shared is None:
            return

        for pair in batch_result.pairs:
            pair_request = TransferLearningRequest(
                method_name=shared.method_name,
                source_path=pair.source_path,
                target_path=pair.target_path,
                normalization=shared.normalization,
                standardization=shared.standardization,
                kernel_type=shared.kernel_type,
                regularization=shared.regularization,
                latent_dim=shared.latent_dim,
                classifier=shared.classifier,
            )
            self._persist_run(pair_request, pair.result, pair.error)

        self._history_tab.refresh()

    def _show_history(self) -> None:
        """Bring the History tab to the front and reload user-specific history."""
        if self._catalog is None or self._current_user is None:
            return
        self._history_tab.set_current_user(self._current_user)
        self._set_history_visible(True)
        self._results_stack.setCurrentIndex(1)
        index = self._results_tabs.indexOf(self._history_tab)
        if index >= 0:
            self._results_tabs.setCurrentIndex(index)

    def _open_user_manager(self) -> None:
        if self._catalog is None:
            return
        dialog = UserManagerDialog(self._catalog, build_style_sheet(self._palette), self)
        dialog.exec_()
        # Names and roles may have changed, including the signed-in user's own.
        if self._current_user is not None and self._current_user.id is not None:
            refreshed = self._catalog.get_user(self._current_user.id)
            if refreshed is not None:
                self._current_user = refreshed
                self._update_user_label()
        self._history_tab.refresh()

    def _open_dataset_registry(self) -> None:
        if self._catalog is None:
            return
        dialog = DatasetRegistryDialog(
            self._catalog, build_style_sheet(self._palette), self
        )
        dialog.exec_()

    def _show_database_info(self) -> None:
        if self._catalog is None:
            message_box.warning(
                self,
                "Database Unavailable",
                "No PostgreSQL connection.\n\n"
                + (self._database_error or "No connection details were configured."),
            )
            return

        try:
            stats = self._catalog.statistics()
        except Exception as exc:  # noqa: BLE001
            message_box.critical(self, "Database Error", str(exc))
            return

        message_box.information(
            self,
            "Database Connection",
            "<b>PostgreSQL connected</b>"
            "<p>Stored records:</p>"
            f"<ul><li>Users: {stats['users']}</li>"
            f"<li>Datasets: {stats['datasets']}</li>"
            f"<li>Experiments: {stats['experiments']}</li>"
            f"<li>Evaluations: {stats['evaluations']}</li></ul>",
        )

    def _apply_configuration(self, experiment: ExperimentRecord) -> None:
        """Copy a stored experiment's settings back into the run form.

        Only the settings — not the dataset paths, which may since have moved
        or been deleted. Selections are applied by ``findData``/``findText`` so
        a value the current build no longer offers is skipped rather than
        silently landing on the wrong entry.
        """
        method_index = self._method_combo.findText(experiment.tl_algorithm)
        if method_index >= 0:
            self._method_combo.setCurrentIndex(method_index)

        for combo, value in (
            (self._classifier_combo, experiment.classifier),
            (self._norm_combo, experiment.normalization_type),
            (self._std_combo, experiment.standardization_type),
            (self._kernel_combo, experiment.kernel_type),
        ):
            if value is None:
                continue
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)

        if experiment.regularization_value is not None:
            self._reg_input.setValue(float(experiment.regularization_value))
        if experiment.latent_feature is not None:
            self._dim_input.setValue(int(experiment.latent_feature))

        self._sync_method_parameters()
        self._set_status(
            f"Loaded configuration from experiment #{experiment.id}", "neutral"
        )

    def _open_experiment_detail_tab(self, entry: ExperimentHistoryEntry) -> None:
        """Open (or raise) a closable tab with every field of *entry*.

        Joins in the source/target ``DatasetRecord`` rows, which
        ``ExperimentHistoryEntry`` doesn't carry (it only denormalises the
        domain *names*) — this is the one place that needs the full rows.
        """
        experiment_id = entry.experiment.id
        if experiment_id is None or self._catalog is None:
            return

        existing = self._experiment_detail_tabs.get(experiment_id)
        if existing is not None:
            index = self._results_tabs.indexOf(existing)
            if index >= 0:
                self._results_tabs.setCurrentIndex(index)
                return
            # The tab was closed without going through _close_results_tab
            # (shouldn't normally happen); fall through and rebuild it.
            del self._experiment_detail_tabs[experiment_id]

        try:
            source_dataset = self._catalog.get_dataset(entry.experiment.source_dataset_id)
            target_dataset = self._catalog.get_dataset(entry.experiment.target_dataset_id)
        except Exception:  # noqa: BLE001 - the detail view just shows "—" instead
            source_dataset = None
            target_dataset = None

        view = ExperimentDetailView(entry, source_dataset, target_dataset, self._palette)
        index = self._results_tabs.addTab(view, f"Experiment #{experiment_id}")
        self._experiment_detail_tabs[experiment_id] = view
        self._results_tabs.setCurrentIndex(index)

    def _open_pair_detail_tab(self) -> None:
        """Open (or raise) a closable tab for the row selected in Batch Results.

        Mirrors ``_open_experiment_detail_tab``'s "select a row, get a tab"
        pattern. Fires on every selection change, so re-selecting an
        already-open pair's row raises its tab instead of duplicating it.
        Batch pairs have no persisted id (unlike an ``ExperimentRecord``), so
        the pair object's identity is the key instead.
        """
        if self._last_batch_result is None:
            return
        row = self._batch_results_table.currentRow()
        if not (0 <= row < len(self._last_batch_result.pairs)):
            return
        pair = self._last_batch_result.pairs[row]
        pair_key = id(pair)

        existing = self._pair_detail_tabs.get(pair_key)
        if existing is not None:
            index = self._results_tabs.indexOf(existing)
            if index >= 0:
                self._results_tabs.setCurrentIndex(index)
                return
            del self._pair_detail_tabs[pair_key]

        view = PairDetailView(pair, self._palette)
        source_name = Path(pair.source_path).stem
        target_name = Path(pair.target_path).stem
        index = self._results_tabs.addTab(view, f"{source_name} → {target_name}")
        self._pair_detail_tabs[pair_key] = view
        self._results_tabs.setCurrentIndex(index)

    def _close_results_tab(self, index: int) -> None:
        """Handle the tab strip's close button — only detail tabs have one.

        Guards against ever deleting one of the six permanent tabs (their
        close buttons are stripped, but this is a second line of defence):
        only a widget actually tracked in ``_experiment_detail_tabs`` or
        ``_pair_detail_tabs`` gets removed and deleted.
        """
        widget = self._results_tabs.widget(index)
        experiment_id = next(
            (eid for eid, view in self._experiment_detail_tabs.items() if view is widget),
            None,
        )
        if experiment_id is not None:
            del self._experiment_detail_tabs[experiment_id]
            self._results_tabs.removeTab(index)
            widget.deleteLater()
            return

        pair_key = next(
            (key for key, view in self._pair_detail_tabs.items() if view is widget),
            None,
        )
        if pair_key is None:
            return
        del self._pair_detail_tabs[pair_key]
        self._results_tabs.removeTab(index)
        widget.deleteLater()

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------
    def _apply_theme_stylesheet(self) -> None:
        stylesheet = build_style_sheet(self._palette)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet)
        self.setStyleSheet(stylesheet)

    def _toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self._palette = LIGHT_PALETTE if self._theme == "light" else DARK_PALETTE
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._apply_theme_stylesheet()
        self._theme_toggle_button.setText("🌙  Dark" if self._theme == "light" else "☀  Light")

        color = self._palette[self._status_key]
        self._status_dot.setStyleSheet(f"color: {color};")
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

        self._charts_widget.set_palette(self._palette)
        self._history_tab.set_palette(self._palette)
        if self._last_result is not None:
            self._show_predictions(self._last_result)

        if hasattr(self, "_learning_materials_dialog") and self._learning_materials_dialog is not None:
            try:
                self._learning_materials_dialog.set_palette(self._palette)
            except RuntimeError:
                self._learning_materials_dialog = None

    def _open_learning_materials(self, category_index: int = 0) -> None:
        dialog = getattr(self, "_learning_materials_dialog", None)
        try:
            if dialog is not None:
                dialog.select_topic(category_index)
                dialog.raise_()
                dialog.activateWindow()
                return
        except RuntimeError:
            pass

        from presentation.qt.learning_materials_dialog import LearningMaterialsDialog

        self._learning_materials_dialog = LearningMaterialsDialog(
            self._palette, parent=self
        )
        self._learning_materials_dialog.select_topic(category_index)
        self._learning_materials_dialog.destroyed.connect(
            lambda: setattr(self, "_learning_materials_dialog", None)
        )
        self._learning_materials_dialog.show()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pick_source(self) -> None:
        self._pick_file(self._source_input, "Select Source Dataset")

    def _pick_target(self) -> None:
        self._pick_file(self._target_input, "Select Target Dataset")

    def _swap_datasets(self) -> None:
        source_text = self._source_input.text()
        target_text = self._target_input.text()
        self._source_input.setText(target_text)
        self._target_input.setText(source_text)

    def _pick_file(self, target_input: QLineEdit, title: str) -> None:
        start_dir = target_input.text().strip() or str(Path.cwd())
        dialog = QFileDialog(self, title, start_dir, DATASET_FILTER)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setStyleSheet(build_style_sheet(self._palette))
        install_title_bar_on_grid(dialog, title)
        if dialog.exec_() == QDialog.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files:
                target_input.setText(selected_files[0])

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    def _on_batch_mode_toggled(self, checked: bool) -> None:
        self._single_dataset_widget.setVisible(not checked)
        self._batch_dataset_widget.setVisible(checked)
        self._run_button.setText("Run Batch" if checked else "Run Experiment")

    def _add_batch_files(self) -> None:
        start_dir = str(Path.cwd())
        dialog = QFileDialog(self, "Select Dataset Files", start_dir, DATASET_FILTER)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setStyleSheet(build_style_sheet(self._palette))
        install_title_bar_on_grid(dialog, "Select Dataset Files")
        file_paths = []
        if dialog.exec_() == QDialog.Accepted:
            file_paths = dialog.selectedFiles()
        existing = {
            self._batch_files_list.item(i).text() for i in range(self._batch_files_list.count())
        }
        for path in file_paths:
            if path not in existing:
                self._batch_files_list.addItem(path)
                existing.add(path)
        self._update_batch_caption()

    def _remove_selected_batch_files(self) -> None:
        for item in self._batch_files_list.selectedItems():
            self._batch_files_list.takeItem(self._batch_files_list.row(item))
        self._update_batch_caption()

    def _clear_batch_files(self) -> None:
        self._batch_files_list.clear()
        self._update_batch_caption()

    def _update_batch_caption(self) -> None:
        count = self._batch_files_list.count()
        if count < 2:
            self._batch_files_caption.setText("Add 2+ files to enable batch mode.")
        else:
            pair_count = count * (count - 1)
            self._batch_files_caption.setText(
                f"{count} files selected → {pair_count} ordered pairs will be run."
            )

    def _run_transfer_learning(self) -> None:
        # Guard against a stale reference: once the C++ QThread is deleted by
        # deleteLater the Python wrapper raises RuntimeError on any method call.
        try:
            if self._worker is not None and self._worker.isRunning():
                return
        except RuntimeError:
            # C++ object already deleted – treat it as "not running".
            self._worker = None

        if self._batch_mode_check.isChecked():
            self._run_batch_experiment()
        else:
            self._run_single_experiment()

    def _run_single_experiment(self) -> None:
        source_path = self._source_input.text().strip()
        target_path = self._target_input.text().strip()

        if not source_path or not target_path:
            message_box.warning(
                self,
                "Missing Dataset",
                "Select both a source and a target dataset before running.",
            )
            return

        missing = [p for p in (source_path, target_path) if not Path(p).is_file()]
        if missing:
            message_box.warning(
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

        # Kept so the completion handlers can record what was actually run,
        # even if the user edits the form while the worker is still going.
        self._last_request = request

        self._set_running(True)
        self._set_status("Running experiment…", "warning")

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

    def _run_batch_experiment(self) -> None:
        file_paths = [
            self._batch_files_list.item(i).text() for i in range(self._batch_files_list.count())
        ]

        if len(file_paths) < 2:
            message_box.warning(
                self,
                "Not Enough Files",
                "Add at least two dataset files to run a batch.",
            )
            return

        missing = [p for p in file_paths if not Path(p).is_file()]
        if missing:
            message_box.warning(
                self,
                "Dataset Not Found",
                "The following path is not a readable file:\n\n" + "\n".join(missing),
            )
            return

        request = BatchTransferLearningRequest(
            method_name=self._method_combo.currentText(),
            file_paths=file_paths,
            normalization=self._norm_combo.currentData(),
            standardization=self._std_combo.currentData(),
            kernel_type=self._kernel_combo.currentData(),
            regularization=self._reg_input.value(),
            latent_dim=self._dim_input.value(),
            classifier=self._classifier_combo.currentData(),
        )

        # Kept so the completion handler can record each pair as its own
        # experiment, even if the user edits the form while the worker runs.
        self._last_batch_request = request

        pair_count = len(file_paths) * (len(file_paths) - 1)
        self._set_running(True)
        self._set_status(f"Running batch ({pair_count} pairs)…", "warning")

        self._worker = BatchTransferLearningWorker(self._use_case, request)
        self._worker.succeeded.connect(self._on_batch_success)
        self._worker.failed.connect(self._on_batch_error)
        self._worker.finished.connect(lambda: self._set_running(False))
        self._worker.finished.connect(lambda: setattr(self, "_worker", None))
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._run_action.setEnabled(not running)
        running_label = "Running…"
        idle_label = "Run Batch" if self._batch_mode_check.isChecked() else "Run Experiment"
        self._run_button.setText(running_label if running else idle_label)
        self._progress_bar.setVisible(running)
        self._batch_mode_check.setEnabled(not running)
        self._add_batch_files_button.setEnabled(not running)
        self._remove_batch_file_button.setEnabled(not running)
        self._clear_batch_files_button.setEnabled(not running)

    def _on_success(self, result: TransferLearningResult) -> None:
        self._last_result = result
        self._set_status("Completed", "success")
        self._export_button.setEnabled(True)
        self._export_action.setEnabled(True)

        self._show_metrics(result.metrics)
        self._show_dataset_summary(result)
        self._show_predictions(result)
        self._show_transformed_features(result)
        self._charts_widget.update_result(result)

        self._results_fullscreen_button.setEnabled(True)

        self._results_stack.setCurrentIndex(1)
        self._update_context_label()
        self._record_run(result, error_message=None)

    def _on_error(self, message: str) -> None:
        self._set_status("Failed", "danger")
        # Recorded before the modal opens: exec_() blocks until dismissed, and
        # a failed configuration is worth keeping either way.
        self._record_run(None, error_message=message)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Experiment Failed")
        box.setText("The transfer learning run could not be completed.")
        box.setInformativeText(message)
        install_title_bar_on_grid(box, "Experiment Failed")
        box.exec_()

    def _on_batch_success(self, batch_result: BatchTransferLearningResult) -> None:
        self._last_batch_result = batch_result
        succeeded = sum(1 for pair in batch_result.pairs if pair.result is not None)
        failed = len(batch_result.pairs) - succeeded
        status = "Batch completed" if failed == 0 else f"Batch completed ({failed} pair(s) failed)"
        self._set_status(status, "success" if failed == 0 else "warning")

        # A single PDF report only makes sense for one pair — disable export
        # rather than have it silently describe a stale single-pair result.
        self._export_button.setEnabled(False)
        self._export_action.setEnabled(False)

        # Metric cards show the batch-wide average, matching the summary
        # line on the Batch Results tab.
        self._show_metrics(batch_result.average_metrics)
        if batch_result.average_metrics is not None:
            evaluable = sum(
                1 for pair in batch_result.pairs if pair.result is not None and pair.result.metrics is not None
            )
            self._counts_label.setText(
                f"Average over {evaluable} of {len(batch_result.pairs)} pairs  ·  "
                f"{batch_result.average_metrics.total_samples} target samples evaluated in total  ·  "
                f"{batch_result.average_metrics.correct_predictions} correct  ·  "
                f"{batch_result.average_metrics.wrong_predictions} incorrect"
            )

        self._populate_batch_results(batch_result)
        self._results_fullscreen_button.setEnabled(True)
        self._results_stack.setCurrentIndex(1)
        if self._batch_results_tab_index >= 0:
            self._results_tabs.setCurrentIndex(self._batch_results_tab_index)
        self._record_batch_run(batch_result)

    def _on_batch_error(self, message: str) -> None:
        self._set_status("Batch Failed", "danger")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Batch Run Failed")
        box.setText("The batch run could not be completed.")
        box.setInformativeText(message)
        install_title_bar_on_grid(box, "Batch Run Failed")
        box.exec_()

    def _populate_batch_results(self, batch_result: BatchTransferLearningResult) -> None:
        table = self._batch_results_table
        table.setRowCount(len(batch_result.pairs))

        success_brush = QBrush(QColor(self._palette["success"]))
        danger_brush = QBrush(QColor(self._palette["danger"]))
        succeeded = 0

        for row, pair in enumerate(batch_result.pairs):
            source_name = Path(pair.source_path).name
            target_name = Path(pair.target_path).name
            metrics = pair.result.metrics if pair.result is not None else None

            if metrics is not None:
                succeeded += 1
                status_text, status_brush = "OK", success_brush
                values = [
                    f"{metrics.accuracy:.4f}",
                    f"{metrics.precision:.4f}",
                    f"{metrics.recall:.4f}",
                    f"{metrics.f1_score:.4f}",
                    str(metrics.total_samples),
                    str(metrics.correct_predictions),
                    str(metrics.wrong_predictions),
                ]
            elif pair.result is not None:
                status_text, status_brush = "No labels", None
                values = ["—"] * 7
            else:
                status_text, status_brush = "Error", danger_brush
                values = [pair.error or "Unknown error"] + [""] * 6

            cells = [source_name, target_name, status_text, *values]
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if column >= 2:
                    item.setTextAlignment(Qt.AlignCenter if column == 2 else Qt.AlignRight | Qt.AlignVCenter)
                if column == 2 and status_brush is not None:
                    item.setForeground(status_brush)
                if column == 3 and status_text == "Error":
                    item.setToolTip(pair.error or "")
                table.setItem(row, column, item)

        header = table.horizontalHeader()
        for column in range(2):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for column in range(2, 10):
            header.setSectionResizeMode(column, QHeaderView.Stretch)

        total = len(batch_result.pairs)
        avg = batch_result.average_metrics
        if avg is not None:
            self._batch_summary_label.setText(
                f"{total} pairs run — {succeeded} produced metrics, {total - succeeded} did not. "
                f"Average over evaluable pairs: accuracy {avg.accuracy:.4f} | "
                f"precision {avg.precision:.4f} | recall {avg.recall:.4f} | f1 {avg.f1_score:.4f} "
                f"({avg.correct_predictions}/{avg.total_samples} correct)"
            )
        else:
            self._batch_summary_label.setText(
                f"{total} pairs run — no pair produced evaluable metrics "
                "(datasets may be missing labels, or every pair failed)."
            )
    def _show_about(self) -> None:
        title = "About Adapta: Transfer Learning Studio"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(title)
        box.setStyleSheet(build_style_sheet(self._palette))
        box.setText(
            "<b>Adapta: Transfer Learning Studio</b>"
            "<p>Desktop workbench for domain adaptation experiments.</p>"
            "<p>Methods: "
            + ", ".join(self._use_case.method_names)
            + ".<br>Classifiers: k-NN, SVM, Random Forest, Logistic Regression, Decision Tree.<br>"
            "The selected classifier is fitted on the transformed source domain and "
            "evaluated on the transformed target domain.</p>"
        )
        install_title_bar_on_grid(box, title)
        box.exec_()

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
        self._summary_view.update_result(result)

    def _show_predictions(self, result: TransferLearningResult) -> None:
        if result.target_predictions is None:
            self._predictions_table.clearContents()
            self._predictions_table.setRowCount(0)
            self._incorrect_only_check.setEnabled(False)
            self._correct_only_check.setEnabled(False)
            self._results_tabs.setTabEnabled(0, False)
            self._results_tabs.setCurrentIndex(1)
            return

        self._results_tabs.setTabEnabled(0, True)
        self._incorrect_only_check.setEnabled(True)
        self._correct_only_check.setEnabled(True)
        self._results_tabs.setCurrentIndex(0)

        populate_predictions_table(self._predictions_table, result, self._palette)
        self._apply_prediction_filter()

    def _on_prediction_filter_toggled(self) -> None:
        sender = self.sender()
        if sender is None or not sender.isChecked():
            self._apply_prediction_filter()
            return
        other = self._incorrect_only_check if sender is self._correct_only_check else self._correct_only_check
        other.blockSignals(True)
        other.setChecked(False)
        other.blockSignals(False)
        self._apply_prediction_filter()

    def _apply_prediction_filter(self) -> None:
        """Hide rows according to the active filter."""
        result = self._last_result
        if result is None or result.prediction_correctness is None:
            return

        correct_only = self._correct_only_check.isChecked()
        incorrect_only = self._incorrect_only_check.isChecked()
        for row, correctness in enumerate(result.prediction_correctness):
            hide = (
                incorrect_only and correctness == "Right"
            ) or (correct_only and correctness == "Wrong")
            self._predictions_table.setRowHidden(row, hide)

    def _open_results_fullscreen(self) -> None:
        """Move the whole results tab widget into a maximized window.

        All tabs (Predictions, Visualizations, Transformed Features, Dataset
        Summary, Batch Results, History) move together and stay switchable —
        this is one full-screen view over the tab strip, not a separate one
        per tab. ``self._results_tabs`` is reparented in place (not copied),
        so it keeps its live state, and is handed back to the same slot in
        ``self._results_stack`` when the window closes.
        """
        win = getattr(self, "_results_window", None)
        try:
            if win is not None:
                win.raise_()
                win.activateWindow()
                return
        except RuntimeError:
            # The user already closed the previous window (WA_DeleteOnClose).
            pass

        origin_index = self._results_stack.indexOf(self._results_tabs)
        self._results_stack.removeWidget(self._results_tabs)
        # The corner button travels with the tab widget (it's the same
        # instance, just reparented) — hide it so a window already in full
        # screen doesn't show a button to open full screen again.
        self._results_fullscreen_button.hide()
        win = FullScreenViewer(
            "Results", self._results_tabs, build_style_sheet(self._palette), parent=None
        )
        # QStackedWidget.removeWidget() leaves the widget hidden (it was only
        # ever shown as the "current" stack page) — without this it silently
        # stays invisible in its new window.
        self._results_tabs.show()
        self._results_window = win

        def _restore() -> None:
            self._results_stack.insertWidget(origin_index, self._results_tabs)
            self._results_tabs.show()
            self._results_stack.setCurrentWidget(self._results_tabs)
            self._results_fullscreen_button.show()
            self._results_window = None

        win.destroyed.connect(_restore)

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

        dialog = QFileDialog(
            self,
            "Export PDF Report",
            str(Path.cwd() / "Transfer_Learning_Report.pdf"),
            "PDF Files (*.pdf)",
        )
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("pdf")
        dialog.setStyleSheet(build_style_sheet(self._palette))
        install_title_bar_on_grid(dialog, "Export PDF Report")
        if dialog.exec_() != QDialog.Accepted:
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return
        file_path = selected_files[0]
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
            message_box.critical(
                self, "Export Failed", f"The report could not be written:\n\n{exc}"
            )
            return

        self._set_status("Report exported", "success")
        message_box.information(
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

        chart_html = self._charts_widget.render_to_base64_html()
        if chart_html:
            parts.append("<h2>Visual Analytics</h2>")
            parts.append(chart_html)

        if result.target_predictions is not None:
            parts.append("<h2>Target Predictions</h2><table width='100%'>")
            parts.append("<tr><th>Sample</th><th>Actual Label</th><th>Predicted Label</th>"
                         "<th>Result</th></tr>")
            for row in range(len(result.target_predictions)):
                is_correct = result.prediction_correctness[row] == "Right"
                colour = self._palette["success"] if is_correct else self._palette["danger"]
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
