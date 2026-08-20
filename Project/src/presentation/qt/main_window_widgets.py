"""Creating every widget the window owns.

One job: construct widgets and set their static properties. No layout
(``main_window_layout.py``), no signal wiring (``main_window_signals.py``),
and no behaviour — the handlers live in the mixins named after what they
do. This is the file to open when asking "where does this control come
from?".
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
)

from presentation.qt.dataset_summary_view import DatasetSummaryView
from presentation.qt.metric_card import MetricCard
from presentation.qt.result_tables import BATCH_RESULTS_HEADERS, METHOD_RESULTS_HEADERS
from presentation.qt.ui_options import (
    CLASSIFIER_OPTIONS,
    KERNEL_OPTIONS,
    NORMALIZATION_OPTIONS,
    STANDARDIZATION_OPTIONS,
)
from presentation.qt.ui_widgets import ClickableLineEdit as _ClickableLineEdit

class MainWindowWidgetsMixin:
    """Creating every widget the window owns."""

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

        # Shown only while the Datasets group is collapsed, so the section
        # still says what it is configured with when its fields are hidden.
        self._dataset_summary_label = QLabel()
        self._dataset_summary_label.setObjectName("captionLabel")
        self._dataset_summary_label.setWordWrap(False)
        self._dataset_summary_label.hide()

        self._dataset_toggle_button = QPushButton("Hide")
        self._dataset_toggle_button.setObjectName("linkButton")
        self._dataset_toggle_button.setCursor(Qt.PointingHandCursor)
        self._dataset_toggle_button.setToolTip(
            "Collapse the dataset fields to give the results more room"
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

        self._multi_method_check = QCheckBox("Compare several methods on this pair")
        self._multi_method_check.setToolTip(
            "Runs each ticked method against the same source and target, with "
            "identical preprocessing and one shared no-adaptation baseline, "
            "then ranks them."
        )

        # Checkable list rather than a multi-select: a tick survives
        # scrolling and mis-clicks, where a ctrl-click selection is one
        # stray click away from being silently thrown out.
        self._method_list = QListWidget()
        self._method_list.setToolTip("Methods to run against this dataset pair")
        self._method_list.setMaximumHeight(150)
        for name in self._use_case.method_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, name)
            self._method_list.addItem(item)

        self._select_all_methods_button = QPushButton("All")
        self._clear_methods_button = QPushButton("None")
        for button in (self._select_all_methods_button, self._clear_methods_button):
            button.setObjectName("linkButton")
            button.setCursor(Qt.PointingHandCursor)

        self._method_selection_caption = QLabel("Tick 2+ methods to compare them.")
        self._method_selection_caption.setObjectName("captionLabel")
        self._method_selection_caption.setWordWrap(True)

        self._method_results_table = self._build_table(list(METHOD_RESULTS_HEADERS))
        self._align_headers(
            self._method_results_table,
            [Qt.AlignCenter, Qt.AlignLeft, Qt.AlignCenter]
            + [Qt.AlignRight] * (len(METHOD_RESULTS_HEADERS) - 3),
        )
        self._method_summary_label = QLabel(
            "Tick several methods in the sidebar to compare them on one dataset pair."
        )
        self._method_summary_label.setObjectName("captionLabel")
        self._method_summary_label.setWordWrap(True)

        # On the tab itself, not only in the sidebar, so it stays reachable
        # from the full-screen results window.
        self._method_export_button = QPushButton("Export Report…")
        self._method_export_button.setCursor(Qt.PointingHandCursor)
        self._method_export_button.setEnabled(False)
        self._method_export_button.setToolTip(
            "Export a PDF comparing every method in this run"
        )

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

        self._home_button = QPushButton("⌂  Home")
        self._home_button.setObjectName("homeButton")
        self._home_button.setCursor(Qt.PointingHandCursor)
        self._home_button.setToolTip("Back to the landing page (Ctrl+H)")

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

        # One click for what dragging the splitter handle all the way up
        # does — that drag is fiddly precisely at the extreme, which is the
        # position most worth reaching. Glyph-only: it shares the tab strip's
        # corner with Full Screen, and nine tabs already compete for that
        # width. The tooltip carries the explanation.
        self._results_expand_button = QPushButton("⌃")
        self._results_expand_button.setObjectName("fullscreenButton")
        self._results_expand_button.setCursor(Qt.PointingHandCursor)
        self._results_expand_button.setFixedWidth(34)
        self._results_expand_button.setToolTip(
            "Give the results the whole column (the summary above stays one "
            "drag away)"
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


__all__ = ["MainWindowWidgetsMixin"]
