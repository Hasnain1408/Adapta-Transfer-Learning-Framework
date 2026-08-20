"""Arranging the constructed widgets into the two pages of the window.

The landing page and the workspace are two pages of one
``QStackedWidget`` rather than separate windows, so the title bar, the
theme, and a running experiment's progress all survive moving between
them.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from presentation.qt.charts_widget import AnalyticsChartsWidget
from presentation.qt.comparison_tab import ComparisonTab
from presentation.qt.history_tab import HistoryTab
from presentation.qt.landing_page import LandingPage
from presentation.qt.processing_status import ProcessingStatusStrip
from presentation.qt.visualization_tab import VisualizationTab

class MainWindowLayoutMixin:
    """Arranging the constructed widgets into the two pages of the window."""

    def _build_layout(self) -> None:
        """Two pages in one window: the landing page, then the workspace.

        A ``QStackedWidget`` rather than a second top-level window — the
        title bar, the theme, and (importantly) the status bar with a
        running experiment's progress all stay put while the user moves
        between them, and neither page has to be rebuilt on the way back.
        """
        workspace = QWidget()
        workspace.setObjectName("centralWidget")
        root = QVBoxLayout(workspace)
        root.setContentsMargins(18, 14, 18, 10)
        root.setSpacing(11)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_content(), 1)
        root.addLayout(body, 1)

        self._landing_page = LandingPage(self._palette)
        self._landing_page.navigate.connect(self._navigate_from_landing)

        self._page_stack = QStackedWidget(self)
        self._page_stack.setObjectName("centralWidget")
        self._page_stack.addWidget(self._landing_page)
        self._page_stack.addWidget(workspace)
        self.setCentralWidget(self._page_stack)

    def _build_header(self) -> QWidget:
        """Workspace header: the way back to the landing page, then session controls."""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(12)

        mark = QLabel()
        mark.setObjectName("brandMark")
        mark.setFixedSize(34, 34)

        layout.addWidget(self._home_button)
        layout.addStretch(1)
        layout.addWidget(self._user_label)
        layout.addWidget(self._logout_button)
        layout.addWidget(self._theme_toggle_button)
        return header

    def _build_sidebar(self) -> QWidget:
        """Left-hand column: method/parameters (scrolling) and run actions (pinned).

        The parameter group already wanted more height than this column has
        at the window's default size, and comparison mode adds a method list
        on top of that — so it scrolls rather than compressing every control
        towards zero (which is what a plain QVBoxLayout does when it runs out
        of room: the list collapsed to a single pixel).

        The action row deliberately sits *outside* the scroll area. Run,
        Pause, and Cancel must stay reachable no matter where the parameters
        are scrolled to; a Cancel button you have to go looking for during a
        long run is worse than no Cancel button at all.
        """
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(320)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        scroll = QScrollArea()
        scroll.setObjectName("sidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(self._build_parameter_group())

        layout.addWidget(scroll, 1)
        layout.addLayout(self._build_action_row())
        return sidebar

    def _build_content(self) -> QWidget:
        """Right-hand column: a draggable split between summary and results.

        Above the handle sits everything that describes the run — datasets,
        metric cards, counts, the progress strip. Below it are the results.
        A ``QSplitter`` rather than a fixed stretch factor because how much
        room the results deserve is a judgement only the user can make: it
        depends on which tab they are reading (a 700-row predictions table
        wants the whole column; a four-value alignment panel does not).

        Only the top pane is collapsible. Dragging the handle to the top
        gives the results the entire column, and the results pane keeps a
        minimum height so the tab strip can never be dragged out of reach —
        which would leave no way to drag it back.
        """
        summary = QWidget()
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(11)

        summary_layout.addWidget(self._build_dataset_group())
        summary_layout.addLayout(self._build_metrics_row())
        summary_layout.addWidget(self._counts_label)
        # Hidden until a run starts, so an idle window looks the way it
        # always did and the strip's appearance is itself a signal. Placed
        # directly above the results it is about, and in the content column
        # rather than the sidebar because the sidebar has no spare height.
        self._processing_strip = ProcessingStatusStrip()
        summary_layout.addWidget(self._processing_strip)
        # Keeps the block top-packed. Without it, a summary pane dragged
        # taller than its content spreads the cards and captions apart down
        # the whole pane instead of leaving the slack at the bottom.
        summary_layout.addStretch(1)

        self._content_splitter = QSplitter(Qt.Vertical)
        self._content_splitter.setObjectName("contentSplitter")
        self._content_splitter.setHandleWidth(10)
        self._content_splitter.addWidget(summary)
        self._content_splitter.addWidget(self._build_results_stack())
        self._content_splitter.setCollapsible(0, True)
        self._content_splitter.setCollapsible(1, False)
        self._content_splitter.setStretchFactor(0, 0)
        self._content_splitter.setStretchFactor(1, 1)
        self._results_stack.setMinimumHeight(180)
        self._summary_pane = summary
        # Start the summary at exactly the height its content wants and give
        # everything else to the results. The stretch factors above then hand
        # any extra from a window resize to the results too, so a maximised
        # window grows the table rather than the whitespace above it.
        self._content_splitter.setSizes([summary.sizeHint().height(), 10_000])

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._content_splitter, 1)
        return content

    def _build_dataset_group(self) -> QGroupBox:
        """The Datasets group, collapsible down to a one-line summary.

        Its inputs matter right up until a run starts and stop mattering the
        moment one finishes — at which point it is roughly 130px of form
        sitting above the results the user actually came to read. Collapsing
        it (see ``_set_datasets_collapsed``) hands that height to the results
        stack, which is the only stretching widget in this column.
        """
        group = QGroupBox("Datasets")
        outer = QVBoxLayout(group)
        outer.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(self._dataset_summary_label, 1)
        header.addWidget(self._dataset_toggle_button, 0, Qt.AlignRight)
        outer.addLayout(header)

        # Everything below the header lives in one widget so collapsing is a
        # single setVisible rather than a list of children to keep in sync.
        self._dataset_body = QWidget()
        body = QVBoxLayout(self._dataset_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)

        body.addWidget(self._batch_mode_check)

        self._single_dataset_widget = self._build_single_dataset_widget()
        self._batch_dataset_widget = self._build_batch_dataset_widget()
        body.addWidget(self._single_dataset_widget)
        body.addWidget(self._batch_dataset_widget)
        self._batch_dataset_widget.setVisible(False)

        hint = QLabel(
            "Accepted formats: CSV, XLSX, ARFF (last column/attribute is the label) and MATLAB .mat"
        )
        hint.setObjectName("captionLabel")
        body.addWidget(hint)

        outer.addWidget(self._dataset_body)
        self._dataset_group = group
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

        layout.addWidget(self._method_label)
        layout.addWidget(self._multi_method_check)
        # One slot, two ways to fill it: the combo for a single method, the
        # checkable list when comparing several. Swapped rather than shown
        # side by side so there is never a question of which one the run
        # will actually use.
        layout.addWidget(self._method_combo)
        layout.addWidget(self._build_method_selection_widget())

        rows = [
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

    def _build_method_selection_widget(self) -> QWidget:
        """The checkable method list, shown in place of the method combo."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._method_list)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        buttons.addWidget(self._select_all_methods_button)
        buttons.addWidget(self._clear_methods_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addWidget(self._method_selection_caption)

        self._method_selection_widget = widget
        widget.setVisible(False)
        return widget

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

        method_page = QWidget()
        method_layout = QVBoxLayout(method_page)
        method_layout.setContentsMargins(12, 10, 12, 12)
        method_layout.setSpacing(8)
        method_header = QHBoxLayout()
        method_header.addWidget(self._method_summary_label, 1)
        method_header.addWidget(self._method_export_button, 0, Qt.AlignTop)
        method_layout.addLayout(method_header)
        method_layout.addWidget(self._method_results_table, 1)

        self._charts_widget = AnalyticsChartsWidget(self._palette)
        self._visualization_tab = VisualizationTab(self._palette)
        self._comparison_tab = ComparisonTab(self._catalog, self._palette)
        self._history_tab = HistoryTab(self._catalog, self._palette)

        self._results_tabs.addTab(predictions_page, "Predictions")
        self._results_tabs.addTab(self._charts_widget, "Visualizations")
        self._results_tabs.addTab(self._visualization_tab, "Feature Space")
        self._results_tabs.addTab(features_page, "Transformed Features")
        self._results_tabs.addTab(self._summary_view, "Dataset Summary")
        # Named for what it holds — the methods of *this* run — to keep it
        # distinct from "Comparison", which ranks across saved experiments.
        self._results_tabs.addTab(method_page, "Method Results")
        self._results_tabs.addTab(batch_page, "Batch Results")
        self._results_tabs.addTab(self._comparison_tab, "Comparison")
        self._results_tabs.addTab(self._history_tab, "History")
        self._method_results_tab_index = self._results_tabs.indexOf(method_page)
        self._batch_results_tab_index = self._results_tabs.indexOf(batch_page)
        self._history_tab_index = self._results_tabs.indexOf(self._history_tab)

        # These are all permanent — only dynamically opened experiment-detail
        # tabs (see _open_experiment_detail_tab) are meant to be closable.
        for index in range(self._results_tabs.count()):
            self._results_tabs.tabBar().setTabButton(index, QTabBar.RightSide, None)
        self._set_history_visible(self._current_user is not None)

        # Both size controls share the corner. They travel with the tab
        # widget when it is reparented into the full-screen window, so the
        # container is what gets hidden there rather than each button.
        self._results_corner = QWidget()
        corner = QHBoxLayout(self._results_corner)
        corner.setContentsMargins(0, 0, 0, 0)
        corner.setSpacing(6)
        corner.addWidget(self._results_expand_button)
        corner.addWidget(self._results_fullscreen_button)
        self._results_tabs.setCornerWidget(self._results_corner, Qt.TopRightCorner)
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


__all__ = ["MainWindowLayoutMixin"]
