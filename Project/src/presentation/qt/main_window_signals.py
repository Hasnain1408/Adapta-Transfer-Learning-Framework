"""Connecting every widget and action to its handler.

Deliberately one place: the handlers themselves are spread across the
behaviour mixins, so this file is the index that says which control
reaches which of them.
"""

from __future__ import annotations


class MainWindowSignalsMixin:
    """Connecting every widget and action to its handler."""

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
        self._home_button.clicked.connect(self._show_landing)
        self._home_action.triggered.connect(self._show_landing)

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


__all__ = ["MainWindowSignalsMixin"]
