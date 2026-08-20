"""Clearing the workspace back to a fresh state.

Resets what is on screen — the configuration inputs, every results surface,
and the dynamically opened detail tabs — without touching the signed-in
user, the theme, or anything already persisted. Clearing the desk is not the
same as deleting past work.
"""

from __future__ import annotations

from presentation.qt import message_box


class MainWindowWorkspaceMixin:
    """The Clear action and the resets behind it."""

    def _clear_all(self) -> None:
        """Reset the whole workspace: configuration, results, and run state.

        Deliberately *not* touched: the signed-in user, the theme, and
        anything already persisted to the database — clearing the desk is
        not the same as deleting past work, and the History tab keeps
        showing it.
        """
        if self._reject_if_busy("clearing the workspace"):
            return

        confirmed = message_box.confirm(
            self,
            "Clear Workspace",
            "Reset the settings and discard the results shown?",
            "Experiments already saved to History are kept — this only clears "
            "what's on screen.",
            confirm_label="Clear",
        )
        if not confirmed:
            return

        # State first: tearing down the results widgets re-emits selection
        # signals, and the handlers behind them read exactly these fields.
        self._last_result = None
        self._last_request = None
        self._last_batch_result = None
        self._last_batch_request = None
        self._last_multi_result = None
        self._last_multi_request = None
        self._last_run_kind = None

        self._reset_configuration()
        self._reset_results()

        self._set_status("Ready", "neutral")
        self._update_context_label()

    def _reset_configuration(self) -> None:
        """Every input back to the value it had at startup."""
        self._source_input.clear()
        self._target_input.clear()
        self._clear_batch_files()
        # A cleared workspace is about to be configured again, so the fields
        # that configure it come back regardless of how the last run left them.
        self._set_datasets_collapsed(False)

        # Unchecking these fires _on_batch_mode_toggled / _on_combined_file_toggled,
        # which is what re-enables the source/target fields the two modes hide.
        self._batch_mode_check.setChecked(False)
        self._combined_file_check.setChecked(False)
        # Also fires _on_multi_method_toggled, which puts the method combo
        # back in place of the checkable list.
        self._multi_method_check.setChecked(False)
        self._set_all_methods_checked(False)

        for combo, index in (
            (self._method_combo, 0),
            (self._classifier_combo, 0),
            (self._kernel_combo, 1),  # Linear, matching _build_widgets
            (self._norm_combo, 0),
            (self._std_combo, 0),
        ):
            combo.setCurrentIndex(index)

        self._reg_input.setValue(1.0)
        self._dim_input.setValue(10)
        self._mu_input.setValue(1.0)
        # Re-greys the controls the freshly selected method ignores.
        self._sync_method_parameters()

    def _reset_results(self) -> None:
        """Empty every results surface and disable what acts on a result."""
        self._close_detail_tabs()

        self._show_metrics(None)
        self._counts_label.setText("Run an experiment to evaluate the target domain.")

        self._predictions_table.clearContents()
        self._predictions_table.setRowCount(0)
        for check in (self._incorrect_only_check, self._correct_only_check):
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)
            check.setEnabled(False)

        self._features_table.clearContents()
        self._features_table.setRowCount(0)
        self._features_table.setColumnCount(0)
        self._features_caption.hide()

        # Emptying the table churns the selection, and this table's
        # itemSelectionChanged opens a pair-detail tab — which would
        # immediately undo the _close_detail_tabs() above.
        self._batch_results_table.blockSignals(True)
        self._batch_results_table.clearContents()
        self._batch_results_table.setRowCount(0)
        self._batch_results_table.clearSelection()
        self._batch_results_table.setCurrentCell(-1, -1)
        self._batch_results_table.blockSignals(False)
        self._batch_summary_label.setText(
            "Run a batch to see per-pair results and the average across the batch."
        )

        # Same signal hazard as the batch table above: this one's
        # itemSelectionChanged opens a method-detail tab.
        self._method_results_table.blockSignals(True)
        self._method_results_table.clearContents()
        self._method_results_table.setRowCount(0)
        self._method_results_table.clearSelection()
        self._method_results_table.setCurrentCell(-1, -1)
        self._method_results_table.blockSignals(False)
        self._method_summary_label.setText(
            "Tick several methods in the sidebar to compare them on one dataset pair."
        )

        self._summary_view.clear()
        self._charts_widget.clear()
        self._visualization_tab.clear()
        # The panel outlives a run on purpose (see _on_success), so a fresh
        # workspace is the one place it has to be taken down explicitly.
        self._processing_strip.clear()
        self._fit_summary_pane()

        self._export_button.setEnabled(False)
        self._export_action.setEnabled(False)
        self._batch_export_button.setEnabled(False)
        self._method_export_button.setEnabled(False)
        self._results_fullscreen_button.setEnabled(False)
        # Back to the "No results yet" page of the stack.
        self._results_stack.setCurrentIndex(0)

    def _close_detail_tabs(self) -> None:
        """Drop every dynamically opened experiment/pair detail tab.

        They describe results that are being discarded, so leaving them
        open would strand the user on a tab about data the rest of the
        window no longer has.
        """
        for registry in (
            self._experiment_detail_tabs,
            self._pair_detail_tabs,
            self._method_detail_tabs,
        ):
            for view in list(registry.values()):
                index = self._results_tabs.indexOf(view)
                if index != -1:
                    self._results_tabs.removeTab(index)
                view.deleteLater()
            registry.clear()


__all__ = ["MainWindowWorkspaceMixin"]
