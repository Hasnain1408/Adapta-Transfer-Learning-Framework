"""Dataset/batch-file picking and run orchestration for ``MainWindow``.

The only mixin that touches ``TransferLearningWorker`` /
``BatchTransferLearningWorker`` — every experiment run passes through here,
off the UI thread, per the worker-thread invariant documented on those
classes. Also owns the small state-sync helpers (``_set_status``,
``_update_context_label``, ``_sync_method_parameters``) since they're driven
almost entirely by run configuration and run completion.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QDialog, QFileDialog, QLineEdit, QMessageBox

from domain.models import (
    BatchTransferLearningRequest,
    BatchTransferLearningResult,
    TransferLearningRequest,
    TransferLearningResult,
)
from domain.progress import ProgressUpdate
from domain.run_control import RunControl
from presentation.qt import message_box
from presentation.qt.duration import format_duration
from presentation.qt.main_window_ui import DATASET_FILTER
from presentation.qt.result_tables import populate_batch_results_table
from presentation.qt.style import build_style_sheet
from presentation.qt.title_bar import install_title_bar_on_grid
from presentation.qt.worker import BatchTransferLearningWorker, TransferLearningWorker


class MainWindowRunMixin:
    """Dataset picking, batch-file management, and run orchestration."""

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

    @staticmethod
    def _timing_note(result: TransferLearningResult) -> str:
        """"completed in 1.20s (adaptation 0.043s)" for the metrics caption.

        Both figures are shown because only the second is comparable across
        methods — the first also carries file loading and classifier fitting,
        which cost the same whichever method was chosen.
        """
        total = format_duration(result.duration_seconds)
        if not total:
            return ""
        adaptation = format_duration(result.adaptation_seconds)
        return (
            f"completed in {total} (adaptation {adaptation})"
            if adaptation
            else f"completed in {total}"
        )

    # ------------------------------------------------------------------
    # Live processing status
    # ------------------------------------------------------------------
    def _on_progress(self, update: ProgressUpdate) -> None:
        """Relay one worker-thread progress report to the UI.

        Arrives on the UI thread: ``RunControl`` calls the worker's
        ``progress.emit`` from the worker thread, and Qt queues the slot
        because the worker object's affinity is this thread. Nothing here
        may block — it runs between paint events.

        A pause is not overwritten: ``_toggle_pause`` has already put
        "Paused" on the status bar, and a report that was in flight when
        the user hit pause would otherwise flip it straight back to
        "Running".
        """
        self._processing_strip.apply(update)
        if self._run_control is not None and self._run_control.is_paused:
            return

        self._set_status(update.headline, "warning")

        # A batch knows how many pairs are left, so its bar can be honest
        # about how far along it is; a single run genuinely does not know
        # how long an eigendecomposition will take, and a fabricated
        # percentage would be worse than the indeterminate sweep.
        fraction = update.fraction if update.pair_total > 0 else None
        if fraction is None:
            if self._progress_bar.maximum() != 0:
                self._progress_bar.setRange(0, 0)
        else:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(int(round(fraction * 100)))

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
    # Actions
    # ------------------------------------------------------------------
    def _pick_source(self) -> None:
        self._pick_file(self._source_input, "Select Source Dataset")

    def _pick_target(self) -> None:
        self._pick_file(self._target_input, "Select Target Dataset")

    def _swap_datasets(self) -> None:
        if self._combined_file_check.isChecked():
            # Nothing to swap: the target field is hidden and unused in
            # combined-file mode. Guarded here too since the Ctrl+Shift+S
            # shortcut and menu action stay reachable even while it's hidden.
            return
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
    # Combined source/target file
    # ------------------------------------------------------------------
    def _on_combined_file_toggled(self, checked: bool) -> None:
        self._source_field_label.setText("Dataset File" if checked else "Source")
        self._target_field_label.setVisible(not checked)
        self._target_input.setVisible(not checked)
        self._swap_button.setVisible(not checked)

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

    # ------------------------------------------------------------------
    # Busy guard
    # ------------------------------------------------------------------
    def _is_worker_busy(self) -> bool:
        """Whether a background run is currently in flight.

        Also clears a stale ``self._worker`` left over from a QThread whose
        C++ object was already deleted by ``deleteLater`` — the Python
        wrapper raises ``RuntimeError`` on any further method call once
        that happens, so this is the one safe place to check it.
        """
        try:
            return self._worker is not None and self._worker.isRunning()
        except RuntimeError:
            self._worker = None
            return False

    def _busy_description(self) -> str:
        """Name whatever background run is in flight, for the message
        ``_reject_if_busy`` shows. Reads the batch checkbox rather than
        ``_last_run_kind`` because it must describe the run *in progress*,
        and the checkbox is disabled (not reset) for the run's duration —
        it still reflects what was actually started."""
        if self._batch_mode_check.isChecked():
            return "The batch experiment"
        request = self._last_request
        return f"The {request.method_name} experiment" if request is not None else "The experiment"

    def _reject_if_busy(self, action: str) -> bool:
        """Block *action* while a run is in flight, telling the user why.

        Every entry point that does meaningful synchronous work on the UI
        thread — a database round-trip, PDF rendering, starting a second
        run — calls this first. Stacking that on top of the worker
        thread's CPU-bound computation is what turns "a bit slow" into
        "the whole app looks hung" (and, under enough load, can make the
        rest of the desktop feel unresponsive too); saying so explicitly
        beats a silent no-op the user is left to guess at.
        """
        if not self._is_worker_busy():
            return False
        message_box.info(
            self,
            "Experiment Running",
            f"{self._busy_description()} is still running.",
            f"Wait for it to finish before {action}, or press Cancel to stop it.",
        )
        return True

    def _run_transfer_learning(self) -> None:
        if self._reject_if_busy("starting another experiment"):
            return

        if self._batch_mode_check.isChecked():
            self._run_batch_experiment()
        else:
            self._run_single_experiment()

    def _run_single_experiment(self) -> None:
        source_path = self._source_input.text().strip()
        combined = self._combined_file_check.isChecked()
        target_path = source_path if combined else self._target_input.text().strip()

        if not source_path or (not combined and not target_path):
            message_box.warn(
                self,
                "Dataset Needed",
                "Pick a dataset file to run."
                if combined
                else "Pick a source and a target dataset to run.",
                "Click a dataset field above to browse for a file.",
            )
            return

        missing = [p for p in {source_path, target_path} if not Path(p).is_file()]
        if missing:
            message_box.warn(
                self,
                "Dataset Not Found",
                "One of the selected files couldn't be opened.",
                "\n".join(missing)
                + "\n\nIt may have been moved, renamed, or deleted since you picked it.",
            )
            return

        if combined and not self._use_case.dataset_has_combined_domains(source_path):
            message_box.warn(
                self,
                "Not a Combined Dataset",
                f"{Path(source_path).name} doesn't contain both domains.",
                "A combined file needs a source and a target section side by side "
                "(keys like X_src/Y_src alongside X_tar/Y_tar). Untick "
                "“Source and target are combined in one file” and choose "
                "two separate files instead.",
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
            mu=self._mu_input.value(),
            classifier=self._classifier_combo.currentData(),
        )

        # Kept so the completion handlers can record what was actually run,
        # even if the user edits the form while the worker is still going.
        self._last_request = request

        self._set_running(True)
        self._set_status("Starting…", "warning")
        # In combined-file mode both halves come from the same file, so
        # "a.mat → a.mat" would just look like a bug.
        names = (
            Path(source_path).name
            if combined
            else f"{Path(source_path).name} → {Path(target_path).name}"
        )
        self._processing_strip.start(f"{request.method_name}  ·  {names}")

        self._run_control = RunControl()
        self._worker = TransferLearningWorker(self._use_case, request, control=self._run_control)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
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
            message_box.warn(
                self,
                "More Files Needed",
                "A batch needs at least two dataset files.",
                "Batch mode runs every ordered pair, so two files give two runs "
                "(A→B and B→A). Use “Add Files…” to choose more.",
            )
            return

        missing = [p for p in file_paths if not Path(p).is_file()]
        if missing:
            count = len(missing)
            message_box.warn(
                self,
                "Dataset Not Found",
                f"{count} of the batch files couldn't be opened."
                if count > 1
                else "One of the batch files couldn't be opened.",
                "\n".join(missing)
                + "\n\nRemove them from the list, or restore them and try again.",
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
            mu=self._mu_input.value(),
            classifier=self._classifier_combo.currentData(),
        )

        # Kept so the completion handler can record each pair as its own
        # experiment, even if the user edits the form while the worker runs.
        self._last_batch_request = request

        pair_count = len(file_paths) * (len(file_paths) - 1)
        self._set_running(True)
        self._set_status(f"Starting batch ({pair_count} pairs)…", "warning")
        self._processing_strip.start(f"{request.method_name}  ·  {pair_count} ordered pairs")

        self._run_control = RunControl()
        self._worker = BatchTransferLearningWorker(self._use_case, request, control=self._run_control)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_batch_success)
        self._worker.failed.connect(self._on_batch_error)
        self._worker.cancelled.connect(self._on_cancelled)
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
        self._combined_file_check.setEnabled(not running)
        self._add_batch_files_button.setEnabled(not running)
        self._remove_batch_file_button.setEnabled(not running)
        self._clear_batch_files_button.setEnabled(not running)
        self._clear_button.setEnabled(not running)

        self._pause_button.setEnabled(running)
        self._pause_button.setText("Pause")
        self._cancel_button.setEnabled(running)
        self._set_progress_bar_paused(False)

        if running:
            # Restart the cycle fresh on every run rather than continuing
            # from wherever a previous run left off.
            self._progress_color_index = 0
            self._progress_bar.setProperty("cycleColor", "0")
            self._progress_bar.style().unpolish(self._progress_bar)
            self._progress_bar.style().polish(self._progress_bar)
            self._progress_color_timer.start()
        else:
            self._progress_color_timer.stop()
            self._run_control = None

    def _set_progress_bar_paused(self, paused: bool) -> None:
        """Freeze the bar (static, no motion) and switch its colour while
        paused; restore the animated indeterminate sweep otherwise. A
        QProgressBar's built-in "busy" animation runs off its own Qt timer,
        independent of whether the underlying work is actually paused, so
        this has to be done explicitly rather than falling out of RunControl."""
        self._progress_bar.setRange(0, 1 if paused else 0)
        if paused:
            self._progress_bar.setValue(1)
        self._progress_bar.setProperty("paused", paused)
        self._progress_bar.style().unpolish(self._progress_bar)
        self._progress_bar.style().polish(self._progress_bar)

    def _advance_progress_bar_color(self) -> None:
        """Ticks once a second (see the QTimer wired up in
        MainWindowUIMixin._build_status_bar) while a run is active and not
        paused, stepping the chunk through cycleColor 0-3 so it visibly
        changes colour over time rather than sitting on one shade."""
        self._progress_color_index = (self._progress_color_index + 1) % 4
        self._progress_bar.setProperty("cycleColor", str(self._progress_color_index))
        self._progress_bar.style().unpolish(self._progress_bar)
        self._progress_bar.style().polish(self._progress_bar)

    def _toggle_pause(self) -> None:
        if self._run_control is None:
            return
        if self._run_control.is_paused:
            self._run_control.resume()
            self._pause_button.setText("Pause")
            self._set_progress_bar_paused(False)
            self._progress_color_timer.start()
            self._processing_strip.set_paused(False)
            # The stage in flight is the truest thing to show on resume; it
            # beats a generic "Running…" that the next report overwrites
            # anyway, and a stage blocked in numpy may not report for a while.
            last = self._run_control.last_update
            self._set_status(
                last.headline if last is not None else "Running…", "warning"
            )
        else:
            self._run_control.pause()
            self._pause_button.setText("Resume")
            self._set_progress_bar_paused(True)
            self._progress_color_timer.stop()
            self._processing_strip.set_paused(True)
            last = self._run_control.last_update
            self._set_status(
                f"Paused — {last.stage}" if last is not None else "Paused", "neutral"
            )

    def _cancel_run(self) -> None:
        if self._run_control is None:
            return
        self._run_control.cancel()
        # Disabled immediately so a second click can't fire while the worker
        # thread is still unwinding to its next checkpoint.
        self._cancel_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._processing_strip.set_paused(False)
        self._set_status("Cancelling…", "warning")

    def _on_cancelled(self) -> None:
        """A single-pair run was cancelled before it produced any result --
        batch cancellation instead comes back through ``_on_batch_success``
        with ``cancelled=True``, since a batch has partial results worth
        keeping. Nothing to record here: there's no run to show."""
        self._set_status("Cancelled", "neutral")
        # Left on screen rather than hidden: the stage still marked active
        # is the answer to "how far did it get before I stopped it?".
        self._processing_strip.finish("Cancelled")

    def _on_success(self, result: TransferLearningResult) -> None:
        self._last_result = result
        self._last_run_kind = "single"
        elapsed = format_duration(result.duration_seconds)
        self._set_status(f"Completed in {elapsed}" if elapsed else "Completed", "success")
        self._processing_strip.finish("Completed")
        self._export_button.setEnabled(True)
        self._export_action.setEnabled(True)

        self._show_metrics(result.metrics, self._timing_note(result))
        self._show_dataset_summary(result)
        self._show_predictions(result)
        self._show_transformed_features(result)
        self._charts_widget.update_result(result)
        request = self._last_request
        self._visualization_tab.update_result(
            result,
            method_name=request.method_name if request else "—",
            source_name=Path(request.source_path).name if request else "—",
            target_name=Path(request.target_path).name if request else "—",
        )

        self._results_fullscreen_button.setEnabled(True)

        self._results_stack.setCurrentIndex(1)
        self._update_context_label()
        self._record_run(result, error_message=None)
        # The run just added a row the comparison charts should include.
        self._comparison_tab.refresh()

    def _on_error(self, message: str) -> None:
        self._set_status("Failed", "danger")
        self._processing_strip.finish("Failed")
        # Recorded before the modal opens: exec_() blocks until dismissed, and
        # a failed configuration is worth keeping either way.
        self._record_run(None, error_message=message)
        request = self._last_request
        method = request.method_name if request is not None else "The method"
        message_box.error(
            self,
            "Experiment Failed",
            f"{method} couldn't finish this run.",
            "Nothing was saved. Check that both datasets have the same number "
            "of feature columns, then adjust the settings and run again.",
            technical=message,
        )

    def _on_batch_success(self, batch_result: BatchTransferLearningResult) -> None:
        self._last_batch_result = batch_result
        self._last_run_kind = "batch"
        succeeded = sum(1 for pair in batch_result.pairs if pair.result is not None)
        failed = len(batch_result.pairs) - succeeded
        if batch_result.cancelled:
            status = f"Batch cancelled ({len(batch_result.pairs)} pair(s) completed)"
            status_key = "neutral"
        elif failed == 0:
            status = "Batch completed"
            status_key = "success"
        else:
            status = f"Batch completed ({failed} pair(s) failed)"
            status_key = "warning"
        batch_elapsed = format_duration(batch_result.duration_seconds)
        if batch_elapsed:
            status += f" in {batch_elapsed}"
        self._set_status(status, status_key)
        self._processing_strip.finish(
            "Cancelled" if batch_result.cancelled else "Completed"
        )

        # _export_pdf_report dispatches on _last_run_kind, so this is safe to
        # leave enabled — it now builds the batch report, not a stale
        # single-pair one.
        self._export_button.setEnabled(True)
        self._export_action.setEnabled(True)
        self._batch_export_button.setEnabled(True)

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

        summary_text = populate_batch_results_table(
            self._batch_results_table, batch_result, self._palette
        )
        if batch_result.cancelled:
            summary_text = "Cancelled before the remaining pairs ran. " + summary_text
        self._batch_summary_label.setText(summary_text)
        self._results_fullscreen_button.setEnabled(True)
        self._results_stack.setCurrentIndex(1)
        if self._batch_results_tab_index >= 0:
            self._results_tabs.setCurrentIndex(self._batch_results_tab_index)
        self._record_batch_run(batch_result)
        self._comparison_tab.refresh()

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------
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

        # Unchecking these fires _on_batch_mode_toggled / _on_combined_file_toggled,
        # which is what re-enables the source/target fields the two modes hide.
        self._batch_mode_check.setChecked(False)
        self._combined_file_check.setChecked(False)

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

        self._summary_view.clear()
        self._charts_widget.clear()
        self._visualization_tab.clear()
        # The panel outlives a run on purpose (see _on_success), so a fresh
        # workspace is the one place it has to be taken down explicitly.
        self._processing_strip.clear()

        self._export_button.setEnabled(False)
        self._export_action.setEnabled(False)
        self._batch_export_button.setEnabled(False)
        self._results_fullscreen_button.setEnabled(False)
        # Back to the "No results yet" page of the stack.
        self._results_stack.setCurrentIndex(0)

    def _close_detail_tabs(self) -> None:
        """Drop every dynamically opened experiment/pair detail tab.

        They describe results that are being discarded, so leaving them
        open would strand the user on a tab about data the rest of the
        window no longer has.
        """
        for registry in (self._experiment_detail_tabs, self._pair_detail_tabs):
            for view in list(registry.values()):
                index = self._results_tabs.indexOf(view)
                if index != -1:
                    self._results_tabs.removeTab(index)
                view.deleteLater()
            registry.clear()

    def _on_batch_error(self, message: str) -> None:
        self._set_status("Batch Failed", "danger")
        self._processing_strip.finish("Failed")
        # Distinct from a single pair failing: execute_batch records those
        # per pair and still returns a result. Reaching here means the batch
        # itself could not run, so no pair produced anything.
        message_box.error(
            self,
            "Batch Failed",
            "The batch couldn't be started.",
            "No pairs were run. Check the file list and your settings, "
            "then try again.",
            technical=message,
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


__all__ = ["MainWindowRunMixin"]
