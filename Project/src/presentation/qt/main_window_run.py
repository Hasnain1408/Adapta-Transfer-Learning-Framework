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
from domain.run_control import RunControl
from presentation.qt import message_box
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
        combined = self._combined_file_check.isChecked()
        target_path = source_path if combined else self._target_input.text().strip()

        if not source_path or (not combined and not target_path):
            message_box.warning(
                self,
                "Missing Dataset",
                "Select a dataset file before running."
                if combined
                else "Select both a source and a target dataset before running.",
            )
            return

        missing = [p for p in {source_path, target_path} if not Path(p).is_file()]
        if missing:
            message_box.warning(
                self,
                "Dataset Not Found",
                "The following path is not a readable file:\n\n" + "\n".join(missing),
            )
            return

        if combined and not self._use_case.dataset_has_combined_domains(source_path):
            message_box.warning(
                self,
                "Not a Combined Dataset",
                "This file does not appear to contain both a source and target "
                "section (expected keys like X_src/Y_src alongside X_tar/Y_tar). "
                "Uncheck \"Source and target are combined in one file\" and pick "
                "separate source/target files instead.",
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
        self._set_status("Running experiment…", "warning")

        self._run_control = RunControl()
        self._worker = TransferLearningWorker(self._use_case, request, control=self._run_control)
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
            mu=self._mu_input.value(),
            classifier=self._classifier_combo.currentData(),
        )

        # Kept so the completion handler can record each pair as its own
        # experiment, even if the user edits the form while the worker runs.
        self._last_batch_request = request

        pair_count = len(file_paths) * (len(file_paths) - 1)
        self._set_running(True)
        self._set_status(f"Running batch ({pair_count} pairs)…", "warning")

        self._run_control = RunControl()
        self._worker = BatchTransferLearningWorker(self._use_case, request, control=self._run_control)
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
            self._set_status(
                "Running batch…" if self._batch_mode_check.isChecked() else "Running experiment…",
                "warning",
            )
        else:
            self._run_control.pause()
            self._pause_button.setText("Resume")
            self._set_progress_bar_paused(True)
            self._progress_color_timer.stop()
            self._set_status("Paused", "neutral")

    def _cancel_run(self) -> None:
        if self._run_control is None:
            return
        self._run_control.cancel()
        # Disabled immediately so a second click can't fire while the worker
        # thread is still unwinding to its next checkpoint.
        self._cancel_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._set_status("Cancelling…", "warning")

    def _on_cancelled(self) -> None:
        """A single-pair run was cancelled before it produced any result --
        batch cancellation instead comes back through ``_on_batch_success``
        with ``cancelled=True``, since a batch has partial results worth
        keeping. Nothing to record here: there's no run to show."""
        self._set_status("Cancelled", "neutral")

    def _on_success(self, result: TransferLearningResult) -> None:
        self._last_result = result
        self._last_run_kind = "single"
        self._set_status("Completed", "success")
        self._export_button.setEnabled(True)
        self._export_action.setEnabled(True)

        self._show_metrics(result.metrics)
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
        self._set_status(status, status_key)

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

    def _on_batch_error(self, message: str) -> None:
        self._set_status("Batch Failed", "danger")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Batch Run Failed")
        box.setText("The batch run could not be completed.")
        box.setInformativeText(message)
        install_title_bar_on_grid(box, "Batch Run Failed")
        box.exec_()

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
