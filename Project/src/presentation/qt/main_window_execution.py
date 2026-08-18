"""Starting runs and handling what comes back.

The only module that touches ``TransferLearningWorker`` /
``BatchTransferLearningWorker`` — every experiment passes through here, off
the UI thread, per the worker-thread invariant documented on those classes.
Also owns the configuration-state helpers, since they are driven almost
entirely by run configuration and run completion.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QMessageBox

from domain.models import (
    BatchTransferLearningRequest,
    BatchTransferLearningResult,
    TransferLearningRequest,
    TransferLearningResult,
)
from domain.run_control import RunControl
from presentation.qt import message_box
from presentation.qt.duration import format_duration
from presentation.qt.result_tables import populate_batch_results_table
from presentation.qt.style import build_style_sheet
from presentation.qt.title_bar import install_title_bar_on_grid
from presentation.qt.worker import BatchTransferLearningWorker, TransferLearningWorker


class MainWindowExecutionMixin:
    """Launching single/batch runs and their success/error handlers."""

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
    # Starting a run
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Completion handlers
    # ------------------------------------------------------------------
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
        """Lives here because what it lists — the registered methods — comes
        off the same use case this module drives."""
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


__all__ = ["MainWindowExecutionMixin"]
