"""PDF report export for ``MainWindow``.

Every export path (a live single run, a live batch, a replayed history
entry, or one pair opened from a batch's detail tab) funnels through
:meth:`_write_pdf_report`: pick sections in ``ExportOptionsDialog``, build
the HTML with the matching function in ``report.py``, then print it via
``QTextDocument``/``QPrinter``. The four ``_export_*`` methods only differ
in what they hand to ``report.py`` — none of them touch Qt printing directly.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import QDialog, QFileDialog

from domain.entities import DatasetRecord, ExperimentHistoryEntry
from domain.models import PairTransferLearningResult
from presentation.qt import message_box
from presentation.qt.export_options_dialog import ExportOptionsDialog
from presentation.qt.main_window_ui import KERNEL_OPTIONS
from presentation.qt.report import (
    build_batch_report_html,
    build_history_report_html,
    build_report_html,
)
from presentation.qt.style import build_style_sheet
from presentation.qt.title_bar import install_title_bar_on_grid

_KERNEL_LABELS = {value: label for label, value in KERNEL_OPTIONS}


class MainWindowReportMixin:
    """PDF export for live runs, batch runs, and replayed history entries."""

    # ------------------------------------------------------------------
    # Config-row builders
    # ------------------------------------------------------------------
    def _config_rows_common(
        self,
        method_name: str,
        classifier: str,
        normalization: str,
        standardization: str,
        kernel_type: str,
        regularization: float,
        latent_dim: int,
        mu: float,
    ) -> list[tuple[str, str]]:
        supported = self._use_case.supported_parameters(method_name)
        rows = [
            ("Method", method_name),
            ("Classifier", classifier),
            ("Normalization", normalization),
            ("Standardization", standardization),
        ]
        if "kernel_type" in supported:
            rows.append(("Kernel", _KERNEL_LABELS.get(kernel_type, kernel_type)))
        if "regularization" in supported:
            rows.append(("Regularization (lambda)", f"{regularization:g}"))
        if "latent_dim" in supported:
            rows.append(("Latent dimensions", str(latent_dim)))
        if "mu" in supported:
            rows.append(("Mu", f"{mu:g}"))
        return rows

    def _report_config_rows(self) -> list[tuple[str, str]]:
        request = self._last_request
        if request is None:
            return []
        rows = self._config_rows_common(
            request.method_name,
            request.classifier,
            request.normalization,
            request.standardization,
            request.kernel_type,
            request.regularization,
            request.latent_dim,
            request.mu,
        )
        rows.insert(2, ("Source dataset", Path(request.source_path).name))
        rows.insert(3, ("Target dataset", Path(request.target_path).name))
        return rows

    def _batch_report_config_rows(self) -> list[tuple[str, str]]:
        request = self._last_batch_request
        if request is None:
            return []
        rows = self._config_rows_common(
            request.method_name,
            request.classifier,
            request.normalization,
            request.standardization,
            request.kernel_type,
            request.regularization,
            request.latent_dim,
            request.mu,
        )
        pair_count = len(request.file_paths) * (len(request.file_paths) - 1)
        rows.insert(2, ("Files in batch", str(len(request.file_paths))))
        rows.insert(3, ("Pairs run", str(pair_count)))
        return rows

    def _pair_config_rows(self, pair: PairTransferLearningResult) -> list[tuple[str, str]]:
        request = self._last_batch_request
        if request is None:
            return []
        rows = self._config_rows_common(
            request.method_name,
            request.classifier,
            request.normalization,
            request.standardization,
            request.kernel_type,
            request.regularization,
            request.latent_dim,
            request.mu,
        )
        rows.insert(2, ("Source dataset", Path(pair.source_path).name))
        rows.insert(3, ("Target dataset", Path(pair.target_path).name))
        return rows

    def _performed_by(self) -> str:
        return self._current_user.name if self._current_user else "Not signed in"

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------
    def _pick_sections(
        self,
        *,
        visualizations_available: bool,
        unavailable_reason: str = "",
        alignment_available: bool = False,
        predictions_available: bool = False,
    ) -> set[str] | None:
        dialog = ExportOptionsDialog(
            build_style_sheet(self._palette),
            visualizations_available=visualizations_available,
            unavailable_reason=unavailable_reason,
            alignment_available=alignment_available,
            predictions_available=predictions_available,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return None
        return dialog.selected_sections()

    def _write_pdf_report(self, html: str, default_filename: str) -> None:
        dialog = QFileDialog(
            self,
            "Export PDF Report",
            str(Path.cwd() / default_filename),
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
            document.setHtml(html)

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

    # ------------------------------------------------------------------
    # Export entry points
    # ------------------------------------------------------------------
    def _export_pdf_report(self) -> None:
        """Export whichever live result — single run or batch — finished most
        recently, per ``self._last_run_kind`` (set by the success handlers in
        ``main_window_run.py``)."""
        if self._last_run_kind == "batch":
            self._export_batch_report()
        else:
            self._export_single_report()

    def _export_single_report(self) -> None:
        # Rendering a report is synchronous work on the UI thread (HTML
        # build + QPrinter paint); stacking it on top of a running worker's
        # CPU-bound computation is exactly what makes the whole app feel
        # hung, so it waits its turn like every other heavy action here.
        if self._reject_if_busy("exporting a report"):
            return
        result = self._last_result
        if result is None:
            return
        sections = self._pick_sections(
            visualizations_available=True,
            alignment_available=result.alignment is not None
            or result.negative_transfer is not None,
            predictions_available=result.target_predictions is not None,
        )
        if sections is None:
            return
        html = build_report_html(
            result,
            self._report_config_rows(),
            self._palette,
            self._charts_widget.render_to_base64_html(),
            self._performed_by(),
            sections,
            feature_space_html=self._visualization_tab.render_to_base64_html(),
            feature_space_rows=self._visualization_tab.analysis_rows(),
        )
        self._write_pdf_report(html, "Transfer_Learning_Report.pdf")

    def _export_batch_report(self) -> None:
        if self._reject_if_busy("exporting a report"):
            return
        batch_result = self._last_batch_result
        if batch_result is None:
            return
        sections = self._pick_sections(
            visualizations_available=False,
            unavailable_reason=(
                "A batch has no single result to chart — open a pair from Batch "
                "Results and export its own report for visualizations."
            ),
        )
        if sections is None:
            return
        html = build_batch_report_html(
            batch_result, self._batch_report_config_rows(), self._performed_by(), sections
        )
        self._write_pdf_report(html, "Transfer_Learning_Batch_Report.pdf")

    def _export_pair_report(self, pair: PairTransferLearningResult, chart_html: str) -> None:
        """Export one batch pair's own report — used by ``PairDetailView``'s
        Export Report button, which already has its own charts widget to
        render *chart_html* from."""
        if self._reject_if_busy("exporting a report"):
            return
        if pair.result is None:
            message_box.warning(
                self, "Nothing to Export", "This pair failed and produced no result."
            )
            return
        sections = self._pick_sections(
            visualizations_available=True,
            alignment_available=pair.result.alignment is not None
            or pair.result.negative_transfer is not None,
            predictions_available=pair.result.target_predictions is not None,
        )
        if sections is None:
            return
        html = build_report_html(
            pair.result,
            self._pair_config_rows(pair),
            self._palette,
            chart_html,
            self._performed_by(),
            sections,
        )
        source_name = Path(pair.source_path).stem
        target_name = Path(pair.target_path).stem
        self._write_pdf_report(html, f"Transfer_Learning_Report_{source_name}_to_{target_name}.pdf")

    def _export_history_report(
        self,
        entry: ExperimentHistoryEntry,
        source_dataset: DatasetRecord | None,
        target_dataset: DatasetRecord | None,
    ) -> None:
        """Export a report rebuilt from a persisted experiment row — no
        per-sample predictions or charts, since neither is stored."""
        if self._reject_if_busy("exporting a report"):
            return
        sections = self._pick_sections(
            visualizations_available=False,
            unavailable_reason=(
                "Past runs only keep their configuration, dataset stats, and "
                "scores — per-sample predictions and charts aren't persisted."
            ),
        )
        if sections is None:
            return
        html = build_history_report_html(entry, source_dataset, target_dataset, sections)
        self._write_pdf_report(
            html, f"Transfer_Learning_Report_Experiment_{entry.experiment.id}.pdf"
        )


__all__ = ["MainWindowReportMixin"]
