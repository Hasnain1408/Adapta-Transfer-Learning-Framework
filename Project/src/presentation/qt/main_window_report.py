"""PDF report export for ``MainWindow``.

Gathers the current run's configuration from the sidebar widgets, hands it
to the pure ``report.build_report_html`` along with the last result and the
rendered chart markup, then prints the resulting HTML to a PDF via
``QTextDocument``/``QPrinter``.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import QDialog, QFileDialog

from presentation.qt import message_box
from presentation.qt.report import build_report_html
from presentation.qt.style import build_style_sheet
from presentation.qt.title_bar import install_title_bar_on_grid


class MainWindowReportMixin:
    """PDF export of the last completed single-run result."""

    def _report_config_rows(self) -> list[tuple[str, str]]:
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
        return config_rows

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
            html = build_report_html(
                result,
                self._report_config_rows(),
                self._palette,
                self._charts_widget.render_to_base64_html(),
            )
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


__all__ = ["MainWindowReportMixin"]
