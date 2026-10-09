"""Exporting the comparison: the table as CSV, the charts as a PDF report."""

from __future__ import annotations

import csv
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QFileDialog

from presentation.qt import message_box
from presentation.qt.comparison_export_dialog import ComparisonExportDialog
from presentation.qt.comparison_model import (
    TABLE_COLUMNS,
    VIEWS,
    VIEW_ALIGNMENT,
    VIEW_EXECUTION,
    VIEW_NEGATIVE,
    VIEW_PAIRS,
    VIEW_PERFORMANCE,
    VIEW_RANKING,
    _fmt,
)
from presentation.qt.duration import format_duration
from presentation.qt.report import REPORT_IMAGE_WIDTH, build_comparison_report_html
from presentation.qt.style import build_style_sheet


class ComparisonExportMixin:
    """Rendering the comparison out to CSV, PDF, or inline HTML."""

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def table_rows(self) -> list[list[str]]:
        """The comparison table as plain strings — header row first."""
        out = [list(TABLE_COLUMNS)]
        for rank, row in self.ranked_rows():
            out.append(
                [
                    "—" if rank is None else f"#{rank}",
                    row.method,
                    _fmt(row.accuracy), _fmt(row.precision), _fmt(row.recall),
                    _fmt(row.f1_score), _fmt(row.auc),
                    _fmt(row.mmd_before, 6), _fmt(row.mmd_after, 6),
                    "—" if row.mmd_change is None else f"{row.mmd_change:+.2f}%",
                    row.alignment_status or "—",
                    "—" if row.overlap_before is None else f"{row.overlap_before:.1f}%",
                    "—" if row.overlap_after is None else f"{row.overlap_after:.1f}%",
                    "—" if row.overlap_change is None else f"{row.overlap_change:+.1f}",
                    format_duration(row.execution_seconds) or "—",
                    _fmt(row.baseline_accuracy),
                    "—" if row.accuracy_delta is None else f"{row.accuracy_delta:+.4f}",
                    "Yes" if row.is_negative_transfer else ("—" if row.accuracy_delta is None else "No"),
                    _fmt(self._overall.get(row.method), 2),
                ]
            )
        return out

    def context(self) -> dict[str, str]:
        """What the current chart is showing, for the report header."""
        return {
            "Source domain": self._source_combo.currentText() or "—",
            "Target domain": self._target_combo.currentText() or "—",
            "View": self._view_combo.currentText(),
            "Metric": self._metric_label(),
            "Sort": self._sort_combo.currentText(),
            "Baseline included": "Yes" if self._baseline_check.isChecked() else "No",
        }

    def render_to_base64_html(self) -> str:
        """Current chart as an inline PNG for the PDF report.

        See ``REPORT_IMAGE_WIDTH`` for why this is a ``width`` attribute
        rather than a CSS style — Qt's print engine ignores percentage/
        max-width styling on ``<img>`` and renders it at native pixel size,
        which is wider than a printed page and gets clipped at the margin.
        """
        import base64
        import io

        if not self._rows:
            return ""
        buffer = io.BytesIO()
        self._figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("ascii")
        return (
            f"<img src='data:image/png;base64,{encoded}' "
            f"width='{REPORT_IMAGE_WIDTH}' style='margin-top:10px;' />"
        )

    def available_views(self) -> list[str]:
        """Views the stored data can actually support for the current pair.

        Drives which checkboxes the export dialog offers — exporting a page
        that just says "no stored MMD values" helps nobody.
        """
        rows = self.visible_rows()
        if not rows:
            return []
        available = [VIEW_PERFORMANCE, VIEW_RANKING, VIEW_PAIRS]
        if any(r.mmd_before is not None and r.mmd_after is not None for r in rows):
            available.append(VIEW_ALIGNMENT)
        if any(r.execution_seconds is not None for r in rows):
            available.append(VIEW_EXECUTION)
        if any(r.baseline_accuracy is not None for r in rows):
            available.append(VIEW_NEGATIVE)
        return [view for view in VIEWS if view in available]

    def _render_views_to_html(self, views: list[str]) -> list[tuple[str, str]]:
        """Draw each requested view once and capture it as inline PNG markup.

        The visible selection is restored afterwards: exporting should not
        leave the user looking at a different chart than they started on.
        """
        original = self._view_combo.currentText()
        captured: list[tuple[str, str]] = []
        try:
            for view in views:
                self._view_combo.blockSignals(True)
                self._view_combo.setCurrentText(view)
                self._view_combo.blockSignals(False)
                self._render()
                captured.append((view, self.render_to_base64_html()))
        finally:
            self._view_combo.blockSignals(True)
            self._view_combo.setCurrentText(original)
            self._view_combo.blockSignals(False)
            self._render()
        return captured

    def _export_pdf(self) -> None:
        if not self.visible_rows():
            message_box.warn(
                self,
                "Nothing to Compare",
                "No stored results for this source-target pair.",
                "Run an experiment on this pair while logged in, then try again.",
            )
            return

        chooser = ComparisonExportDialog(
            build_style_sheet(self._palette), self.available_views(), parent=self
        )
        if chooser.exec_() != QDialog.Accepted:
            return
        views = chooser.selected_views()
        include_table = chooser.include_table()

        source = self._source_combo.currentText() or "results"
        target = self._target_combo.currentText() or "comparison"
        dialog = QFileDialog(
            self,
            "Export Comparison Report",
            f"comparison_{source}_to_{target}.pdf",
            "PDF Files (*.pdf)",
        )
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("pdf")
        dialog.setStyleSheet(build_style_sheet(self._palette))
        if dialog.exec_() != QDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        path = selected[0]
        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        html = build_comparison_report_html(
            self.context(),
            self._render_views_to_html(views),
            self.table_rows() if include_table else [],
            self._current_user.name if self._current_user else "Not signed in",
            include_table=include_table,
        )
        try:
            from PyQt5.QtGui import QTextDocument
            from PyQt5.QtPrintSupport import QPrinter

            document = QTextDocument()
            document.setHtml(html)
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(path)
            printer.setPageMargins(15, 15, 15, 15, QPrinter.Millimeter)
            document.print_(printer)
        except Exception as exc:  # noqa: BLE001 - raw text goes to the details pane
            message_box.error(
                self,
                "Export Failed",
                "The comparison report couldn't be saved.",
                f"Nothing was written to {Path(path).name}. Check that the folder "
                "is writable and that the file isn't open in another app.",
                technical=str(exc),
            )
            return
        message_box.success(
            self,
            "Report Exported",
            "Your comparison report is ready.",
            f"Saved as {Path(path).name} in {Path(path).parent}.",
        )

    def _export_csv(self) -> None:
        rows = self.table_rows()
        if len(rows) < 2:
            return
        source = self._source_combo.currentText() or "results"
        target = self._target_combo.currentText() or "comparison"
        dialog = QFileDialog(
            self, "Export Comparison CSV", f"comparison_{source}_to_{target}.csv",
            "CSV Files (*.csv)",
        )
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("csv")
        dialog.setStyleSheet(build_style_sheet(self._palette))
        from PyQt5.QtWidgets import QDialog

        if dialog.exec_() != QDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        path = selected[0]
        if not path.lower().endswith(".csv"):
            path += ".csv"

        from presentation.qt import message_box

        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([f"# {key}: {value}" for key, value in self.context().items()])
                writer.writerows(rows)
        except OSError as exc:
            message_box.error(
                self,
                "Export Failed",
                "The CSV couldn't be saved.",
                f"Nothing was written to {Path(path).name}. Check that the folder "
                "is writable and that the file isn't open in another app.",
                technical=str(exc),
            )
            return
        message_box.success(
            self,
            "CSV Exported",
            "Your comparison table is ready.",
            f"Saved as {Path(path).name} in {Path(path).parent}.",
        )


__all__ = [
    "ComparisonTab",
    "MethodRow",
    "rows_from_entries",
    "METRICS",
    "VIEWS",
    "VIEW_PERFORMANCE",
    "VIEW_RANKING",
    "VIEW_ALIGNMENT",
    "VIEW_EXECUTION",
    "VIEW_NEGATIVE",
    "VIEW_PAIRS",
    "SORT_OPTIONS",
    "SORT_METRIC_ASC",
    "SORT_METRIC_DESC",
    "SORT_NAME",
    "SORT_TIME",
    "TABLE_COLUMNS",
    "BASELINE_LABEL",
]


__all__ = ["ComparisonExportMixin"]
