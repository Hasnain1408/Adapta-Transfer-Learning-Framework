"""The "which views to export" dialog shown before writing a comparison PDF."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from presentation.qt import message_box
from presentation.qt.comparison_model import VIEWS
from presentation.qt.title_bar import install_title_bar


class ComparisonExportDialog(QDialog):
    """Pick which comparison views (and the table) go into the PDF.

    Separate from ``ExportOptionsDialog``: that one gates the sections of a
    *single run's* report, while these are the comparison views, which vary
    with what the database actually holds.
    """

    def __init__(self, stylesheet: str, available: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(stylesheet)
        self.setMinimumWidth(380)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        intro = QLabel("Choose which comparisons to include in the exported report.")
        intro.setObjectName("captionLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._boxes: dict[str, QCheckBox] = {}
        for view in VIEWS:
            box = QCheckBox(view)
            box.setChecked(view in available)
            if view not in available:
                box.setEnabled(False)
                box.setToolTip("No stored data supports this view for the current pair.")
            layout.addWidget(box)
            self._boxes[view] = box

        self._table_box = QCheckBox("Exact-values table")
        self._table_box.setChecked(True)
        layout.addWidget(self._table_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export…")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addSpacing(4)
        layout.addWidget(buttons)

        install_title_bar(self, "Export Comparison Report", body)

    def _validate_and_accept(self) -> None:
        if not self.selected_views() and not self.include_table():
            message_box.warn(
                self,
                "Nothing Selected",
                "Pick at least one chart or the table to export.",
                "An empty report would contain only the title page.",
            )
            return
        self.accept()

    def selected_views(self) -> list[str]:
        return [view for view, box in self._boxes.items() if box.isChecked()]

    def include_table(self) -> bool:
        return self._table_box.isChecked()


__all__ = ["ComparisonExportDialog"]
