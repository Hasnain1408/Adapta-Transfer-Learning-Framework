"""Section picker shown before every PDF export.

One dialog, reused for a single-run report, a batch report, and a report
regenerated from history — which sections are actually offered varies by
caller (a history entry has no live chart data to draw from, for instance),
but the "pick what goes in the PDF" interaction is identical everywhere.
"""

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
from presentation.qt.title_bar import install_title_bar

#: Stable keys shared with ``report.py``'s ``sections`` parameter.
SECTION_CONFIGURATION = "configuration"
SECTION_DATASET = "dataset"
SECTION_EVALUATION = "evaluation"
SECTION_VISUALIZATIONS = "visualizations"

_SECTION_LABELS: list[tuple[str, str]] = [
    (SECTION_CONFIGURATION, "Experiment configuration"),
    (SECTION_DATASET, "Dataset properties"),
    (SECTION_EVALUATION, "Evaluation results"),
    (SECTION_VISUALIZATIONS, "Visualizations"),
]


class ExportOptionsDialog(QDialog):
    """Modal checkbox picker for which report sections to include.

    Call :meth:`selected_sections` after ``exec_()`` returns ``Accepted``;
    it is a ``set`` of the ``SECTION_*`` keys above, ready to pass straight
    to ``report.build_report_html`` (and friends) as ``sections``.
    """

    def __init__(
        self,
        stylesheet: str,
        *,
        visualizations_available: bool = True,
        unavailable_reason: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(stylesheet)
        self.setMinimumWidth(360)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        intro = QLabel("Choose which sections to include in the exported report.")
        intro.setObjectName("captionLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._checkboxes: dict[str, QCheckBox] = {}
        for key, label in _SECTION_LABELS:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            if key == SECTION_VISUALIZATIONS and not visualizations_available:
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
                checkbox.setToolTip(
                    unavailable_reason or "Visualizations are not available for this report."
                )
            layout.addWidget(checkbox)
            self._checkboxes[key] = checkbox

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export…")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addSpacing(4)
        layout.addWidget(buttons)

        install_title_bar(self, "Export Report", body)

    def _validate_and_accept(self) -> None:
        if not any(box.isChecked() for box in self._checkboxes.values()):
            message_box.warning(self, "Nothing Selected", "Select at least one section to export.")
            return
        self.accept()

    def selected_sections(self) -> set[str]:
        return {key for key, box in self._checkboxes.items() if box.isChecked()}


__all__ = [
    "ExportOptionsDialog",
    "SECTION_CONFIGURATION",
    "SECTION_DATASET",
    "SECTION_EVALUATION",
    "SECTION_VISUALIZATIONS",
]
