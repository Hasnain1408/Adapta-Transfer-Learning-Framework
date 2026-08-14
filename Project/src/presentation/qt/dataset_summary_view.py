"""Card-based Dataset Summary view, shown for both single runs and batch pairs.

Replaces the earlier flat 8-column table with a pair of "Source Dataset" /
"Target Dataset" cards: a row of headline stat tiles (reusing ``MetricCard``,
the same tile already used for accuracy/precision/recall/F1) followed by a
detail form, mirroring the "Overview" / "Dataset" group boxes already used in
``ExperimentDetailView`` so the whole app shares one visual language for
"read-only record of facts" pages.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from domain.models import Dataset, TransferLearningResult
from presentation.qt.metric_card import MetricCard
from presentation.qt.result_tables import dataset_class_names, dataset_dtype_summary
from presentation.qt.style import DARK_PALETTE

#: Accent colours for the Samples / Features / Classes / Missing Values tiles,
#: reusing the same four accents as the headline metric row for consistency.
_STAT_ACCENTS = ("teal", "purple", "amber", "pink")


class DatasetSummaryView(QWidget):
    """Read-only "Source Dataset" / "Target Dataset" card pair."""

    def __init__(self, palette: dict[str, str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 4, 4)
        content_layout.setSpacing(0)

        self._row = QHBoxLayout()
        self._row.setSpacing(16)
        content_layout.addLayout(self._row)
        content_layout.addStretch(1)
        scroll.setWidget(self._content)

        self._placeholder = QLabel("Run an experiment to see dataset details.")
        self._placeholder.setObjectName("captionLabel")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._row.addWidget(self._placeholder, 1)

    def update_result(self, result: TransferLearningResult) -> None:
        self._clear()
        self._row.addWidget(self._build_card("Source Dataset", result.source_dataset))
        self._row.addWidget(self._build_card("Target Dataset", result.target_dataset))

    def clear(self) -> None:
        """Drop both dataset cards and put the placeholder back.

        ``_clear`` deletes every widget in the row, the original placeholder
        included, so this builds a fresh one rather than re-showing it.
        """
        self._clear()
        self._placeholder = QLabel("Run an experiment to see dataset details.")
        self._placeholder.setObjectName("captionLabel")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._row.addWidget(self._placeholder, 1)

    def _clear(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Removing an item from a layout does not hide the widget —
                # it stays visible at its last geometry until actually
                # deleted, so hide it up front rather than waiting on the
                # deferred deleteLater() to take effect.
                widget.hide()
                widget.deleteLater()

    def _build_card(self, title: str, dataset: Dataset) -> QGroupBox:
        class_names = dataset_class_names(dataset.labels)
        dtype_summary = dataset_dtype_summary(dataset.dtypes)

        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(14)

        missing_tooltip = (
            "Missing/non-numeric feature cells were filled with 0.0 at load time."
            if dataset.missing_value_count
            else ""
        )
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        specs = [
            ("Samples", str(dataset.features.shape[0]), ""),
            ("Features", str(dataset.features.shape[1]), ""),
            ("Classes", str(len(class_names)), ""),
            ("Missing Values", str(dataset.missing_value_count), missing_tooltip),
        ]
        for (stat_title, value, tooltip), accent in zip(specs, _STAT_ACCENTS):
            card = MetricCard(stat_title, tooltip=tooltip, accent=accent)
            card.set_value(value)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            stats_row.addWidget(card)
        layout.addLayout(stats_row)

        form = QFormLayout()
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.addRow(self._label("Label Column"), self._value(dataset.label_name or "—"))
        form.addRow(
            self._label("Class Names"),
            self._value(", ".join(class_names) if class_names else "—"),
        )

        dtype_value = self._value(dtype_summary or "—")
        if dataset.dtypes:
            dtype_value.setToolTip(", ".join(dataset.dtypes))
        form.addRow(self._label("Data Types"), dtype_value)

        form.addRow(
            self._label("Feature Names"), self._value(", ".join(dataset.feature_names) or "—")
        )
        layout.addLayout(form)
        layout.addStretch(1)
        return group

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _value(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        return label


__all__ = ["DatasetSummaryView"]
