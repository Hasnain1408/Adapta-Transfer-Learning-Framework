"""Shared headline-metric tile, used by the run form and experiment detail view."""

from __future__ import annotations

from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout


class MetricCard(QFrame):
    """A single headline metric, shown as a titled value tile.

    accent selects the colour of the card's top border (see the
    QFrame#metricCard[accent=...] rules in style.py) so a row of stat
    cards reads as a set of distinct, colour-coded tiles rather than
    identical grey boxes.
    """

    def __init__(self, title: str, tooltip: str = "", accent: str = "teal") -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self.setProperty("accent", accent)
        if tooltip:
            self.setToolTip(tooltip)

        self._value_label = QLabel("—")
        self._value_label.setObjectName("metricCardValue")
        self._value_label.setProperty("placeholder", True)

        title_label = QLabel(title)
        title_label.setObjectName("metricCardTitle")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 11)
        layout.setSpacing(2)
        layout.addWidget(title_label)
        layout.addWidget(self._value_label)

    def set_value(self, text: str | None) -> None:
        is_placeholder = text is None
        self._value_label.setText("—" if is_placeholder else text)
        self._value_label.setProperty("placeholder", is_placeholder)
        # Dynamic properties only affect QSS after a re-polish.
        self._value_label.style().unpolish(self._value_label)
        self._value_label.style().polish(self._value_label)
