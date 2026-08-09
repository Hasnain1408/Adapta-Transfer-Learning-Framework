"""Shared helpers for the predictions table and dataset-summary formatting.

Used by ``MainWindow``'s own results tabs, ``PairDetailView`` (the per-pair
batch detail tab), and ``DatasetSummaryView``, so a single-run result and a
batch pair's result render identically instead of copies of the same logic
drifting apart.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from domain.models import TransferLearningResult

#: Column headers for a predictions table.
PREDICTIONS_HEADERS = ["Sample", "Actual Label", "Predicted Label", "Result"]


def dataset_class_names(labels) -> list[str]:
    """Distinct label values, sorted, as strings."""
    if labels is None:
        return []
    return sorted({str(value) for value in labels})


def dataset_dtype_summary(dtypes: list[str]) -> str:
    """Compact ``"float64 x24, int64 x2"``-style summary for a table cell."""
    if not dtypes:
        return ""
    counts: dict[str, int] = {}
    for dtype in dtypes:
        counts[dtype] = counts.get(dtype, 0) + 1
    return ", ".join(f"{dtype} x{count}" for dtype, count in counts.items())


def populate_predictions_table(
    table: QTableWidget, result: TransferLearningResult, palette: dict[str, str]
) -> None:
    """Fill a ``PREDICTIONS_HEADERS`` table. No-op (empty) if unlabelled."""
    table.clearContents()
    if result.target_predictions is None:
        table.setRowCount(0)
        return

    correct_brush = QBrush(QColor(palette["success"]))
    wrong_brush = QBrush(QColor(palette["danger"]))

    row_count = len(result.target_predictions)
    table.setRowCount(row_count)

    for row in range(row_count):
        is_correct = result.prediction_correctness[row] == "Right"
        cells = [
            f"Sample {row + 1}",
            str(result.target_actuals[row]),
            str(result.target_predictions[row]),
            "✓ Correct" if is_correct else "✗ Incorrect",
        ]
        for column, value in enumerate(cells):
            item = QTableWidgetItem(value)
            if column:
                item.setTextAlignment(Qt.AlignCenter)
            if column == 3:
                item.setForeground(correct_brush if is_correct else wrong_brush)
            table.setItem(row, column, item)

    header = table.horizontalHeader()
    for column in range(4):
        header.setSectionResizeMode(column, QHeaderView.Stretch)


__all__ = [
    "PREDICTIONS_HEADERS",
    "dataset_class_names",
    "dataset_dtype_summary",
    "populate_predictions_table",
]
