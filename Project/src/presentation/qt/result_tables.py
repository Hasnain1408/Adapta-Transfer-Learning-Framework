"""Shared helpers for the predictions table and dataset-summary formatting.

Used by ``MainWindow``'s own results tabs, ``PairDetailView`` (the per-pair
batch detail tab), and ``DatasetSummaryView``, so a single-run result and a
batch pair's result render identically instead of copies of the same logic
drifting apart.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from domain.models import BatchTransferLearningResult, TransferLearningResult

#: Column headers for a predictions table.
PREDICTIONS_HEADERS = ["Sample", "Actual Label", "Predicted Label", "Result"]

#: Column headers for a batch results table.
BATCH_RESULTS_HEADERS = [
    "Source",
    "Target",
    "Status",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Samples",
    "Correct",
    "Wrong",
]


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


def populate_batch_results_table(
    table: QTableWidget, batch_result: BatchTransferLearningResult, palette: dict[str, str]
) -> str:
    """Fill a ``BATCH_RESULTS_HEADERS`` table and return the summary caption."""
    table.setRowCount(len(batch_result.pairs))

    success_brush = QBrush(QColor(palette["success"]))
    danger_brush = QBrush(QColor(palette["danger"]))
    succeeded = 0

    for row, pair in enumerate(batch_result.pairs):
        source_name = Path(pair.source_path).name
        target_name = Path(pair.target_path).name
        metrics = pair.result.metrics if pair.result is not None else None

        if metrics is not None:
            succeeded += 1
            status_text, status_brush = "OK", success_brush
            values = [
                f"{metrics.accuracy:.4f}",
                f"{metrics.precision:.4f}",
                f"{metrics.recall:.4f}",
                f"{metrics.f1_score:.4f}",
                str(metrics.total_samples),
                str(metrics.correct_predictions),
                str(metrics.wrong_predictions),
            ]
        elif pair.result is not None:
            status_text, status_brush = "No labels", None
            values = ["—"] * 7
        else:
            status_text, status_brush = "Error", danger_brush
            values = [pair.error or "Unknown error"] + [""] * 6

        cells = [source_name, target_name, status_text, *values]
        for column, value in enumerate(cells):
            item = QTableWidgetItem(value)
            if column >= 2:
                item.setTextAlignment(Qt.AlignCenter if column == 2 else Qt.AlignRight | Qt.AlignVCenter)
            if column == 2 and status_brush is not None:
                item.setForeground(status_brush)
            if column == 3 and status_text == "Error":
                item.setToolTip(pair.error or "")
            table.setItem(row, column, item)

    header = table.horizontalHeader()
    for column in range(2):
        header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
    for column in range(2, 10):
        header.setSectionResizeMode(column, QHeaderView.Stretch)

    total = len(batch_result.pairs)
    avg = batch_result.average_metrics
    if avg is not None:
        return (
            f"{total} pairs run — {succeeded} produced metrics, {total - succeeded} did not. "
            f"Average over evaluable pairs: accuracy {avg.accuracy:.4f} | "
            f"precision {avg.precision:.4f} | recall {avg.recall:.4f} | f1 {avg.f1_score:.4f} "
            f"({avg.correct_predictions}/{avg.total_samples} correct)"
        )
    return (
        f"{total} pairs run — no pair produced evaluable metrics "
        "(datasets may be missing labels, or every pair failed)."
    )


__all__ = [
    "BATCH_RESULTS_HEADERS",
    "PREDICTIONS_HEADERS",
    "dataset_class_names",
    "dataset_dtype_summary",
    "populate_batch_results_table",
    "populate_predictions_table",
]
