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

from presentation.qt.duration import format_duration
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
    "Time",
]


#: Column headers for the per-method results of a multi-method run.
#: "vs Baseline" is the accuracy delta against the *shared* no-adaptation
#: baseline — the one figure that says whether a method helped at all,
#: which a raw accuracy column cannot.
METHOD_RESULTS_HEADERS = [
    "Rank",
    "Method",
    "Status",
    "Avg Rank",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "AUC",
    "vs Baseline",
    "MMD Δ",
    "Overlap Δ",
    "Adapt Time",
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


def _pair_time(pair) -> str:
    """One pair's adaptation time — the method's own cost, excluding the
    file loading and classifier fitting that surround it."""
    if pair.result is None:
        return ""
    return format_duration(pair.result.adaptation_seconds)


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
                _pair_time(pair),
            ]
        elif pair.result is not None:
            status_text, status_brush = "No labels", None
            values = ["—"] * 7 + [_pair_time(pair)]
        else:
            status_text, status_brush = "Error", danger_brush
            # A failed pair has no result to time.
            values = [pair.error or "Unknown error"] + [""] * 7

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
    for column in range(2, len(BATCH_RESULTS_HEADERS)):
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


def _method_row_values(entry, baseline, average_rank) -> tuple[list[str], str]:
    """The scored cells for one method, plus its status word."""
    metrics = entry.metrics
    if metrics is None and entry.result is None:
        return ([entry.error or "Unknown error"] + [""] * 9, "Error")
    if metrics is None:
        return (["—"] * 10, "No labels")

    result = entry.result
    alignment = result.alignment
    negative = result.negative_transfer

    if negative is not None:
        delta = f"{negative.accuracy_delta:+.4f}"
    elif baseline is not None:
        delta = f"{metrics.accuracy - baseline.accuracy:+.4f}"
    else:
        delta = "—"

    return (
        [
            f"{average_rank:.2f}" if average_rank is not None else "—",
            f"{metrics.accuracy:.4f}",
            f"{metrics.precision:.4f}",
            f"{metrics.recall:.4f}",
            f"{metrics.f1_score:.4f}",
            f"{metrics.auc:.4f}" if metrics.auc is not None else "—",
            delta,
            f"{alignment.change_percent:+.1f}%" if alignment is not None else "—",
            f"{alignment.overlap_change_points:+.1f}"
            if alignment is not None and alignment.overlap_change_points is not None
            else "—",
            format_duration(result.adaptation_seconds),
        ],
        "OK",
    )


def populate_method_results_table(
    table: QTableWidget, multi_result, palette: dict[str, str]
) -> str:
    """Fill a ``METHOD_RESULTS_HEADERS`` table and return the summary caption.

    Rows come out in rank order rather than the order the methods ran, so
    the table answers "which won" by its shape rather than by making the
    reader scan a column. Methods that produced no metrics are unrankable
    and sink to the bottom with a blank rank.
    """
    ranked = multi_result.ranked()
    rank_by_name = {entry.method_name: position for position, entry in enumerate(ranked, start=1)}
    unranked = [entry for entry in multi_result.methods if entry.method_name not in rank_by_name]
    ordered = ranked + unranked

    table.setRowCount(len(ordered))
    success_brush = QBrush(QColor(palette["success"]))
    danger_brush = QBrush(QColor(palette["danger"]))
    winner_brush = QBrush(QColor(palette["accent"]))
    baseline = multi_result.baseline
    average_ranks = multi_result.method_ranks()

    for row, entry in enumerate(ordered):
        values, status_text = _method_row_values(
            entry, baseline, average_ranks.get(entry.method_name)
        )
        rank = rank_by_name.get(entry.method_name)
        cells = ["" if rank is None else str(rank), entry.method_name, status_text, *values]

        for column, value in enumerate(cells):
            item = QTableWidgetItem(value)
            if column == 0 or column == 2:
                item.setTextAlignment(Qt.AlignCenter)
            elif column > 2:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if column == 2:
                if status_text == "OK":
                    item.setForeground(success_brush)
                elif status_text == "Error":
                    item.setForeground(danger_brush)
            if column == 3 and status_text == "Error":
                item.setToolTip(entry.error or "")
            # The winner is marked on the row itself rather than only by
            # sitting first: the table is sortable by click, and a colour
            # that survives a re-sort is what keeps the answer readable.
            if rank == 1 and column in (0, 1):
                item.setForeground(winner_brush)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            table.setItem(row, column, item)

    # ResizeToContents, not Stretch: thirteen columns stretched into the
    # available width clip their own headers ("Precision" became "recision",
    # "Avg Rank" became "vg Rank"), and a column whose name is unreadable is
    # worse than one you have to scroll to. The last section is un-stretched
    # for the same reason.
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    for column in range(len(METHOD_RESULTS_HEADERS)):
        header.setSectionResizeMode(column, QHeaderView.ResizeToContents)

    return method_results_caption(multi_result)


def method_results_caption(multi_result) -> str:
    """One line summarising who won and against what."""
    total = len(multi_result.methods)
    evaluated = multi_result.evaluated
    best = multi_result.best

    if best is None:
        return (
            f"{total} method(s) run — none produced evaluable metrics "
            "(the datasets may be missing labels, or every method failed)."
        )

    parts = [
        f"{len(evaluated)} of {total} method(s) comparable  ·  "
        f"best: {best.method_name} (accuracy {best.metrics.accuracy:.4f}, "
        f"F1 {best.metrics.f1_score:.4f})"
    ]
    baseline = multi_result.baseline
    if baseline is not None:
        delta = best.metrics.accuracy - baseline.accuracy
        verdict = "above" if delta >= 0 else "below"
        parts.append(
            f"{abs(delta):.4f} {verdict} the shared no-adaptation baseline "
            f"({baseline.accuracy:.4f})"
        )
    if len(evaluated) > 1:
        accuracies = [entry.metrics.accuracy for entry in evaluated]
        parts.append(f"spread {max(accuracies) - min(accuracies):.4f} across the compared methods")
    return "  ·  ".join(parts)


__all__ = [
    "BATCH_RESULTS_HEADERS",
    "METHOD_RESULTS_HEADERS",
    "PREDICTIONS_HEADERS",
    "method_results_caption",
    "populate_method_results_table",
    "dataset_class_names",
    "dataset_dtype_summary",
    "populate_batch_results_table",
    "populate_predictions_table",
]
