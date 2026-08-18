"""Report covering every pair of a batch run at once."""

from __future__ import annotations

from html import escape
from pathlib import Path

from domain.models import BatchTransferLearningResult
from presentation.qt.export_options_dialog import (
    SECTION_CONFIGURATION,
    SECTION_DATASET,
    SECTION_EVALUATION,
)
from presentation.qt.report_common import config_table, metrics_tables, page_header


def build_batch_report_html(
    batch_result: BatchTransferLearningResult,
    config_rows: list[tuple[str, str]],
    performed_by: str,
    sections: set[str],
) -> str:
    """Render every pair of a batch run as one standalone HTML report.

    No Visual Analytics section exists here regardless of *sections* — a
    batch has no single result to chart; export an individual pair's report
    from its detail tab for that.
    """
    parts = page_header(
        "Transfer Learning Batch Report",
        performed_by,
        batch_result.started_at,
        batch_result.duration_seconds,
    )

    if SECTION_CONFIGURATION in sections:
        parts += config_table(config_rows)

    if SECTION_DATASET in sections:
        parts.append("<h2>Dataset Pairs</h2><table width='100%'>")
        parts.append("<tr><th>Source</th><th>Target</th><th>Status</th></tr>")
        for pair in batch_result.pairs:
            status = "OK" if pair.result is not None else "Error"
            parts.append(
                f"<tr><td>{escape(Path(pair.source_path).name)}</td>"
                f"<td>{escape(Path(pair.target_path).name)}</td>"
                f"<td>{escape(status)}</td></tr>"
            )
        parts.append("</table>")

    if SECTION_EVALUATION in sections:
        avg = batch_result.average_metrics
        if avg is not None:
            parts.append("<h2>Average Across Batch</h2>")
            parts += metrics_tables(avg)

        parts.append("<h2>Per-Pair Results</h2><table width='100%'>")
        parts.append(
            "<tr><th>Source</th><th>Target</th><th>Status</th><th>Accuracy</th>"
            "<th>Precision</th><th>Recall</th><th>F1</th></tr>"
        )
        for pair in batch_result.pairs:
            metrics = pair.result.metrics if pair.result is not None else None
            source_name = escape(Path(pair.source_path).name)
            target_name = escape(Path(pair.target_path).name)
            if metrics is not None:
                parts.append(
                    f"<tr><td>{source_name}</td><td>{target_name}</td><td>OK</td>"
                    f"<td class='num'>{metrics.accuracy:.4f}</td>"
                    f"<td class='num'>{metrics.precision:.4f}</td>"
                    f"<td class='num'>{metrics.recall:.4f}</td>"
                    f"<td class='num'>{metrics.f1_score:.4f}</td></tr>"
                )
            elif pair.result is not None:
                parts.append(
                    f"<tr><td>{source_name}</td><td>{target_name}</td>"
                    f"<td>No labels</td><td colspan='4'>—</td></tr>"
                )
            else:
                parts.append(
                    f"<tr><td>{source_name}</td><td>{target_name}</td>"
                    f"<td>Error</td><td colspan='4'>{escape(pair.error or '')}</td></tr>"
                )
        parts.append("</table>")

    parts.append("</body></html>")
    return "".join(parts)


__all__ = ["build_batch_report_html"]
