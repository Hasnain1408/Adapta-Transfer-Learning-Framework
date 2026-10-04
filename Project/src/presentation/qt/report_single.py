"""Report for a single source→target run, live or replayed from a batch pair."""

from __future__ import annotations

from html import escape

from domain.models import TransferLearningResult
from presentation.qt.export_options_dialog import (
    SECTION_ALIGNMENT,
    SECTION_CONFIGURATION,
    SECTION_DATASET,
    SECTION_EVALUATION,
    SECTION_PREDICTIONS,
    SECTION_VISUALIZATIONS,
)
from presentation.qt.report_common import (
    alignment_tables,
    config_table,
    metrics_tables,
    page_header,
    timing_table,
)


def build_report_html(
    result: TransferLearningResult,
    config_rows: list[tuple[str, str]],
    palette: dict[str, str],
    chart_html: str,
    performed_by: str,
    sections: set[str],
    feature_space_html: str = "",
    feature_space_rows: list[tuple[str, str]] | None = None,
) -> str:
    """Render one transfer-learning run (single or one batch pair) as a
    standalone HTML report.

    *config_rows* are the method/classifier/dataset/preprocessing settings
    shown under "Configuration". *chart_html* is pre-rendered markup (e.g.
    from AnalyticsChartsWidget.render_to_base64_html()); pass "" to omit
    the Visual Analytics section regardless of *sections*.

    *feature_space_html* / *feature_space_rows* carry the Feature Space tab's
    scatter plot and the projection/stage it was drawn with, so the reader
    knows which view of the data they are looking at.
    """
    metrics = result.metrics
    parts = page_header(
        "Transfer Learning Evaluation Report",
        performed_by,
        result.started_at,
        result.duration_seconds,
    )

    if SECTION_CONFIGURATION in sections:
        parts += config_table(config_rows)

    if SECTION_DATASET in sections:
        parts.append("<h2>Dataset Summary</h2><table width='100%'>")
        parts.append(
            "<tr><th>Dataset</th><th>Samples</th><th>Features</th><th>Label Column</th></tr>"
        )
        for role, dataset in (("Source", result.source_dataset), ("Target", result.target_dataset)):
            parts.append(
                f"<tr><td>{role}</td><td class='num'>{dataset.features.shape[0]}</td>"
                f"<td class='num'>{dataset.features.shape[1]}</td>"
                f"<td>{escape(str(dataset.label_name))}</td></tr>"
            )
        parts.append("</table>")

    if SECTION_EVALUATION in sections and metrics is not None:
        parts += metrics_tables(metrics)

    if SECTION_EVALUATION in sections:
        parts += timing_table(result)

    if SECTION_VISUALIZATIONS in sections and chart_html:
        parts.append("<h2>Visual Analytics</h2>")
        parts.append(chart_html)

    if SECTION_VISUALIZATIONS in sections and feature_space_html:
        parts.append("<h2>Feature Space Projection</h2>")
        if feature_space_rows:
            parts.append("<table width='100%'>")
            for label, value in feature_space_rows:
                parts.append(
                    f"<tr><th style='width:35%'>{escape(label)}</th>"
                    f"<td>{escape(str(value))}</td></tr>"
                )
            parts.append("</table>")
        parts.append(feature_space_html)

    if SECTION_ALIGNMENT in sections:
        parts += alignment_tables(result, palette)

    # Its own section rather than riding along with Evaluation: this table is
    # one row per target sample, so on a large run it dominates the PDF and
    # is the section a reader is most likely to want left out.
    if SECTION_PREDICTIONS in sections and result.target_predictions is not None:
        parts.append("<h2>Target Predictions</h2><table width='100%'>")
        parts.append(
            "<tr><th>Sample</th><th>Actual Label</th><th>Predicted Label</th><th>Result</th></tr>"
        )
        for row in range(len(result.target_predictions)):
            is_correct = result.prediction_correctness[row] == "Right"
            colour = palette["success"] if is_correct else palette["danger"]
            verdict = "Correct" if is_correct else "Incorrect"
            parts.append(
                f"<tr><td>Sample {row + 1}</td>"
                f"<td>{escape(str(result.target_actuals[row]))}</td>"
                f"<td>{escape(str(result.target_predictions[row]))}</td>"
                f"<td style='color:{colour}; font-weight:bold;'>{verdict}</td></tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    return "".join(parts)


__all__ = ["build_report_html"]
