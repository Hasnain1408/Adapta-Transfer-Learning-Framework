"""Pure HTML rendering for the PDF evaluation report.

No Qt/``self`` dependency, mirroring ``result_tables.py`` — the caller
(``MainWindowReportMixin._export_pdf_report``) gathers the current run's
settings and chart markup and hands them in, so this function is just a
string builder that can be read (and tested) on its own.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from domain.models import TransferLearningResult


def build_report_html(
    result: TransferLearningResult,
    config_rows: list[tuple[str, str]],
    palette: dict[str, str],
    chart_html: str,
) -> str:
    """Render one transfer-learning run as a standalone HTML report.

    *config_rows* are the method/classifier/dataset/preprocessing settings
    shown under "Configuration". *chart_html* is pre-rendered markup (e.g.
    from ``AnalyticsChartsWidget.render_to_base64_html()``); pass "" to omit
    the Visual Analytics section.
    """
    metrics = result.metrics
    generated = datetime.now().strftime("%d %B %Y, %H:%M")

    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body { font-family: 'Helvetica', Arial, sans-serif; color: #1a2229; font-size: 10pt; }",
        "h1 { font-size: 16pt; margin: 0 0 2px 0; }",
        "h2 { font-size: 12pt; margin: 22px 0 6px 0; border-bottom: 1px solid #c3ccd6;"
        " padding-bottom: 3px; }",
        "p.meta { color: #5b6b7c; font-size: 9pt; margin: 0 0 4px 0; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #dde2e8; padding: 5px 7px; text-align: left; }",
        "th { background-color: #eef1f4; font-size: 9pt; }",
        "td.num { text-align: right; }",
        "</style></head><body>",
        "<h1>Transfer Learning Evaluation Report</h1>",
        f"<p class='meta'>Generated {escape(generated)}</p>",
        "<h2>Configuration</h2><table width='100%'>",
    ]
    for label, value in config_rows:
        parts.append(f"<tr><th style='width:35%'>{escape(label)}</th>"
                     f"<td>{escape(str(value))}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>Dataset Summary</h2><table width='100%'>")
    parts.append("<tr><th>Dataset</th><th>Samples</th><th>Features</th>"
                 "<th>Label Column</th></tr>")
    for role, dataset in (("Source", result.source_dataset), ("Target", result.target_dataset)):
        parts.append(
            f"<tr><td>{role}</td><td class='num'>{dataset.features.shape[0]}</td>"
            f"<td class='num'>{dataset.features.shape[1]}</td>"
            f"<td>{escape(str(dataset.label_name))}</td></tr>"
        )
    parts.append("</table>")

    if metrics is not None:
        parts.append("<h2>Evaluation Metrics</h2><table width='100%'>")
        parts.append("<tr><th>Accuracy</th><th>Precision</th><th>Recall</th>"
                     "<th>F1 Score</th></tr>")
        parts.append(
            f"<tr><td class='num'>{metrics.accuracy:.4f}</td>"
            f"<td class='num'>{metrics.precision:.4f}</td>"
            f"<td class='num'>{metrics.recall:.4f}</td>"
            f"<td class='num'>{metrics.f1_score:.4f}</td></tr></table>"
        )
        parts.append("<table width='100%' style='margin-top:8px'>")
        parts.append("<tr><th>Total Samples</th><th>Correct</th><th>Incorrect</th></tr>")
        parts.append(
            f"<tr><td class='num'>{metrics.total_samples}</td>"
            f"<td class='num'>{metrics.correct_predictions}</td>"
            f"<td class='num'>{metrics.wrong_predictions}</td></tr></table>"
        )

    if chart_html:
        parts.append("<h2>Visual Analytics</h2>")
        parts.append(chart_html)

    if result.target_predictions is not None:
        parts.append("<h2>Target Predictions</h2><table width='100%'>")
        parts.append("<tr><th>Sample</th><th>Actual Label</th><th>Predicted Label</th>"
                     "<th>Result</th></tr>")
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
