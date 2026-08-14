"""Pure HTML rendering for exported PDF reports.

No Qt/``self`` dependency, mirroring ``result_tables.py`` — the caller
(``MainWindowReportMixin``) gathers the run's settings, user, and chart
markup and hands them in, so these are just string builders that can be
read (and tested) on their own.

Three entry points share one visual language (``_HEAD``/``_page_header``):

- :func:`build_report_html` — a single source→target run, live or replayed
  from a batch pair (both are a ``TransferLearningResult``).
- :func:`build_batch_report_html` — every pair of a batch run at once.
- :func:`build_history_report_html` — a run replayed from the persisted
  ``experiments``/``evaluations`` tables, which has no per-sample data (no
  predictions table, no charts) since none of that is stored.

``sections`` (a set of the ``export_options_dialog.SECTION_*`` keys) gates
the Configuration/Dataset/Evaluation/Visualizations blocks in all three;
"Performed by" and "Generated" are not gateable — a report always says who
ran it and when it was produced.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from domain.entities import DatasetRecord, EvaluationRecord, ExperimentHistoryEntry
from domain.models import BatchTransferLearningResult, TransferLearningResult
from presentation.qt.export_options_dialog import (
    SECTION_ALIGNMENT,
    SECTION_CONFIGURATION,
    SECTION_DATASET,
    SECTION_EVALUATION,
    SECTION_VISUALIZATIONS,
)

_HEAD = (
    "<html><head><meta charset='utf-8'><style>"
    "body { font-family: 'Helvetica', Arial, sans-serif; color: #1a2229; font-size: 10pt; }"
    "h1 { font-size: 16pt; margin: 0 0 2px 0; }"
    "h2 { font-size: 12pt; margin: 22px 0 6px 0; border-bottom: 1px solid #c3ccd6;"
    " padding-bottom: 3px; }"
    "p.meta { color: #5b6b7c; font-size: 9pt; margin: 0 0 4px 0; }"
    "table { border-collapse: collapse; width: 100%; }"
    "th, td { border: 1px solid #dde2e8; padding: 5px 7px; text-align: left; }"
    "th { background-color: #eef1f4; font-size: 9pt; }"
    "td.num { text-align: right; }"
    "</style></head><body>"
)


def _page_header(title: str, performed_by: str) -> list[str]:
    generated = datetime.now().strftime("%d %B %Y, %H:%M")
    return [
        _HEAD,
        f"<h1>{escape(title)}</h1>",
        f"<p class='meta'>Generated {escape(generated)}  ·  Performed by {escape(performed_by)}</p>",
    ]


def _config_table(config_rows: list[tuple[str, str]]) -> list[str]:
    parts = ["<h2>Configuration</h2><table width='100%'>"]
    for label, value in config_rows:
        parts.append(
            f"<tr><th style='width:35%'>{escape(label)}</th><td>{escape(str(value))}</td></tr>"
        )
    parts.append("</table>")
    return parts


def _metrics_tables(metrics) -> list[str]:
    parts = ["<h2>Evaluation Metrics</h2><table width='100%'>"]
    parts.append(
        "<tr><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>"
    )
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
    return parts


def _alignment_tables(result: TransferLearningResult, palette: dict[str, str]) -> list[str]:
    """Domain alignment (MMD) and the negative-transfer verdict.

    Both come off the result as recorded by the use case, so the PDF states
    the same numbers the Feature Space tab showed — nothing is recomputed
    here from the plot.
    """
    alignment = result.alignment
    analysis = result.negative_transfer
    if alignment is None and analysis is None:
        return []

    parts: list[str] = ["<h2>Domain Alignment &amp; Transfer Analysis</h2>"]

    if alignment is not None:
        parts.append("<table width='100%'>")
        parts.append(
            "<tr><th>MMD Before</th><th>MMD After</th><th>Change</th>"
            "<th>Alignment Status</th></tr>"
        )
        parts.append(
            f"<tr><td class='num'>{alignment.mmd_before:.6f}</td>"
            f"<td class='num'>{alignment.mmd_after:.6f}</td>"
            f"<td class='num'>{alignment.change_percent:+.2f}%</td>"
            f"<td>{escape(alignment.status)}</td></tr></table>"
        )
        parts.append(
            "<p class='meta'>MMD is measured across the domain gap in the "
            "preprocessed input space (before) and the method's output space "
            "(after); a positive change means the domains moved closer.</p>"
        )

    if analysis is not None:
        parts.append("<table width='100%' style='margin-top:8px'>")
        parts.append(
            "<tr><th>Target Metric</th><th>Without Adaptation</th>"
            "<th>After Adaptation</th><th>Change</th></tr>"
        )
        for label, baseline, adapted, delta in (
            ("Accuracy", analysis.baseline.accuracy, analysis.adapted.accuracy, analysis.accuracy_delta),
            ("F1 Score", analysis.baseline.f1_score, analysis.adapted.f1_score, analysis.f1_delta),
        ):
            parts.append(
                f"<tr><td>{label}</td><td class='num'>{baseline:.4f}</td>"
                f"<td class='num'>{adapted:.4f}</td>"
                f"<td class='num'>{delta:+.4f}</td></tr>"
            )
        parts.append("</table>")

        if analysis.is_negative_transfer:
            parts.append(
                f"<p style='color:{palette['danger']}; font-weight:bold; margin-top:8px;'>"
                "Potential Negative Transfer — target accuracy fell against the same "
                "classifier trained without adaptation.</p>"
            )
    return parts


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
    from ``AnalyticsChartsWidget.render_to_base64_html()``); pass "" to omit
    the Visual Analytics section regardless of *sections*.

    *feature_space_html* / *feature_space_rows* carry the Feature Space tab's
    scatter plot and the projection/stage it was drawn with, so the reader
    knows which view of the data they are looking at.
    """
    metrics = result.metrics
    parts = _page_header("Transfer Learning Evaluation Report", performed_by)

    if SECTION_CONFIGURATION in sections:
        parts += _config_table(config_rows)

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
        parts += _metrics_tables(metrics)

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
        parts += _alignment_tables(result, palette)

    if SECTION_EVALUATION in sections and result.target_predictions is not None:
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
    parts = _page_header("Transfer Learning Batch Report", performed_by)

    if SECTION_CONFIGURATION in sections:
        parts += _config_table(config_rows)

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
            parts += _metrics_tables(avg)

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


def build_history_report_html(
    entry: ExperimentHistoryEntry,
    source_dataset: DatasetRecord | None,
    target_dataset: DatasetRecord | None,
    sections: set[str],
) -> str:
    """Render a report from a persisted experiment row.

    Only what the ``experiments``/``evaluations``/``datasets`` tables
    actually store: no per-sample predictions and no charts, since neither
    is kept once the run that produced them ends. ``performed_by`` comes
    from the row's owner rather than a live signed-in user, since the two
    can differ (anyone can browse the shared history, and an experiment's
    owner may since have signed out or be a different account entirely).
    """
    experiment = entry.experiment
    performed_by = entry.user.name if entry.user else "Unknown user (account deleted)"
    parts = _page_header(f"Experiment #{experiment.id} Report", performed_by)

    if SECTION_CONFIGURATION in sections:
        config_rows = [
            ("Method", experiment.tl_algorithm),
            ("Classifier", experiment.classifier or "—"),
            ("Source dataset", experiment.source_domain),
            ("Target dataset", experiment.target_domain),
            ("Normalization", experiment.normalization_type),
            ("Standardization", experiment.standardization_type),
        ]
        if experiment.kernel_type is not None:
            config_rows.append(("Kernel", experiment.kernel_type))
        if experiment.regularization_value is not None:
            config_rows.append(("Regularization (lambda)", f"{experiment.regularization_value:g}"))
        if experiment.latent_feature is not None:
            config_rows.append(("Latent dimensions", str(experiment.latent_feature)))
        parts += _config_table(config_rows)

    if SECTION_DATASET in sections:
        parts.append("<h2>Dataset Summary</h2><table width='100%'>")
        parts.append(
            "<tr><th>Dataset</th><th>File</th><th>Samples</th><th>Features</th>"
            "<th>Label Column</th><th>Classes</th></tr>"
        )
        for role, domain_name, dataset in (
            ("Source", experiment.source_domain, source_dataset),
            ("Target", experiment.target_domain, target_dataset),
        ):
            if dataset is None:
                parts.append(
                    f"<tr><td>{role}</td><td>{escape(domain_name)}</td>"
                    "<td colspan='4'>No longer in the dataset registry.</td></tr>"
                )
                continue
            parts.append(
                f"<tr><td>{role}</td><td>{escape(dataset.name)}</td>"
                f"<td class='num'>{dataset.num_samples}</td>"
                f"<td class='num'>{dataset.num_features}</td>"
                f"<td>{escape(dataset.label_name or '—')}</td>"
                f"<td class='num'>{dataset.num_classes}</td></tr>"
            )
        parts.append("</table>")

    if SECTION_EVALUATION in sections:
        evaluation = entry.evaluation
        if evaluation is not None:
            parts += _metrics_tables(evaluation)
        else:
            parts.append("<h2>Evaluation Metrics</h2><p>No metrics were recorded for this run.</p>")
        if experiment.error_message:
            parts.append(
                f"<p style='color:#b3261e;'><b>Error:</b> {escape(experiment.error_message)}</p>"
            )

    parts.append("</body></html>")
    return "".join(parts)


__all__ = ["build_report_html", "build_batch_report_html", "build_history_report_html"]
