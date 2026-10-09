"""Report replayed from a persisted experiment row.

Covers only what the ``experiments``/``evaluations``/``datasets`` tables
actually store — no per-sample predictions and no charts, since neither
survives the run that produced them.
"""

from __future__ import annotations

from html import escape

from domain.entities import DatasetRecord, ExperimentHistoryEntry
from presentation.qt.export_options_dialog import (
    SECTION_CONFIGURATION,
    SECTION_DATASET,
    SECTION_EVALUATION,
)
from presentation.qt.report_common import config_table, metrics_tables, page_header


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
    parts = page_header(
        f"Experiment #{experiment.id} Report", performed_by, experiment.date
    )

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
        parts += config_table(config_rows)

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
            parts += metrics_tables(evaluation)
        else:
            parts.append("<h2>Evaluation Metrics</h2><p>No metrics were recorded for this run.</p>")
        if experiment.error_message:
            parts.append(
                f"<p style='color:#b3261e;'><b>Error:</b> {escape(experiment.error_message)}</p>"
            )

    parts.append("</body></html>")
    return "".join(parts)


__all__ = ["build_history_report_html"]
