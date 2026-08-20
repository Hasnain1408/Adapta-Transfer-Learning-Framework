"""Report comparing several methods run against one dataset pair.

Distinct from ``report_comparison``, which ranks *stored* experiments
across a history: this covers one run, where the datasets, preprocessing,
classifier, and baseline were identical by construction rather than by
coincidence. That guarantee is worth stating in the report itself, because
it is the reason these numbers can be set against each other at all — so
the header block says so, and the baseline is reported once rather than
repeated per method.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from domain.models import MultiMethodTransferLearningResult
from domain.overlap import OVERLAP_DISCLAIMER
from presentation.qt.duration import format_duration
from presentation.qt.export_options_dialog import (
    SECTION_ALIGNMENT,
    SECTION_CONFIGURATION,
    SECTION_DATASET,
    SECTION_EVALUATION,
)
from presentation.qt.report_common import config_table, metrics_tables, page_header


def _dataset_section(multi_result: MultiMethodTransferLearningResult) -> list[str]:
    best = multi_result.best
    reference = best.result if best is not None else next(
        (entry.result for entry in multi_result.methods if entry.result is not None), None
    )
    parts = ["<h2>Dataset Pair</h2><table width='100%'>"]
    parts.append(
        f"<tr><th style='width:35%'>Source</th>"
        f"<td>{escape(Path(multi_result.source_path).name)}</td></tr>"
    )
    parts.append(
        f"<tr><th>Target</th><td>{escape(Path(multi_result.target_path).name)}</td></tr>"
    )
    if reference is not None:
        source_shape = reference.source_dataset.features.shape
        target_shape = reference.target_dataset.features.shape
        parts.append(
            f"<tr><th>Source shape</th><td>{source_shape[0]} rows x {source_shape[1]} features</td></tr>"
        )
        parts.append(
            f"<tr><th>Target shape</th><td>{target_shape[0]} rows x {target_shape[1]} features</td></tr>"
        )
    parts.append("</table>")
    return parts


def _ranking_table(multi_result: MultiMethodTransferLearningResult) -> list[str]:
    baseline = multi_result.baseline
    ranked = multi_result.ranked()
    ranked_names = {entry.method_name for entry in ranked}
    unranked = [e for e in multi_result.methods if e.method_name not in ranked_names]

    average_ranks = multi_result.method_ranks()

    # class='compact' because this is the widest table in the app's reports:
    # nine columns on A4. At the default cell padding both the headers and
    # the values wrapped mid-word ("Accurac / y", "0. / 2410").
    parts = ["<h2>Method Ranking</h2><table width='100%' class='compact'>"]
    parts.append(
        "<tr><th>#</th><th>Method</th><th>Avg Rank</th><th>Accuracy</th>"
        "<th>Precision</th><th>Recall</th><th>F1</th><th>vs Base</th>"
        "<th>Time</th></tr>"
    )
    for position, entry in enumerate(ranked, start=1):
        metrics = entry.metrics
        delta = (
            f"{metrics.accuracy - baseline.accuracy:+.4f}" if baseline is not None else "—"
        )
        average = average_ranks.get(entry.method_name)
        average_cell = f"{average:.2f}" if average is not None else "—"
        name = escape(entry.method_name)
        if position == 1:
            name = f"<b>{name}</b>"
        parts.append(
            f"<tr><td class='num'>{position}</td><td>{name}</td>"
            f"<td class='num'>{average_cell}</td>"
            f"<td class='num'>{metrics.accuracy:.4f}</td>"
            f"<td class='num'>{metrics.precision:.4f}</td>"
            f"<td class='num'>{metrics.recall:.4f}</td>"
            f"<td class='num'>{metrics.f1_score:.4f}</td>"
            f"<td class='num'>{delta}</td>"
            f"<td class='num'>{escape(format_duration(entry.result.adaptation_seconds))}</td></tr>"
        )
    for entry in unranked:
        reason = "No labels to evaluate against" if entry.result is not None else (entry.error or "Failed")
        parts.append(
            f"<tr><td class='num'>—</td><td>{escape(entry.method_name)}</td>"
            f"<td colspan='7'>{escape(reason)}</td></tr>"
        )
    parts.append("</table>")
    parts.append(
        "<p class='meta'>Ranked by mean rank across accuracy, precision, recall, F1, "
        "and AUC — not by accuracy alone, which on an imbalanced target would favour "
        "a classifier that simply predicts the majority class. Accuracy breaks a tie, "
        "then adaptation time. Adaptation time is the method's own cost, excluding "
        "the dataset loading and classifier fitting shared by every row.</p>"
    )
    return parts


def _alignment_table(multi_result: MultiMethodTransferLearningResult) -> list[str]:
    """How far each method moved the domains together.

    Two narrow tables rather than one eight-column one: on A4 the combined
    version wrapped its own cells ("No Significa / nt Change", "0. / 579133"),
    which is the same trap the single-run report avoids by keeping MMD and
    Overlap apart.
    """
    entries = [e for e in multi_result.methods if e.result is not None and e.result.alignment]
    if not entries:
        return []

    parts = ["<h2>Domain Alignment by Method</h2>", "<table width='100%'>"]
    parts.append(
        "<tr><th width='24%'>Method</th><th width='20%'>MMD Before</th>"
        "<th width='20%'>MMD After</th><th width='14%'>Change</th>"
        "<th width='22%'>Status</th></tr>"
    )
    for entry in entries:
        alignment = entry.result.alignment
        parts.append(
            f"<tr><td>{escape(entry.method_name)}</td>"
            f"<td class='num'>{alignment.mmd_before:.4f}</td>"
            f"<td class='num'>{alignment.mmd_after:.4f}</td>"
            f"<td class='num'>{alignment.change_percent:+.1f}%</td>"
            f"<td>{escape(alignment.status)}</td></tr>"
        )
    parts.append("</table>")
    parts.append(
        "<p class='meta'>MMD Before is identical across methods by construction — "
        "every method received the same preprocessed features. Only MMD After and "
        "its change distinguish them; a positive change means the domains moved "
        "closer together.</p>"
    )

    overlapping = [
        e for e in entries if e.result.alignment.overlap_change_points is not None
    ]
    if overlapping:
        parts.append("<h3>Domain Overlap by Method</h3><table width='100%'>")
        parts.append(
            "<tr><th width='28%'>Method</th><th width='24%'>Overlap Before</th>"
            "<th width='24%'>Overlap After</th><th width='24%'>Change</th></tr>"
        )
        for entry in overlapping:
            alignment = entry.result.alignment
            parts.append(
                f"<tr><td>{escape(entry.method_name)}</td>"
                f"<td class='num'>{alignment.overlap_before:.1f}%</td>"
                f"<td class='num'>{alignment.overlap_after:.1f}%</td>"
                f"<td class='num'>{alignment.overlap_change_points:+.1f} pts</td></tr>"
            )
        parts.append("</table>")
        parts.append(f"<p class='meta'>{escape(OVERLAP_DISCLAIMER)}</p>")

    return parts


def build_multi_method_report_html(
    multi_result: MultiMethodTransferLearningResult,
    config_rows: list[tuple[str, str]],
    performed_by: str,
    sections: set[str],
) -> str:
    """Render a whole multi-method comparison as one standalone HTML report.

    No Visual Analytics section regardless of *sections*: a comparison has
    several results and no single one to chart. Open an individual method
    from the Method Results table and export its own report for that.
    """
    parts = page_header(
        "Transfer Learning Method Comparison",
        performed_by,
        multi_result.started_at,
        multi_result.duration_seconds,
    )

    best = multi_result.best
    if best is not None:
        summary = (
            f"{len(multi_result.evaluated)} of {len(multi_result.methods)} methods "
            f"comparable  ·  best: {best.method_name} at {best.metrics.accuracy:.4f} accuracy"
        )
    else:
        summary = f"{len(multi_result.methods)} methods run; none produced evaluable metrics"
    if multi_result.cancelled:
        summary = "Cancelled before the remaining methods ran  ·  " + summary
    parts.append(f"<p class='meta'>{escape(summary)}</p>")
    parts.append(
        "<p class='meta'>Every method below ran against the same source and target "
        "files, with the same normalization, standardization, and classifier, from a "
        "single shared preparation — so any difference between them is the method.</p>"
    )

    if SECTION_CONFIGURATION in sections:
        parts += config_table(config_rows)

    if SECTION_DATASET in sections:
        parts += _dataset_section(multi_result)

    if SECTION_EVALUATION in sections:
        parts += _ranking_table(multi_result)
        if multi_result.baseline is not None:
            parts.append("<h2>Shared No-Adaptation Baseline</h2>")
            parts.append(
                "<p class='meta'>The same classifier fitted on the preprocessed source "
                "and evaluated on the preprocessed target, with no adaptation at all. "
                "Measured once and used for every method's “vs Baseline” figure above.</p>"
            )
            parts += metrics_tables(multi_result.baseline)

    if SECTION_ALIGNMENT in sections:
        parts += _alignment_table(multi_result)

    parts.append("</body></html>")
    return "".join(parts)


__all__ = ["build_multi_method_report_html"]
