"""Shared building blocks for every exported PDF report.

The four ``build_*_report_html`` entry points (single run, batch, comparison,
history) each live in their own module and all draw their visual language
from here: one ``<head>``, one title block, and the table fragments that more
than one of them needs. Nothing in this module knows which report it is
rendering — callers pass in what they have.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from domain.models import TransferLearningResult
from domain.overlap import OVERLAP_DISCLAIMER
from presentation.qt.duration import format_duration

#: Safe pixel width for a chart ``<img>`` embedded in the printed report.
#: Measured against the actual print geometry ``_write_pdf_report`` uses in
#: ``main_window_report.py`` (A4, 15mm margins on every side): that leaves a
#: 509pt-wide printable column, verified with
#: ``QPrinter().pageRect(QPrinter.Point)``. 480 leaves a small buffer below
#: that so a Letter-sized default (slightly wider) is covered too.
#:
#: This has to be a plain HTML ``width`` attribute, not a CSS percentage —
#: Qt's ``QTextDocument`` (what actually prints the report) does not scale
#: ``<img>`` to a percentage or ``max-width`` style; it renders at native
#: pixel size, which is far wider than a printed page and gets clipped at
#: the right margin. Height is left for the caller to omit so Qt scales it
#: proportionally instead of guessing an aspect ratio.
REPORT_IMAGE_WIDTH = 480

PAGE_HEAD = (
    "<html><head><meta charset='utf-8'><style>"
    "body { font-family: 'Helvetica', Arial, sans-serif; color: #1a2229; font-size: 10pt; }"
    "h1 { font-size: 16pt; margin: 0 0 2px 0; }"
    "h2 { font-size: 12pt; margin: 22px 0 6px 0; border-bottom: 1px solid #c3ccd6;"
    " padding-bottom: 3px; }"
    "h3 { font-size: 10.5pt; margin: 14px 0 4px 0; color: #33414f; }"
    "p.meta { color: #5b6b7c; font-size: 9pt; margin: 0 0 4px 0; }"
    "table { border-collapse: collapse; width: 100%; }"
    "th, td { border: 1px solid #dde2e8; padding: 5px 7px; text-align: left; }"
    "th { background-color: #eef1f4; font-size: 9pt; }"
    "td.num { text-align: right; }"
    # For tables with more columns than the default padding leaves room for.
    # At 5px/7px padding a nine-column table on A4 has ~48pt of content
    # width per column, which is under what "Precision" needs as a header
    # and what "0.2410" needs as a value -- both then wrap mid-word. Tighter
    # padding and a smaller face buy back roughly 25pt per column, which is
    # the difference between a readable row and a broken one.
    "table.compact th, table.compact td { padding: 3px 4px; font-size: 8.5pt; }"
    "</style></head><body>"
)


def format_timestamp(value: datetime | None) -> str:
    """Render a run timestamp in local time, tolerating a tz-aware value.

    Persisted rows come back from a TIMESTAMPTZ column as tz-aware, while a
    live run records a naive local ``datetime.now()``; ``astimezone()`` on a
    naive value assumes local time, so both land in the reader's zone.
    """
    if value is None:
        return ""
    try:
        return value.astimezone().strftime("%d %B %Y, %H:%M:%S")
    except (ValueError, OSError):  # out-of-range or unconvertible timestamp
        return str(value)


def page_header(
    title: str,
    performed_by: str,
    experiment_time: datetime | None = None,
    duration_seconds: float | None = None,
) -> list[str]:
    """Title block: who ran it, when it ran, and when the PDF was produced.

    The experiment time and the generation time are deliberately separate
    lines — a report can be exported days after the run (or re-exported from
    history repeatedly), so collapsing the two would misdate the experiment.
    """
    generated = datetime.now().strftime("%d %B %Y, %H:%M")
    parts = [PAGE_HEAD, f"<h1>{escape(title)}</h1>"]
    if experiment_time is not None:
        line = f"Experiment run {format_timestamp(experiment_time)}"
        duration = format_duration(duration_seconds)
        if duration:
            line += f"  ·  took {duration}"
        parts.append(f"<p class='meta'>{escape(line)}</p>")
    parts.append(
        f"<p class='meta'>Generated {escape(generated)}  ·  Performed by {escape(performed_by)}</p>"
    )
    return parts


def config_table(config_rows: list[tuple[str, str]]) -> list[str]:
    parts = ["<h2>Configuration</h2><table width='100%'>"]
    for label, value in config_rows:
        parts.append(
            f"<tr><th style='width:35%'>{escape(label)}</th><td>{escape(str(value))}</td></tr>"
        )
    parts.append("</table>")
    return parts


def metrics_tables(metrics) -> list[str]:
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


def alignment_tables(result: TransferLearningResult, palette: dict[str, str]) -> list[str]:
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

        if alignment.overlap_before is not None and alignment.overlap_after is not None:
            change = alignment.overlap_change_points or 0.0
            parts.append("<h3>Domain Overlap</h3>")
            parts.append("<table width='100%'>")
            parts.append(
                "<tr><th>Overlap Before</th><th>Overlap After</th><th>Change</th></tr>"
            )
            parts.append(
                f"<tr><td class='num'>{alignment.overlap_before:.1f}%</td>"
                f"<td class='num'>{alignment.overlap_after:.1f}%</td>"
                f"<td class='num'>{change:+.1f} points</td></tr></table>"
            )
            # The same caveat the Feature Space panel carries. A report is
            # read away from the app, often by someone who never saw the
            # scatter plot, so the qualifier travels with the number.
            parts.append(f"<p class='meta'>{escape(OVERLAP_DISCLAIMER)}</p>")

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


def timing_table(result: TransferLearningResult) -> list[str]:
    """Execution cost: whole run vs. the method's own adaptation step."""
    total = format_duration(result.duration_seconds)
    if not total:
        return []
    adaptation = format_duration(result.adaptation_seconds)
    parts = ["<h2>Execution Time</h2><table width='100%'>"]
    parts.append("<tr><th>Total Run</th><th>Adaptation (method only)</th></tr>")
    parts.append(
        f"<tr><td class='num'>{escape(total)}</td>"
        f"<td class='num'>{escape(adaptation or '—')}</td></tr></table>"
    )
    parts.append(
        "<p class='meta'>Total covers loading, preprocessing, adaptation, and "
        "evaluation; the adaptation figure isolates the method itself and is "
        "the one to compare across methods. Time spent paused is excluded.</p>"
    )
    return parts


__all__ = [
    "PAGE_HEAD",
    "REPORT_IMAGE_WIDTH",
    "alignment_tables",
    "config_table",
    "format_timestamp",
    "metrics_tables",
    "page_header",
    "timing_table",
]
