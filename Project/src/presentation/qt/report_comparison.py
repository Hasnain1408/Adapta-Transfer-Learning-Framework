"""Report for the stored-results comparison view.

Owns the wide-table chunking, which only this report needs: the Comparison
tab's "Exact Values" table can carry far more columns than a printed page
holds.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from presentation.qt.report_common import page_header

#: How many data columns (beyond the anchor pair below) a printed table
#: holds before headers like "Baseline Acc" start wrapping mid-word
#: ("Baselin/e Acc") or losing their one-line layout. Tuned empirically
#: against the actual print geometry (A4, 15mm margins, 9pt table font,
#: see REPORT_IMAGE_WIDTH): 7 data columns still wrapped headers like
#: "Precision" mid-word, 5 does not. The Comparison report's "Exact Values"
#: table can carry up to 16 columns (see comparison_tab.TABLE_COLUMNS)
#: — CSV export has no width limit and uses the same rows unsplit, so this
#: only reshapes the PDF's copy.
_MAX_TABLE_DATA_COLUMNS = 5
#: Leading columns repeated in every chunk of a split table (here: Rank,
#: Method) so a reader can still tell which row is which after the split.
_TABLE_ANCHOR_COLUMNS = 2


def _chunk_wide_table(
    header: list[str], body: list[list[str]]
) -> list[tuple[list[str], list[list[str]]]]:
    """Split a many-column table into page-width-friendly chunks.

    A table wider than the printable page doesn't reflow in Qt's rich-text
    renderer the way a browser would — the columns just get squeezed until
    header text wraps letter by letter. Splitting into several narrower
    tables, each repeating the first _TABLE_ANCHOR_COLUMNS columns so a
    reader can still tell which row is which, keeps every header on one
    line. A table that already fits is returned unchanged as a single chunk.
    """
    anchors = header[:_TABLE_ANCHOR_COLUMNS]
    rest_count = len(header) - len(anchors)
    if rest_count <= _MAX_TABLE_DATA_COLUMNS:
        return [(header, body)]

    chunks: list[tuple[list[str], list[list[str]]]] = []
    for start in range(0, rest_count, _MAX_TABLE_DATA_COLUMNS):
        width = min(_MAX_TABLE_DATA_COLUMNS, rest_count - start)
        indices = list(range(_TABLE_ANCHOR_COLUMNS)) + [
            _TABLE_ANCHOR_COLUMNS + start + offset for offset in range(width)
        ]
        chunks.append(
            (
                [header[i] for i in indices],
                [[row[i] for i in indices] for row in body],
            )
        )
    return chunks


def build_comparison_report_html(
    context: dict[str, str],
    chart_sections: list[tuple[str, str]],
    table_rows: list[list[str]],
    performed_by: str,
    include_table: bool = True,
) -> str:
    """Render the stored-results comparison as a standalone report.

    *chart_sections* is [(view name, <img> markup), ...] — one entry per
    comparison view the user ticked, already rendered by the tab that owns
    the figure. *table_rows* is the exact-values table with its header row
    first. Nothing is computed here; this only lays out what it is given.
    """
    parts = page_header("Experimental Results Comparison", performed_by, datetime.now())

    parts.append("<h2>Comparison Scope</h2><table width='100%'>")
    for label, value in context.items():
        parts.append(
            f"<tr><th style='width:35%'>{escape(label)}</th><td>{escape(str(value))}</td></tr>"
        )
    parts.append("</table>")

    for title, chart_html in chart_sections:
        if not chart_html:
            continue
        parts.append(f"<h2>{escape(title)}</h2>")
        parts.append(chart_html)

    if include_table and len(table_rows) > 1:
        parts.append("<h2>Exact Values</h2>")
        header, *body = table_rows
        for chunk_header, chunk_body in _chunk_wide_table(header, body):
            parts.append("<table width='100%' style='margin-top:8px'>")
            parts.append(
                "<tr>" + "".join(f"<th>{escape(str(cell))}</th>" for cell in chunk_header) + "</tr>"
            )
            for row in chunk_body:
                cells = "".join(
                    f"<td class='num'>{escape(str(cell))}</td>" if index else
                    f"<td>{escape(str(cell))}</td>"
                    for index, cell in enumerate(row)
                )
                parts.append(f"<tr>{cells}</tr>")
            parts.append("</table>")
        parts.append(
            "<p class='meta'>An em-dash means the value was never recorded for "
            "that run, not that it was zero.</p>"
        )

    parts.append("</body></html>")
    return "".join(parts)


__all__ = ["build_comparison_report_html"]
