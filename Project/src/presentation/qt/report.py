"""Pure HTML rendering for exported PDF reports.

No Qt/self dependency, mirroring result_tables.py — the caller
(MainWindowReportMixin) gathers the run's settings, user, and chart
markup and hands them in, so these are just string builders that can be
read (and tested) on their own.

This module is the facade: one import site for every report kind, with the
actual builders split one per module. Four entry points share the visual
language in report_common.py (PAGE_HEAD/page_header):

- build_report_html (report_single) — a single source→target
  run, live or replayed from a batch pair (both are a
  TransferLearningResult).
- build_batch_report_html (report_batch) — every pair of a
  batch run at once.
- build_multi_method_report_html (report_multi_method) — every
  method of one comparison run, ranked, against one dataset pair.
- build_comparison_report_html (report_comparison) — the
  stored-results comparison view.
- build_history_report_html (report_history) — a run replayed
  from the persisted experiments/evaluations tables, which has no
  per-sample data (no predictions table, no charts) since none of that is
  stored.

sections (a set of the export_options_dialog.SECTION_* keys) gates
the Configuration/Dataset/Evaluation/Visualizations/Alignment/Predictions
blocks. The header is not gateable — a report always says who ran it, when
the experiment itself ran, and when the PDF was produced.
"""

from __future__ import annotations

from presentation.qt.report_batch import build_batch_report_html
from presentation.qt.report_common import REPORT_IMAGE_WIDTH
from presentation.qt.report_comparison import build_comparison_report_html
from presentation.qt.report_history import build_history_report_html
from presentation.qt.report_multi_method import build_multi_method_report_html
from presentation.qt.report_single import build_report_html

__all__ = [
    "REPORT_IMAGE_WIDTH",
    "build_batch_report_html",
    "build_comparison_report_html",
    "build_history_report_html",
    "build_multi_method_report_html",
    "build_report_html",
]
