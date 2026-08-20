"""The data model behind the comparison view: one row per method, ranked.

Pure data and arithmetic over ``ExperimentHistoryEntry`` — no Qt — so the
ranking rules can be read (and checked) apart from the widget that plots
them. Ranking is *competition* ranking: ties share a place and the next
place is skipped, which is what a reader expects from a leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from domain.entities import ExperimentHistoryEntry
from domain.ranking import average_metric_ranks, competition_ranks

#: ``(label, EvaluationRecord attribute)`` for every rankable score.
METRICS: list[tuple[str, str]] = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1_score"),
    ("AUC", "auc"),
]

VIEW_PERFORMANCE = "Performance Comparison"
VIEW_RANKING = "Method Ranking"
VIEW_ALIGNMENT = "Domain Alignment (MMD)"
VIEW_EXECUTION = "Execution Time"
VIEW_NEGATIVE = "Baseline vs. Transfer"
VIEW_PAIRS = "Source–Target Pairs"

VIEWS = [
    VIEW_PERFORMANCE,
    VIEW_RANKING,
    VIEW_ALIGNMENT,
    VIEW_EXECUTION,
    VIEW_NEGATIVE,
    VIEW_PAIRS,
]

SORT_METRIC_DESC = "Metric (high → low)"
SORT_METRIC_ASC = "Metric (low → high)"
SORT_NAME = "Method name"
SORT_TIME = "Execution time (fast → slow)"

SORT_OPTIONS = [SORT_METRIC_DESC, SORT_METRIC_ASC, SORT_NAME, SORT_TIME]

#: Sentinel metric key: rank by mean rank across all recorded metrics
#: instead of by any single one. Lower is better, unlike every real metric.
METRIC_OVERALL = "__overall__"
METRIC_OVERALL_LABEL = "Overall (avg. rank)"

#: Label used for the no-adaptation reference in charts and exports.
BASELINE_LABEL = "Baseline"

TABLE_COLUMNS = [
    "Rank",
    "Method",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "AUC",
    "MMD Before",
    "MMD After",
    "MMD Change",
    "Alignment",
    "Overlap Before",
    "Overlap After",
    "Overlap Δ (pts)",
    "Exec Time",
    "Baseline Acc",
    "Δ Accuracy",
    "Negative?",
    "Avg. Rank",
]


@dataclass(frozen=True)
class MethodRow:
    """One method's stored numbers for the selected pair.

    Every score is optional: this is a view onto whatever the database
    actually holds, and callers must render ``None`` as a gap.
    """

    method: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc: Optional[float]
    mmd_before: Optional[float]
    mmd_after: Optional[float]
    mmd_change: Optional[float]
    overlap_before: Optional[float]
    overlap_after: Optional[float]
    overlap_change: Optional[float]
    alignment_status: Optional[str]
    execution_seconds: Optional[float]
    baseline_accuracy: Optional[float]
    baseline_f1: Optional[float]
    accuracy_delta: Optional[float]
    experiment_id: Optional[int]

    @property
    def is_negative_transfer(self) -> bool:
        return self.accuracy_delta is not None and self.accuracy_delta < 0

    def value(self, attribute: str) -> Optional[float]:
        return getattr(self, attribute, None)


def rows_from_entries(entries: list[ExperimentHistoryEntry]) -> list[MethodRow]:
    """Flatten history entries into plot-ready rows, dropping unscored runs."""
    rows: list[MethodRow] = []
    for entry in entries:
        evaluation = entry.evaluation
        if evaluation is None:
            continue
        rows.append(
            MethodRow(
                method=entry.experiment.tl_algorithm,
                accuracy=evaluation.accuracy,
                precision=evaluation.precision,
                recall=evaluation.recall,
                f1_score=evaluation.f1_score,
                auc=evaluation.auc,
                mmd_before=evaluation.mmd_before,
                mmd_after=evaluation.mmd_after,
                mmd_change=evaluation.mmd_change_percent,
                overlap_before=evaluation.overlap_before,
                overlap_after=evaluation.overlap_after,
                overlap_change=evaluation.overlap_change_points,
                alignment_status=evaluation.alignment_status,
                execution_seconds=evaluation.execution_seconds,
                baseline_accuracy=evaluation.baseline_accuracy,
                baseline_f1=evaluation.baseline_f1,
                accuracy_delta=evaluation.accuracy_delta,
                experiment_id=entry.experiment.id,
            )
        )
    return rows


def _fmt(value: Optional[float], places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def average_ranks(rows: list["MethodRow"]) -> dict[str, Optional[float]]:
    """Mean competition rank per method across every recorded metric.

    This is the aggregate the ranking view calls "Overall": a method that
    wins on accuracy but loses on F1 should not be declared best outright,
    and averaging its per-metric ranks is the usual way results tables
    settle that. Each method is averaged only over the metrics it actually
    has, so a missing AUC costs nothing rather than counting as a loss.

    The arithmetic itself is ``domain.ranking``'s, shared with the
    multi-method run's own ranking so the two comparison surfaces cannot
    disagree about which method won.
    """
    metric_rows = [
        {attribute: row.value(attribute) for _label, attribute in METRICS} for row in rows
    ]
    averages = average_metric_ranks(
        metric_rows, tuple(attribute for _label, attribute in METRICS)
    )
    return {row.method: average for row, average in zip(rows, averages)}


__all__ = [
    "BASELINE_LABEL",
    "_fmt",
    "METRICS",
    "METRIC_OVERALL",
    "METRIC_OVERALL_LABEL",
    "SORT_METRIC_ASC",
    "SORT_METRIC_DESC",
    "SORT_NAME",
    "SORT_OPTIONS",
    "SORT_TIME",
    "TABLE_COLUMNS",
    "VIEWS",
    "VIEW_ALIGNMENT",
    "VIEW_EXECUTION",
    "VIEW_NEGATIVE",
    "VIEW_PAIRS",
    "VIEW_PERFORMANCE",
    "VIEW_RANKING",
    "MethodRow",
    "average_ranks",
    "competition_ranks",
    "rows_from_entries",
]
