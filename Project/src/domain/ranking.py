"""Ordering methods against each other by their scores.

Lives in the domain because two different surfaces rank methods and must
agree on how: the Comparison tab, which ranks *stored* experiments pulled
from the database, and a multi-method run, which ranks the methods it just
executed. Two implementations of "which method won" that disagreed would
be worse than either.

The rule is competition ranking averaged across metrics rather than a
single headline number. A single metric is not enough on the kind of data
this app is usually pointed at: on an imbalanced defect-prediction set a
classifier that predicts the majority class for every sample scores well
on accuracy and zero on precision, recall, and F1. Ranking on accuracy
alone would crown it. Averaging its rank across all five metrics puts it
where it belongs.
"""

from __future__ import annotations

from typing import Optional

#: The scores a method is ranked on, in the order a reader meets them.
#: AUC is included but frequently absent (a classifier with no probability
#: output, a single-class target); a method is only ever averaged over the
#: metrics it actually has, so a missing AUC costs nothing.
RANKED_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1_score", "auc")


def competition_ranks(
    values: list[Optional[float]], higher_is_better: bool = True
) -> list[Optional[int]]:
    """Standard competition ranking (1, 2, 2, 4) over *values*.

    Ties share the better rank and consume the ones after it, which is what
    "joint second" means in a results table. ``None`` inputs are unranked
    (``None`` out) rather than ranked last: a method that never recorded a
    metric has not lost on it, and sorting it to the bottom would silently
    invent a comparison the data does not support.
    """
    scored = [(index, value) for index, value in enumerate(values) if value is not None]
    scored.sort(key=lambda pair: pair[1], reverse=higher_is_better)

    ranks: list[Optional[int]] = [None] * len(values)
    previous_value: Optional[float] = None
    previous_rank = 0
    for position, (index, value) in enumerate(scored, start=1):
        if previous_value is not None and value == previous_value:
            ranks[index] = previous_rank
        else:
            ranks[index] = position
            previous_rank = position
            previous_value = value
    return ranks


def average_metric_ranks(
    metric_rows: list[dict[str, Optional[float]]],
    metrics: tuple[str, ...] = RANKED_METRICS,
) -> list[Optional[float]]:
    """Mean competition rank per row across *metrics*. Lower is better.

    Each row is a ``{metric name: value}`` mapping. ``None`` comes back for
    a row with nothing rankable at all.
    """
    if len(metric_rows) == 1:
        scored = any(metric_rows[0].get(name) is not None for name in metrics)
        return [1.0 if scored else None]

    per_metric: list[list[Optional[int]]] = []
    for name in metrics:
        values = [row.get(name) for row in metric_rows]
        # A metric only one row recorded is not a contest — ranking it first
        # there would hand it a win nothing else could compete for and drag
        # its average down for free.
        if sum(value is not None for value in values) < 2:
            continue
        per_metric.append(competition_ranks(values))

    averages: list[Optional[float]] = []
    for index in range(len(metric_rows)):
        collected = [ranks[index] for ranks in per_metric if ranks[index] is not None]
        averages.append(sum(collected) / len(collected) if collected else None)
    return averages


__all__ = ["RANKED_METRICS", "average_metric_ranks", "competition_ranks"]
