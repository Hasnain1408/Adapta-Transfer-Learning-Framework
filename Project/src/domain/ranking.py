"""Method performance ranking algorithms across multi-metric benchmarks."""

from __future__ import annotations

from typing import Optional

# Standard evaluation metrics used for competition rank averaging.
RANKED_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1_score", "auc")


def competition_ranks(
    values: list[Optional[float]], higher_is_better: bool = True
) -> list[Optional[int]]:
    """Compute standard 1224 competition ranks for a list of score values.

    Ties receive the shared higher rank and consume subsequent positions.
    None inputs remain unranked and return None.
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
    """Calculate mean competition rank across metrics for each method row.

    Lower average rank indicates superior overall performance across metrics.
    """
    if len(metric_rows) == 1:
        scored = any(metric_rows[0].get(name) is not None for name in metrics)
        return [1.0 if scored else None]

    per_metric: list[list[Optional[int]]] = []
    for name in metrics:
        values = [row.get(name) for row in metric_rows]
        if sum(value is not None for value in values) < 2:
            continue
        per_metric.append(competition_ranks(values))

    averages: list[Optional[float]] = []
    for index in range(len(metric_rows)):
        collected = [ranks[index] for ranks in per_metric if ranks[index] is not None]
        averages.append(sum(collected) / len(collected) if collected else None)
    return averages


__all__ = ["RANKED_METRICS", "average_metric_ranks", "competition_ranks"]
