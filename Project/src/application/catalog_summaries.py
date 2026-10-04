"""Pure helper functions for summarizing dataset characteristics and repository metrics."""

from __future__ import annotations

import numpy as np


def class_names(labels: np.ndarray | None) -> list[str]:
    """Extract distinct sorted class label names as strings from label arrays."""
    if labels is None:
        return []
    try:
        return [str(value) for value in np.unique(labels)]
    except (TypeError, ValueError):
        return sorted({str(value) for value in labels})


def dtype_summary(dtypes: list[str]) -> str:
    """Generate a compact data type summary string (such as float64 x24, int64 x2)."""
    if not dtypes:
        return ""
    counts: dict[str, int] = {}
    for dtype in dtypes:
        counts[dtype] = counts.get(dtype, 0) + 1
    return ", ".join(f"{dtype} x{count}" for dtype, count in counts.items())


def safe_count(repository: object) -> int:
    """Execute repository.count() safely, returning 0 on failure or if unsupported."""
    counter = getattr(repository, "count", None)
    if callable(counter):
        try:
            return int(counter())
        except Exception:
            return 0
    return 0


__all__ = ["class_names", "dtype_summary", "safe_count"]
