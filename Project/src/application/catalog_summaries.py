"""Small pure summarisers used while cataloguing a dataset.

All three are deliberately forgiving: they run *after* a run has already
succeeded, so a bookkeeping edge case (an oddly typed label column, a
repository without ``count``) must degrade to a sensible value rather than
turn a finished experiment into an error.
"""

from __future__ import annotations

import numpy as np


def class_names(labels: np.ndarray | None) -> list[str]:
    """Distinct label values, sorted, as strings.

    ``np.unique`` on a mixed-type object array raises, so the fallback keeps
    catalogue registration working for oddly typed label columns instead of
    failing the whole recording step.
    """
    if labels is None:
        return []
    try:
        return [str(value) for value in np.unique(labels)]
    except (TypeError, ValueError):
        return sorted({str(value) for value in labels})


def dtype_summary(dtypes: list[str]) -> str:
    """Compact ``"float64 x24, int64 x2"``-style summary for a table cell.

    Preserves first-seen order (``dict`` insertion order) rather than
    alphabetising, so a mostly-numeric file reads as e.g. "float64 x24, str
    x2" instead of burying the more common type behind the sort order.
    """
    if not dtypes:
        return ""
    counts: dict[str, int] = {}
    for dtype in dtypes:
        counts[dtype] = counts.get(dtype, 0) + 1
    return ", ".join(f"{dtype} x{count}" for dtype, count in counts.items())


def safe_count(repository: object) -> int:
    """Read ``repository.count()`` when it exists, else 0.

    ``count`` is an implementation convenience on the Postgres repositories
    rather than part of the domain protocols, so this stays tolerant of an
    implementation that omits it.
    """
    counter = getattr(repository, "count", None)
    if callable(counter):
        try:
            return int(counter())
        except Exception:
            return 0
    return 0


__all__ = ["class_names", "dtype_summary", "safe_count"]
