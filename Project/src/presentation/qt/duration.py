"""One way to write a duration, shared by the UI and the exported report.

Kept separate from both so the status bar, the batch table, the Feature
Space metadata panel, and the PDF cannot drift into formatting the same
number four different ways.
"""

from __future__ import annotations


def format_duration(seconds: float | None) -> str:
    """A run length as a short human string; "" for missing or negative.

    Sub-second runs keep three decimals because the fast methods (CORAL,
    GFK) finish in a few milliseconds on small datasets, and rounding those
    to "0.0s" would make every one of them look identical.
    """
    if seconds is None or seconds < 0:
        return ""
    if seconds < 1:
        return f"{seconds:.3f}s"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


__all__ = ["format_duration"]
