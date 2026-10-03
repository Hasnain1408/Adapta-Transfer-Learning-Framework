"""Formatting helpers shared by the catalogue dialogs."""

from __future__ import annotations

from datetime import datetime


def format_timestamp(value: object) -> str:
    """Render a stored timestamp in local time; "" for a missing value."""
    if isinstance(value, datetime):
        return value.astimezone().strftime("%Y-%m-%d %H:%M")
    return str(value or "")


__all__ = ["format_timestamp"]
