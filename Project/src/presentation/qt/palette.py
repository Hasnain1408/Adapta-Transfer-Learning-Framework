"""Colour tokens for the dark and light themes.

Split out of ``style.py`` so the palettes can be imported without pulling in
the whole QSS template. ``PALETTE`` is also read directly by widgets for the
colours that must go through the Qt/Matplotlib API rather than QSS (table
cell foregrounds, the status dot, chart series), which keeps one source of
truth for the theme.
"""

from __future__ import annotations

from pathlib import Path

_ASSETS = (Path(__file__).parent / "assets").as_posix()

DARK_PALETTE: dict[str, str] = {
    # Surfaces
    "page": "#202730", #"#292322",
    "surface": "#2A313B",
    "surface_alt": "#2A303D",
    "surface_sunken": "#2A303D",
    # Borders
    "border": "#262f3b",
    "border_strong": "#38424f",
    # Text
    "text": "#eaeff5",
    "text_secondary": "#9aa7b8",
    "text_muted": "#6b7787",
    "text_disabled": "#454e5b",
    # Accent (teal)
    "accent": "#2da798",
    "accent_hover": "#14a394",
    "accent_pressed": "#108a7d",
    "accent_soft": "#12332f",
    # Semantic
    "success": "#2fbf71",
    "danger": "#f0554a",
    "warning": "#f2ab3d",
    "neutral": "#9aa7b8",
    # Per-metric accent colours (used on the headline stat cards / charts)
    "metric_teal": "#2dd4bf",
    "metric_purple": "#a78bfa",
    "metric_amber": "#fbbf24",
    "metric_pink": "#f472b6",
    # Overlay surfaces
    "tooltip_bg": "#05070a",
    "tooltip_text": "#eaeff5",
}

LIGHT_PALETTE: dict[str, str] = {
    # Surfaces
    "page": "#dfe4ec",
    "surface": "#fffdfd",
    "surface_alt": "#f0f2f7",
    "surface_sunken": "#e7eaf1",
    # Borders
    "border": "#dde2eb",
    "border_strong": "#c5ccd9",
    # Text
    "text": "#171d26",
    "text_secondary": "#4c5867",
    "text_muted": "#788495",
    "text_disabled": "#aab2c0",
    # Accent (teal)
    "accent": "#0e9488",
    "accent_hover": "#0c8377",
    "accent_pressed": "#0a6f65",
    "accent_soft": "#dcf3f0",
    # Semantic
    "success": "#1a9c56",
    "danger": "#d33f33",
    "warning": "#c07f18",
    "neutral": "#4c5867",
    # Per-metric accent colours (used on the headline stat cards / charts)
    "metric_teal": "#0e9488",
    "metric_purple": "#7c5cf0",
    "metric_amber": "#d1930f",
    "metric_pink": "#d1478f",
    # Overlay surfaces
    "tooltip_bg": "#171d26",
    "tooltip_text": "#eaeff5",
}

#: Default/back-compat export — prefer ``DARK_PALETTE`` / ``LIGHT_PALETTE`` and
#: ``build_style_sheet`` for anything that needs to react to a theme switch.
PALETTE = DARK_PALETTE

__all__ = ["DARK_PALETTE", "LIGHT_PALETTE", "PALETTE"]
