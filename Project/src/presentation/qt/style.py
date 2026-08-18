"""Central visual language for the desktop app.

Everything visual lives here: the colour tokens in ``PALETTE`` (defined in
``palette.py``) and the single ``STYLE_SHEET`` applied to the main window.
Widgets are targeted by object name (``runButton``, ``metricCard``, ...) so
no module needs an inline stylesheet.

The theme is a dark, card-based dashboard look (dark navy surfaces, a teal
accent, per-card colour accents) so the app reads like a professional
analytics workbench rather than a plain form.

This module assembles the sheet; the QSS itself is split by subject across
sibling modules, concatenated here in the order they must cascade:

- ``style_base`` — window surfaces, title bar, header, labels, group boxes.
- ``style_controls`` — text inputs, combo/spin boxes, buttons.
- ``style_data`` — sidebar, metric cards, badges, tabs, tables, lists.
- ``style_chrome`` — menu bar, status bar, progress bars.
- ``style_dialogs`` — the feedback dialogs in ``message_box``.
- ``style_panels`` — processing strip, separators, scroll bars, tooltips.

Order matters: later fragments override earlier ones, so ``style_base``
comes first and the specific panels come last. Add a rule to the fragment
that owns its subject rather than appending to whichever file is open.

``PALETTE`` is re-exported here for the modules that import it from
``presentation.qt.style``; it is also imported by ``main_window`` and
``charts_widget`` for the colours that must be applied through the
Qt/Matplotlib API rather than QSS (table cell foregrounds, the status dot,
chart series), which keeps a single source of truth for the theme.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from presentation.qt.palette import DARK_PALETTE, LIGHT_PALETTE, PALETTE
from presentation.qt.style_base import BASE_QSS
from presentation.qt.style_chrome import CHROME_QSS
from presentation.qt.style_controls import CONTROLS_QSS
from presentation.qt.style_data import DATA_QSS
from presentation.qt.style_dialogs import DIALOGS_QSS
from presentation.qt.style_panels import PANELS_QSS

_ASSETS = (Path(__file__).parent / "assets").as_posix()

# Qt style sheets are full of braces, so ``str.format`` is unusable here.
# ``string.Template`` only treats ``$`` as special, and QSS never uses it.
_TEMPLATE = Template(
    BASE_QSS + CONTROLS_QSS + DATA_QSS + CHROME_QSS + DIALOGS_QSS + PANELS_QSS
)


def build_style_sheet(palette: dict[str, str]) -> str:
    """Render the QSS template for a given palette (dark or light)."""
    return _TEMPLATE.substitute(assets=_ASSETS, **palette)


STYLE_SHEET = build_style_sheet(DARK_PALETTE)

__all__ = [
    "DARK_PALETTE",
    "LIGHT_PALETTE",
    "PALETTE",
    "STYLE_SHEET",
    "build_style_sheet",
]
