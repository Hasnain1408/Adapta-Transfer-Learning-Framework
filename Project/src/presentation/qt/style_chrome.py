"""Window chrome: the menu bar, the status bar, and the progress bars.

The run progress bar carries cycleColor 0-3 and paused dynamic
properties, set from main_window_progress.py; each combination needs
its own rule here because QSS cannot compute a colour from a property.
"""

from __future__ import annotations

#: QSS fragment; concatenated into the one string.Template built in
#: style.py. $token placeholders are substituted from the palette
#: there, so this stays a plain string and is never substituted alone.
CHROME_QSS = """/* ---------- Menu bar ---------- */
QMenuBar {
    background-color: $surface;
    border-bottom: 1px solid $border;
    padding: 2px 6px;
}
QMenuBar::item {
    background: transparent;
    padding: 6px 11px;
    border-radius: 5px;
    color: $text;
}
QMenuBar::item:selected {
    background-color: $accent_soft;
    color: $accent;
}
QMenu {
    background-color: $surface_alt;
    border: 1px solid $border_strong;
    border-radius: 8px;
    padding: 5px;
}
QMenu::item {
    padding: 7px 26px 7px 16px;
    border-radius: 5px;
    color: $text;
}
QMenu::item:selected {
    background-color: $accent_soft;
    color: $accent;
}
QMenu::item:disabled {
    color: $text_disabled;
}
QMenu::separator {
    height: 1px;
    background-color: $border;
    margin: 5px 8px;
}

/* ---------- Status bar ---------- */
QStatusBar {
    background-color: $surface;
    border-top: 1px solid $border;
    color: $text_secondary;
    min-height: 28px;
}
QStatusBar::item {
    border: none;
}
QStatusBar QLabel {
    color: $text_secondary;
    font-size: 12px;
}

QProgressBar {
    background-color: $surface_sunken;
    border: none;
    border-radius: 3px;
    max-height: 6px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: $accent;
    border-radius: 3px;
}

/* Run progress bar: wider and slightly taller than the generic bar above so
   it reads clearly in the status bar. While running, its chunk colour steps
   through cycleColor 0-3 once a second (see _advance_progress_bar_color,
   which sets that property on a 1s QTimer); while paused, the timer is
   stopped and the [paused="true"] rule below takes over instead -- it's
   declared last so it wins the cascade regardless of whatever cycleColor
   was left on when the timer stopped, keeping "paused" visually distinct
   from "still cycling". */
QProgressBar#runProgressBar {
    background-color: $surface_sunken;
    border: none;
    border-radius: 4px;
    max-height: 9px;
    text-align: center;
}
QProgressBar#runProgressBar::chunk {
    background-color: $accent;
    border-radius: 4px;
}
QProgressBar#runProgressBar[cycleColor="0"]::chunk {
    background-color: #20D046;
}
QProgressBar#runProgressBar[cycleColor="1"]::chunk {
    background-color: #FFDF20;
}
QProgressBar#runProgressBar[cycleColor="2"]::chunk {
    background-color: #3C48D3;
}
QProgressBar#runProgressBar[cycleColor="3"]::chunk {
    background-color: #000080;
}
QProgressBar#runProgressBar[paused="true"]::chunk {
    background-color: #FF0000;
}

"""


__all__ = ["CHROME_QSS"]
