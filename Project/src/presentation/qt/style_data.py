"""Data-presentation surfaces: sidebar, metric cards, badges, tabs, tables, lists.

Never set font-weight in a QTabBar::tab rule — Qt measures the tab
with the base font and paints it with the heavier one, which clips the
label. Selection is signalled with colour and the bottom border instead.
"""

from __future__ import annotations

#: QSS fragment; concatenated into the one string.Template built in
#: style.py. $token placeholders are substituted from the palette
#: there, so this stays a plain string and is never substituted alone.
DATA_QSS = """/* ---------- Sidebar ---------- */
QWidget#sidebar {
    background-color: transparent;
}
/* The scroll viewport would otherwise paint the default window colour
   behind the parameter group and break the sidebar's transparency. */
QScrollArea#sidebarScroll, QScrollArea#sidebarScroll > QWidget > QWidget {
    background-color: transparent;
}

/* ---------- Metric cards ---------- */
QFrame#metricCard {
    background-color: $surface;
    border: 1px solid $border;
    border-top: 3px solid $border_strong;
    border-radius: 12px;
}
QFrame#metricCard[accent="teal"] {
    border-top: 3px solid $metric_teal;
}
QFrame#metricCard[accent="purple"] {
    border-top: 3px solid $metric_purple;
}
QFrame#metricCard[accent="amber"] {
    border-top: 3px solid $metric_amber;
}
QFrame#metricCard[accent="pink"] {
    border-top: 3px solid $metric_pink;
}
QLabel#metricCardTitle {
    color: $text_secondary;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QLabel#metricCardValue {
    color: $text;
    font-size: 24px;
    font-weight: 700;
}
QLabel#metricCardValue[placeholder="true"] {
    color: $text_disabled;
}

/* ---------- Status badge (experiment detail view) ---------- */
QLabel#statusBadge {
    border-radius: 11px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    background-color: $surface_alt;
    color: $text_secondary;
}
QLabel#statusBadge[status="completed"] {
    background-color: $accent_soft;
    color: $success;
}
QLabel#statusBadge[status="failed"] {
    background-color: $accent_soft;
    color: $danger;
}
QLabel#statusBadge[status="running"] {
    background-color: $accent_soft;
    color: $warning;
}
QLabel#statusBadge[status="pending"] {
    background-color: $accent_soft;
    color: $text_muted;
}

/* ---------- Tabs ---------- */
QTabWidget::pane {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 12px;
    top: -1px;
}
/* Do not set font-weight on tabs: Qt measures the tab with the base font but
   paints it with the heavier one, which clips the label. Selection is conveyed
   by colour and the underline instead. */
QTabBar::tab {
    background: transparent;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    color: $text_secondary;
    padding: 9px 18px;
    margin-right: 2px;
}
QTabBar::tab:hover:!selected {
    color: $text;
}
QTabBar::tab:selected {
    color: $accent;
    border-bottom: 2px solid $accent;
}
QTabBar::tab:disabled {
    color: $text_disabled;
}

/* ---------- Tables ---------- */
QTableWidget {
    background-color: $surface;
    alternate-background-color: $surface_alt;
    border: none;
    gridline-color: $border;
    color: $text;
}
QTableWidget::item {
    padding: 6px 8px;
    border: none;
}
QTableWidget::item:selected {
    background-color: $accent_soft;
    color: $text;
}
QHeaderView {
    background-color: $surface_sunken;
}
QHeaderView::section {
    background-color: $surface_sunken;
    color: $text_secondary;
    padding: 8px;
    font-size: 12px;
    font-weight: 600;
    border: none;
    border-bottom: 1px solid $border_strong;
    border-right: 1px solid $border;
}
QHeaderView::section:last {
    border-right: none;
}
QTableCornerButton::section {
    background-color: $surface_sunken;
    border: none;
}

/* ---------- List widgets (batch dataset file list) ---------- */
QListWidget {
    background-color: $surface;
    alternate-background-color: $surface_alt;
    border: 1px solid $border_strong;
    border-radius: 7px;
    color: $text;
    outline: none;
    padding: 4px;
}
QListWidget::item {
    padding: 5px 8px;
    border: none;
}
QListWidget::item:hover {
    background-color: $surface_alt;
}
QListWidget::item:selected {
    background-color: $accent_soft;
    color: $accent;
}

/* ---------- Check box ---------- */
QCheckBox {
    color: $text_secondary;
    spacing: 7px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid $border_strong;
    border-radius: 4px;
    background-color: $surface_alt;
}
QCheckBox::indicator:hover {
    border-color: $accent;
}
QCheckBox::indicator:checked {
    background-color: $accent;
    border-color: $accent;
    image: url($assets/check.svg);
}

"""


__all__ = ["DATA_QSS"]
