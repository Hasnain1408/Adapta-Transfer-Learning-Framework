"""Central visual language for the desktop app.

Everything visual lives here: the colour tokens in ``PALETTE`` and the single
``STYLE_SHEET`` applied to the main window. Widgets are targeted by object name
(``runButton``, ``metricCard``, ...) so no module needs an inline stylesheet.

``PALETTE`` is also imported by ``main_window`` for the few colours that must be
applied through the Qt API rather than QSS (table cell foregrounds, the status
dot), which keeps a single source of truth for the theme.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

_ASSETS = (Path(__file__).parent / "assets").as_posix()

PALETTE: dict[str, str] = {
    # Surfaces
    "page": "#f4f6f8",
    "surface": "#ffffff",
    "surface_alt": "#fafbfc",
    "surface_sunken": "#eef1f4",
    # Borders
    "border": "#dde2e8",
    "border_strong": "#c3ccd6",
    # Text
    "text": "#1a2229",
    "text_secondary": "#5b6b7c",
    "text_muted": "#8a97a5",
    "text_disabled": "#b6bfc9",
    # Accent
    "accent": "#1e6fd9",
    "accent_hover": "#1a60bd",
    "accent_pressed": "#164f9e",
    "accent_soft": "#e8f0fc",
    # Semantic
    "success": "#14804a",
    "danger": "#b3261e",
    "warning": "#8a5300",
    "neutral": "#5b6b7c",
}

# Qt style sheets are full of braces, so ``str.format`` is unusable here.
# ``string.Template`` only treats ``$`` as special, and QSS never uses it.
_TEMPLATE = Template(
    """
/* ---------- Base ---------- */
QWidget {
    font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', 'DejaVu Sans', Arial, sans-serif;
    font-size: 13px;
    color: $text;
}
QMainWindow, QWidget#centralWidget {
    background-color: $page;
}

/* ---------- Labels ---------- */
QLabel {
    color: $text;
    background: transparent;
}
QLabel#fieldLabel {
    color: $text_secondary;
    font-size: 13px;
    font-weight: 500;
}
QLabel#fieldLabel:disabled {
    color: $text_disabled;
}
QLabel#captionLabel {
    color: $text_secondary;
    font-size: 12px;
}
QLabel#emptyStateTitle {
    color: $text_secondary;
    font-size: 15px;
    font-weight: 600;
}
QLabel#emptyStateBody {
    color: $text_muted;
    font-size: 13px;
}

/* ---------- Group boxes ---------- */
QGroupBox {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 8px;
    margin-top: 10px;
    padding: 14px 14px 12px 14px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: $text;
    font-size: 13px;
    font-weight: 600;
    background-color: $page;
}

/* ---------- Text inputs ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: $surface;
    border: 1px solid $border_strong;
    border-radius: 6px;
    color: $text;
    padding: 6px 10px;
    min-height: 20px;
    selection-background-color: $accent;
    selection-color: #ffffff;
}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: $text_muted;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: $accent;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: $surface_sunken;
    border-color: $border;
    color: $text_disabled;
}
QLineEdit[readOnly="true"] {
    background-color: $surface_alt;
    color: $text_secondary;
}

/* The drop-down indicator must be given an explicit image: styling QComboBox
   at all suppresses the platform arrow, leaving the widget looking like a
   plain text field. */
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 26px;
    border: none;
    background: transparent;
}
QComboBox::down-arrow {
    image: url($assets/chevron_down.svg);
    width: 12px;
    height: 12px;
}
QComboBox::down-arrow:disabled {
    image: url($assets/chevron_down_disabled.svg);
}
QComboBox QAbstractItemView {
    background-color: $surface;
    border: 1px solid $border_strong;
    border-radius: 6px;
    outline: none;
    padding: 4px;
    color: $text;
    selection-background-color: $accent_soft;
    selection-color: $accent;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    width: 20px;
    border: none;
    background: transparent;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-position: top right;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-position: bottom right;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url($assets/spin_up.svg);
    width: 8px;
    height: 8px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url($assets/spin_down.svg);
    width: 8px;
    height: 8px;
}

/* ---------- Buttons ---------- */
QPushButton {
    background-color: $surface;
    border: 1px solid $border_strong;
    border-radius: 6px;
    color: $text;
    padding: 7px 16px;
    font-weight: 500;
    min-height: 20px;
}
QPushButton:hover {
    background-color: $surface_alt;
    border-color: $text_muted;
}
QPushButton:pressed {
    background-color: $surface_sunken;
}
QPushButton:disabled {
    background-color: $surface_sunken;
    border-color: $border;
    color: $text_disabled;
}
QPushButton:focus {
    border-color: $accent;
}
QPushButton#runButton {
    background-color: $accent;
    border: 1px solid $accent;
    color: #ffffff;
    font-weight: 600;
    padding: 8px 22px;
}
QPushButton#runButton:hover {
    background-color: $accent_hover;
    border-color: $accent_hover;
}
QPushButton#runButton:pressed {
    background-color: $accent_pressed;
    border-color: $accent_pressed;
}
QPushButton#runButton:disabled {
    background-color: $surface_sunken;
    border-color: $border;
    color: $text_disabled;
}

/* ---------- Metric cards ---------- */
QFrame#metricCard {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 8px;
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
    font-size: 22px;
    font-weight: 600;
}
QLabel#metricCardValue[placeholder="true"] {
    color: $text_disabled;
}

/* ---------- Tabs ---------- */
QTabWidget::pane {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 8px;
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
    padding: 8px 18px;
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
    background-color: $surface;
}
QCheckBox::indicator:hover {
    border-color: $accent;
}
QCheckBox::indicator:checked {
    background-color: $accent;
    border-color: $accent;
    image: url($assets/check.svg);
}

/* ---------- Menu bar ---------- */
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
    background-color: $surface;
    border: 1px solid $border_strong;
    border-radius: 6px;
    padding: 5px;
}
QMenu::item {
    padding: 7px 26px 7px 16px;
    border-radius: 4px;
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

/* ---------- Separators ---------- */
QFrame#divider {
    background-color: $border;
    max-height: 1px;
    min-height: 1px;
    border: none;
}
QFrame#vDivider {
    background-color: $border;
    max-width: 1px;
    min-width: 1px;
    border: none;
}

/* ---------- Scroll bars ---------- */
QScrollBar:vertical {
    background: transparent;
    width: 11px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: $border_strong;
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover {
    background: $text_muted;
}
QScrollBar:horizontal {
    background: transparent;
    height: 11px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: $border_strong;
    border-radius: 5px;
    min-width: 28px;
}
QScrollBar::handle:horizontal:hover {
    background: $text_muted;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}

/* ---------- Tooltips ---------- */
QToolTip {
    background-color: $text;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 9px;
    font-size: 12px;
}
"""
)

STYLE_SHEET = _TEMPLATE.substitute(assets=_ASSETS, **PALETTE)
