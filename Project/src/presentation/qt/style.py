"""Central visual language for the desktop app.

Everything visual lives here: the colour tokens in ``PALETTE`` and the single
``STYLE_SHEET`` applied to the main window. Widgets are targeted by object name
(``runButton``, ``metricCard``, ...) so no module needs an inline stylesheet.

The theme is a dark, card-based dashboard look (dark navy surfaces, a teal
accent, per-card colour accents) so the app reads like a professional
analytics workbench rather than a plain form.

``PALETTE`` is also imported by ``main_window`` and ``charts_widget`` for the
colours that must be applied through the Qt/Matplotlib API rather than QSS
(table cell foregrounds, the status dot, chart series), which keeps a single
source of truth for the theme.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

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
QMainWindow, QWidget#centralWidget, QDialog {
    background-color: $page;
}
QMainWindow, QDialog {
    border: 1px solid $border_strong;
}
QWidget#authCard {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 12px;
}

/* ---------- Custom title bar ---------- */
QWidget#titleBar {
    background-color: $surface;
}
QLabel#titleBarLabel {
    color: $text;
    font-weight: 600;
    font-size: 13px;
}
QPushButton#titleBarMinButton,
QPushButton#titleBarMaxButton,
QPushButton#titleBarCloseButton {
    background: transparent;
    border: none;
    color: $text_secondary;
    font-size: 14px;
}
QPushButton#titleBarMinButton:hover,
QPushButton#titleBarMaxButton:hover {
    background-color: $surface_alt;
    color: $text;
}
QPushButton#titleBarCloseButton:hover {
    background-color: $danger;
    color: #ffffff;
}

/* ---------- Header / brand ---------- */
QLabel#brandMark {
    background-color: $accent;
    border-radius: 9px;
}
QLabel#appTitle {
    color: $text;
    font-size: 18px;
    font-weight: 700;
}
QLabel#appSubtitle {
    color: $text_muted;
    font-size: 12px;
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
    color: $text_muted;
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
    border-radius: 12px;
    margin-top: 12px;
    padding: 16px 16px 14px 16px;
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
    background-color: $surface_alt;
    border: 1px solid $border_strong;
    border-radius: 7px;
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
    background-color: $surface_sunken;
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
    background-color: $surface_alt;
    border: 1px solid $border_strong;
    border-radius: 7px;
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
    background-color: $surface_alt;
    border: 1px solid $border_strong;
    border-radius: 7px;
    color: $text;
    padding: 7px 16px;
    font-weight: 500;
    min-height: 20px;
}
QPushButton:hover {
    background-color: $surface;
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
    color: #04211d;
    font-weight: 700;
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
QPushButton#fullscreenButton {
    background-color: $accent_soft;
    border: 1px solid $accent;
    color: $accent;
    font-weight: 600;
    padding: 6px 14px;
}
QPushButton#fullscreenButton:hover {
    background-color: $accent;
    color: #04211d;
}
QPushButton#fullscreenButton:pressed {
    background-color: $accent_pressed;
    border-color: $accent_pressed;
    color: #ffffff;
}
QPushButton#fullscreenButton:disabled {
    background-color: $surface_sunken;
    border-color: $border;
    color: $text_disabled;
}
QPushButton#swapButton {
    background-color: $surface_alt;
    border: 1px solid $border_strong;
    color: $text;
    padding: 10px 12px;
    font-weight: 600;
}
QPushButton#swapButton:hover {
    background-color: $accent_soft;
    border-color: $accent;
    color: $accent;
}
QPushButton#swapButton:pressed {
    background-color: $surface_sunken;
}
/* Destructive actions (delete a user, an experiment, a catalogue entry).
   Outlined rather than filled so it reads as available but not inviting. */
QPushButton#dangerButton {
    background-color: transparent;
    border: 1px solid $danger;
    color: $danger;
    font-weight: 600;
}
QPushButton#dangerButton:hover {
    background-color: $danger;
    color: #ffffff;
}
QPushButton#dangerButton:pressed {
    background-color: $danger;
    color: #ffffff;
}
QPushButton#dangerButton:disabled {
    background-color: transparent;
    border-color: $border;
    color: $text_disabled;
}
QPushButton#linkButton {
    background-color: transparent;
    border: none;
    color: $accent;
    padding: 4px 6px;
    font-weight: 600;
}
QPushButton#linkButton:hover {
    color: $accent_hover;
    text-decoration: underline;
}
QPushButton#linkButton:pressed {
    color: $accent_pressed;
}
QPushButton#themeToggleButton {
    background-color: $surface_alt;
    border: 1px solid $border_strong;
    border-radius: 16px;
    color: $text_secondary;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton#themeToggleButton:hover {
    border-color: $accent;
    color: $accent;
}
QPushButton#themeToggleButton:pressed {
    background-color: $accent_soft;
}

/* ---------- Sidebar ---------- */
QWidget#sidebar {
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

/* ---------- Live processing status (above the results area) ----------
   Stage chips are distinguished by colour and their leading glyph only.
   Deliberately no font-weight switch on the active chip: re-weighting text
   Qt has already measured shifts every chip after it sideways on each
   stage change, which reads as the whole strip twitching. */
QFrame#processingStrip {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 8px;
}
QLabel#processingElapsed {
    color: $text_secondary;
    font-family: "DejaVu Sans Mono", "Menlo", monospace;
    font-size: 11px;
}
QLabel#processingCaption {
    color: $text_muted;
    font-size: 11px;
}
QLabel#stageArrow {
    color: $text_disabled;
    font-size: 11px;
    padding: 0px 6px;
}
QLabel#stageChip {
    font-size: 11px;
}
QLabel#stageChip[stageState="pending"] {
    color: $text_muted;
}
QLabel#stageChip[stageState="active"] {
    color: $accent;
}
QLabel#stageChip[stageState="done"] {
    color: $success;
}
QLabel#stageChip[stageState="skipped"] {
    color: $text_disabled;
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

/* ---------- Scroll areas ---------- */
/* Unstyled, QScrollArea (and the viewport widget Qt creates for it) paint
   with the platform's base palette colour rather than the app's dark/light
   surface, which shows through as a mismatched panel wherever a detail view
   is wrapped in one for vertical scrolling. */
QScrollArea {
    background-color: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
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
    background: $accent;
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
    background: $accent;
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
    background-color: $tooltip_bg;
    color: $tooltip_text;
    border: 1px solid $border_strong;
    border-radius: 5px;
    padding: 6px 9px;
    font-size: 12px;
}

/* ---------- File picker dialog (dataset selection) ---------- */
/* The QFileDialog's internal widgets do not inherit QSS rules targeted at
   the rest of the app, so they fall back to the platform's light grey. */
QFileDialog QToolButton {
    background-color: $surface_alt;
    border: 1px solid $border_strong;
    border-radius: 5px;
    padding: 4px;
}
QFileDialog QToolButton:hover {
    background-color: $surface;
    border-color: $text_muted;
}
QFileDialog QToolButton:pressed {
    background-color: $surface_sunken;
}
QFileDialog QToolButton:disabled {
    background-color: $surface_sunken;
    border-color: $border;
}
QFileDialog QStackedWidget,
QFileDialog QSplitter,
QFileDialog QFrame,
QFileDialog QDialogButtonBox,
QFileDialog QSizeGrip {
    background-color: $surface;
}
QFileDialog QListView,
QFileDialog QTreeView {
    background-color: $surface;
    border: none;
    color: $text;
}
QFileDialog QListView::item,
QFileDialog QTreeView::item {
    padding: 4px 6px;
    border: none;
}
QFileDialog QListView::item:hover,
QFileDialog QTreeView::item:hover {
    background-color: $surface_alt;
}
QFileDialog QListView::item:selected,
QFileDialog QTreeView::item:selected {
    background-color: $accent_soft;
    color: $accent;
}
QComboBoxPrivateContainer {
    background-color: $surface_alt;
    border: 1px solid $border_strong;
    border-radius: 7px;
}
"""
)


def build_style_sheet(palette: dict[str, str]) -> str:
    """Render the QSS template for a given palette (dark or light)."""
    return _TEMPLATE.substitute(assets=_ASSETS, **palette)


STYLE_SHEET = build_style_sheet(DARK_PALETTE)
