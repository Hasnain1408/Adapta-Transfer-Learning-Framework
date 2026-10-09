"""Remaining panels and decorations: the live processing strip, separators,
scroll areas and bars, tooltips, and the file-picker dialog.
"""

from __future__ import annotations

#: QSS fragment; concatenated into the one ``string.Template`` built in
#: ``style.py``. ``$token`` placeholders are substituted from the palette
#: there, so this stays a plain string and is never substituted alone.
PANELS_QSS = """/* ---------- Live processing status (above the results area) ----------
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
/* ---------- Content splitter ----------
   The divider between the run summary and the results. Given a visible
   grip rather than left as bare background: an unmarked 10px gap is a
   control nobody discovers, and this one is the whole point of the split. */
QSplitter#contentSplitter::handle:vertical {
    background-color: transparent;
    border-top: 1px solid $border;
    margin: 4px 0;
    height: 5px;
}
QSplitter#contentSplitter::handle:vertical:hover {
    border-top: 2px solid $accent;
}
QSplitter#contentSplitter::handle:vertical:pressed {
    border-top: 2px solid $accent_pressed;
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
}"""


__all__ = ["PANELS_QSS"]
