"""Foundation styling: window surfaces, title bar, header, labels, group boxes.

The first fragment concatenated into the sheet, so anything here is the
default other fragments override.
"""

from __future__ import annotations

#: QSS fragment; concatenated into the one ``string.Template`` built in
#: ``style.py``. ``$token`` placeholders are substituted from the palette
#: there, so this stays a plain string and is never substituted alone.
BASE_QSS = """
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

/* ---------- Landing page ----------
   The page paints its own drifting backdrop, so #landingPage itself stays
   transparent here; only the text and the two hero buttons are styled. */
QWidget#landingPage {
    background: transparent;
}
QLabel#landingEyebrow {
    color: $accent;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#landingTitle {
    color: $text;
    font-size: 42px;
    font-weight: 800;
}
QLabel#landingTagline {
    color: $text;
    font-size: 17px;
    font-weight: 500;
}
QLabel#landingBlurb {
    color: $text_secondary;
    font-size: 13px;
}
QLabel#landingSection {
    color: $text_muted;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#landingStatus {
    color: $text_muted;
    font-size: 12px;
}
QFrame#landingFooter {
    background: transparent;
    border-top: 1px solid $border;
}
QPushButton#landingPrimaryButton {
    background-color: $accent;
    border: none;
    border-radius: 8px;
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
    padding: 12px 22px;
}
QPushButton#landingPrimaryButton:hover {
    background-color: $accent_hover;
}
QPushButton#landingPrimaryButton:pressed {
    background-color: $accent_pressed;
}
QPushButton#landingGhostButton {
    background-color: transparent;
    border: 1px solid $border_strong;
    border-radius: 8px;
    color: $text_secondary;
    font-size: 14px;
    font-weight: 600;
    padding: 12px 20px;
}
QPushButton#landingGhostButton:hover {
    border-color: $accent;
    color: $accent;
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

"""


__all__ = ["BASE_QSS"]
