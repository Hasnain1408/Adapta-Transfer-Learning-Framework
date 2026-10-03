"""Interactive controls: text inputs, combo boxes, spin boxes, buttons.

Two Qt traps live in here; don't undo them. Styling ``QComboBox`` at all
suppresses the platform drop-down arrow, so an explicit ``::down-arrow``
image is required (hence the SVGs under ``assets/``), and the same goes
for spin-box arrows.
"""

from __future__ import annotations

#: QSS fragment; concatenated into the one ``string.Template`` built in
#: ``style.py``. ``$token`` placeholders are substituted from the palette
#: there, so this stays a plain string and is never substituted alone.
CONTROLS_QSS = """/* ---------- Text inputs ---------- */
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
/* Quit the app. Deliberately quieter than #dangerButton at rest — it shares
   the sidebar with Cancel, and two red outlines there would read as two ways
   of stopping the same thing. It only takes on the danger colour on hover,
   once the pointer is already committed to it. */
QPushButton#exitButton {
    background-color: transparent;
    border: 1px solid $border_strong;
    color: $text_muted;
    font-weight: 600;
}
QPushButton#exitButton:hover {
    background-color: $danger;
    border-color: $danger;
    color: #ffffff;
}
QPushButton#exitButton:pressed {
    background-color: $danger;
    border-color: $danger;
    color: #ffffff;
}
/* Back to the landing page, top-left of the workspace header. */
QPushButton#homeButton {
    background-color: transparent;
    border: 1px solid $border_strong;
    border-radius: 8px;
    color: $text_secondary;
    font-weight: 600;
    padding: 7px 14px;
}
QPushButton#homeButton:hover {
    border-color: $accent;
    color: $accent;
}
QPushButton#homeButton:pressed {
    background-color: $accent_soft;
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

"""


__all__ = ["CONTROLS_QSS"]
