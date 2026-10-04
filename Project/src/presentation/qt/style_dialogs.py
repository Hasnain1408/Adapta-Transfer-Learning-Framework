"""Feedback dialogs — the badge, headline, detail, and buttons of
message_box.MessageDialog, keyed off its kind property.
"""

from __future__ import annotations

#: QSS fragment; concatenated into the one string.Template built in
#: style.py. $token placeholders are substituted from the palette
#: there, so this stays a plain string and is never substituted alone.
DIALOGS_QSS = """/* ---------- Feedback dialogs (message_box.MessageDialog) ----------
   The badge replaces QMessageBox's platform icon. Its colour comes from
   the kind property so a theme switch repaints it automatically and
   message_box.py never names a colour itself. Each kind gets a tinted
   disc rather than a saturated block: at 34px a full-strength fill next
   to body text reads as an alarm even when the message is a success. */
QWidget#messageBody {
    background-color: $page;
}
QLabel#messageBadge {
    border-radius: 17px;
    font-size: 17px;
    font-weight: 700;
}
QLabel#messageBadge[kind="success"] {
    background-color: $accent_soft;
    color: $success;
}
QLabel#messageBadge[kind="error"] {
    background-color: $surface_alt;
    color: $danger;
}
QLabel#messageBadge[kind="warning"] {
    background-color: $surface_alt;
    color: $warning;
}
QLabel#messageBadge[kind="info"] {
    background-color: $accent_soft;
    color: $accent;
}
QLabel#messageBadge[kind="question"] {
    background-color: $accent_soft;
    color: $accent;
}
QLabel#messageHeadline {
    color: $text;
    font-size: 14px;
    font-weight: 600;
}
QLabel#messageDetail {
    color: $text_secondary;
    font-size: 12px;
}
QPushButton#messageDisclosure {
    background: transparent;
    border: none;
    color: $text_muted;
    font-size: 11px;
    padding: 2px 0px;
    text-align: left;
}
QPushButton#messageDisclosure:hover {
    color: $accent;
}
QPlainTextEdit#messageTechnical {
    background-color: $surface_sunken;
    border: 1px solid $border;
    border-radius: 6px;
    color: $text_secondary;
    font-family: "DejaVu Sans Mono", "Menlo", monospace;
    font-size: 10px;
    padding: 6px;
}

/* Dialog buttons. Primary carries the accent, secondary stays quiet, and
   danger is reserved for the confirm side of a destructive prompt so the
   irreversible choice never looks like the routine one. */
QPushButton#dialogPrimaryButton,
QPushButton#dialogSecondaryButton,
QPushButton#dialogDangerButton {
    border-radius: 7px;
    font-size: 13px;
    font-weight: 600;
    min-height: 22px;
    padding: 7px 20px;
}
QPushButton#dialogPrimaryButton {
    background-color: $accent;
    border: none;
    color: #ffffff;
}
QPushButton#dialogPrimaryButton:hover {
    background-color: $accent_hover;
}
QPushButton#dialogPrimaryButton:pressed {
    background-color: $accent_pressed;
}
QPushButton#dialogSecondaryButton {
    background-color: transparent;
    border: 1px solid $border_strong;
    color: $text_secondary;
}
QPushButton#dialogSecondaryButton:hover {
    border-color: $text_muted;
    color: $text;
}
QPushButton#dialogDangerButton {
    background-color: $danger;
    border: none;
    color: #ffffff;
}
QPushButton#dialogDangerButton:hover {
    background-color: $danger;
    border: none;
}

"""


__all__ = ["DIALOGS_QSS"]
