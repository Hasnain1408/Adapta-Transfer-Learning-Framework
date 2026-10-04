"""The app's feedback dialogs: success, error, warning, info, confirm.

Replaces QMessageBox's static helpers. Three things the stock dialog
could not give us:

- **Its own visual language.** QMessageBox draws the *platform's*
  icons (a Windows-95-lineage yellow triangle, a glossy red cross), which
  land in this app's flat teal dashboard looking pasted in from another
  decade. Here the icon is a flat badge coloured from style.PALETTE,
  so it follows a theme switch like every other widget.
- **A success state.** QMessageBox has no "it worked" icon — the
  convention is to reuse Information, so "report exported" and "log in
  is unavailable" arrived looking identical. Success is its own kind here.
- **Somewhere to put raw exception text.** Driver errors ("connection to
  server at ... port 5432 failed") are the wrong first thing to read, but
  the right thing to be able to copy into a bug report. They go behind a
  disclosure toggle instead of into the headline.

Every message is written as **headline + detail**: the headline says what
happened in one plain sentence, the detail says what to do about it.
Detail is optional; a headline that needs no follow-up is a finished
message on its own.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from presentation.qt.title_bar import install_title_bar

#: Semantic kind -> (badge glyph, window-title prefix used when the caller
#: gives no explicit title). The glyphs are plain geometric characters
#: rather than emoji: emoji render at a different baseline and weight on
#: every platform, and colour is already carrying the meaning here.
KIND_SUCCESS = "success"
KIND_ERROR = "error"
KIND_WARNING = "warning"
KIND_INFO = "info"
KIND_QUESTION = "question"

_GLYPHS = {
    KIND_SUCCESS: "✓",
    KIND_ERROR: "✕",
    KIND_WARNING: "!",
    KIND_INFO: "i",
    KIND_QUESTION: "?",
}


class MessageDialog(QDialog):
    """One feedback dialog: badge, headline, detail, optional raw text."""

    def __init__(
        self,
        kind: str,
        title: str,
        headline: str,
        detail: str = "",
        technical: str = "",
        confirm_label: str = "OK",
        cancel_label: str = "",
        destructive: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setObjectName("messageDialog")

        body = QWidget()
        body.setObjectName("messageBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        layout.addLayout(self._build_message_row(kind, headline, detail))

        self._technical_box: QPlainTextEdit | None = None
        if technical:
            layout.addLayout(self._build_technical_section(technical))

        layout.addLayout(
            self._build_button_row(confirm_label, cancel_label, destructive)
        )

        install_title_bar(self, title, body)
        # Wide enough that a filesystem path or a driver error wraps to two
        # or three lines instead of one long ribbon, but not so wide it
        # dwarfs a three-word confirmation.
        self.setMinimumWidth(430)
        self.setMaximumWidth(560)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_message_row(self, kind: str, headline: str, detail: str) -> QHBoxLayout:
        badge = QLabel(_GLYPHS.get(kind, _GLYPHS[KIND_INFO]))
        badge.setObjectName("messageBadge")
        # The kind drives colour purely through QSS property selectors, so a
        # theme switch repaints it without this module knowing any colours.
        badge.setProperty("kind", kind)
        badge.setFixedSize(34, 34)
        badge.setAlignment(Qt.AlignCenter)

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(5)

        headline_label = QLabel(headline)
        headline_label.setObjectName("messageHeadline")
        headline_label.setWordWrap(True)
        # Selectable so a user can quote the exact wording when asking for
        # help, without being able to edit it.
        headline_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_column.addWidget(headline_label)

        if detail:
            detail_label = QLabel(detail)
            detail_label.setObjectName("messageDetail")
            detail_label.setWordWrap(True)
            detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            text_column.addWidget(detail_label)

        # No trailing stretch: the labels already size to their wrapped text,
        # and stretching the column just opens a gap above whatever follows
        # (the technical disclosure) that reads as a layout mistake.

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)
        row.addWidget(badge, 0, Qt.AlignTop)
        row.addLayout(text_column, 1)
        return row

    def _build_technical_section(self, technical: str) -> QVBoxLayout:
        """A disclosure toggle over the raw error text.

        Collapsed by default: the headline above already says what went
        wrong in the user's terms, and a stack of driver jargon competing
        with it is what made the old dialogs unreadable. Kept one click
        away (and selectable) because it is exactly what a bug report
        needs.
        """
        toggle = QPushButton("▸  Technical details")
        toggle.setObjectName("messageDisclosure")
        toggle.setCursor(Qt.PointingHandCursor)
        toggle.setCheckable(True)
        toggle.setFlat(True)

        box = QPlainTextEdit(technical.strip())
        box.setObjectName("messageTechnical")
        box.setReadOnly(True)
        box.setFixedHeight(96)
        box.hide()
        self._technical_box = box

        def on_toggled(checked: bool) -> None:
            toggle.setText(("▾  " if checked else "▸  ") + "Technical details")
            box.setVisible(checked)
            # The dialog was sized for the collapsed state; let it shrink
            # back rather than leaving a blank gap when re-collapsed.
            self.adjustSize()

        toggle.toggled.connect(on_toggled)

        section = QVBoxLayout()
        section.setContentsMargins(48, 0, 0, 0)  # aligned under the text column
        section.setSpacing(6)
        section.addWidget(toggle, 0, Qt.AlignLeft)
        section.addWidget(box)
        return section

    def _build_button_row(
        self, confirm_label: str, cancel_label: str, destructive: bool
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addStretch(1)

        if cancel_label:
            cancel = QPushButton(cancel_label)
            cancel.setObjectName("dialogSecondaryButton")
            cancel.setCursor(Qt.PointingHandCursor)
            cancel.clicked.connect(self.reject)
            row.addWidget(cancel)

        confirm = QPushButton(confirm_label)
        confirm.setObjectName(
            "dialogDangerButton" if destructive else "dialogPrimaryButton"
        )
        confirm.setCursor(Qt.PointingHandCursor)
        confirm.clicked.connect(self.accept)
        row.addWidget(confirm)

        # Enter activates the confirm button; Esc rejects (QDialog default).
        # On a destructive prompt the safe choice takes focus instead, so a
        # reflexive Enter cancels rather than deletes.
        if destructive and cancel_label:
            self._default_focus = row.itemAt(1).widget()
        else:
            confirm.setDefault(True)
            confirm.setAutoDefault(True)
            self._default_focus = confirm
        return row

    def showEvent(self, event):  # noqa: N802 - Qt naming
        super().showEvent(event)
        focus_target = getattr(self, "_default_focus", None)
        if focus_target is not None:
            focus_target.setFocus()


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------
def _run(
    kind: str,
    parent: QWidget | None,
    title: str,
    headline: str,
    detail: str = "",
    technical: str = "",
    confirm_label: str = "OK",
    cancel_label: str = "",
    destructive: bool = False,
) -> bool:
    dialog = MessageDialog(
        kind,
        title,
        headline,
        detail=detail,
        technical=technical,
        confirm_label=confirm_label,
        cancel_label=cancel_label,
        destructive=destructive,
        parent=parent,
    )
    # Dialogs opened from a frameless parent inherit no stylesheet of their
    # own; the app-wide sheet covers them, but a parentless dialog (used by
    # a few call sites and by tests) needs it applied explicitly.
    app = QApplication.instance()
    if parent is None and app is not None:
        dialog.setStyleSheet(app.styleSheet())
    return dialog.exec_() == QDialog.Accepted


def success(
    parent: QWidget | None, title: str, headline: str, detail: str = ""
) -> bool:
    """An action completed. Use for outcomes the user asked for and got."""
    return _run(KIND_SUCCESS, parent, title, headline, detail)


def error(
    parent: QWidget | None,
    title: str,
    headline: str,
    detail: str = "",
    technical: str = "",
) -> bool:
    """Something failed. *technical* holds raw exception text, if any."""
    return _run(KIND_ERROR, parent, title, headline, detail, technical=technical)


def warn(
    parent: QWidget | None, title: str, headline: str, detail: str = ""
) -> bool:
    """The action can't proceed yet — usually missing or invalid input."""
    return _run(KIND_WARNING, parent, title, headline, detail)


def info(parent: QWidget | None, title: str, headline: str, detail: str = "") -> bool:
    """Neutral news: no action failed, nothing was completed."""
    return _run(KIND_INFO, parent, title, headline, detail)


def confirm(
    parent: QWidget | None,
    title: str,
    headline: str,
    detail: str = "",
    confirm_label: str = "Continue",
    destructive: bool = False,
) -> bool:
    """Ask before doing something the user cannot undo.

    Returns True only if they chose the confirm button. *confirm_label*
    should name the action ("Delete", "Clear Everything") rather than say
    "OK" — a labelled verb is what makes the consequence readable at a
    glance from the button itself.
    """
    return _run(
        KIND_QUESTION,
        parent,
        title,
        headline,
        detail,
        confirm_label=confirm_label,
        cancel_label="Cancel",
        destructive=destructive,
    )


# ----------------------------------------------------------------------
# Back-compat shims
# ----------------------------------------------------------------------
# The previous API mirrored QMessageBox's statics. Call sites have all been
# migrated to the semantic helpers above; these remain so any straggler (or
# a test that patches them) keeps working rather than raising AttributeError.
def warning(
    parent: QWidget | None,
    title: str,
    text: str,
    buttons=None,
    default_button=None,
) -> QMessageBox.StandardButton:
    warn(parent, title, text)
    return QMessageBox.Ok


def critical(
    parent: QWidget | None,
    title: str,
    text: str,
    buttons=None,
    default_button=None,
) -> QMessageBox.StandardButton:
    error(parent, title, text)
    return QMessageBox.Ok


def information(
    parent: QWidget | None,
    title: str,
    text: str,
    buttons=None,
    default_button=None,
) -> QMessageBox.StandardButton:
    info(parent, title, text)
    return QMessageBox.Ok


def question(
    parent: QWidget | None,
    title: str,
    text: str,
    buttons=None,
    default_button=None,
) -> QMessageBox.StandardButton:
    accepted = confirm(parent, title, text)
    return QMessageBox.Yes if accepted else QMessageBox.No


__all__ = [
    "MessageDialog",
    "confirm",
    "critical",
    "error",
    "info",
    "information",
    "question",
    "success",
    "warn",
    "warning",
]
