"""Frameless drop-ins for ``QMessageBox``'s static convenience methods.

Same signatures as ``QMessageBox.warning`` / ``.critical`` / ``.information``
/ ``.question``, but the resulting dialog gets the app's custom title bar
(via :func:`title_bar.install_title_bar_on_grid`) instead of the OS-native
one, so it matches every other window in the app.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QMessageBox, QWidget

from presentation.qt.title_bar import install_title_bar_on_grid


def _show(
    icon: QMessageBox.Icon,
    parent: QWidget | None,
    title: str,
    text: str,
    buttons: QMessageBox.StandardButtons = QMessageBox.Ok,
    default_button: QMessageBox.StandardButton = QMessageBox.NoButton,
) -> QMessageBox.StandardButton:
    box = QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(buttons)
    if default_button != QMessageBox.NoButton:
        box.setDefaultButton(default_button)
    install_title_bar_on_grid(box, title)
    return box.exec_()


def warning(
    parent: QWidget | None,
    title: str,
    text: str,
    buttons: QMessageBox.StandardButtons = QMessageBox.Ok,
    default_button: QMessageBox.StandardButton = QMessageBox.NoButton,
) -> QMessageBox.StandardButton:
    return _show(QMessageBox.Warning, parent, title, text, buttons, default_button)


def critical(
    parent: QWidget | None,
    title: str,
    text: str,
    buttons: QMessageBox.StandardButtons = QMessageBox.Ok,
    default_button: QMessageBox.StandardButton = QMessageBox.NoButton,
) -> QMessageBox.StandardButton:
    return _show(QMessageBox.Critical, parent, title, text, buttons, default_button)


def information(
    parent: QWidget | None,
    title: str,
    text: str,
    buttons: QMessageBox.StandardButtons = QMessageBox.Ok,
    default_button: QMessageBox.StandardButton = QMessageBox.NoButton,
) -> QMessageBox.StandardButton:
    return _show(QMessageBox.Information, parent, title, text, buttons, default_button)


def question(
    parent: QWidget | None,
    title: str,
    text: str,
    buttons: QMessageBox.StandardButtons = QMessageBox.Yes | QMessageBox.No,
    default_button: QMessageBox.StandardButton = QMessageBox.NoButton,
) -> QMessageBox.StandardButton:
    return _show(QMessageBox.Question, parent, title, text, buttons, default_button)
