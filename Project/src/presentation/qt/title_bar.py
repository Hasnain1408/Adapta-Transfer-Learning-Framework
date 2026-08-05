"""Custom title bar drawn in-app for the frameless main window.

The native OS decoration can't be restyled from Qt (its look is whatever the
window manager's theme dictates), so the window is made frameless and this
widget stands in for it: app icon, title, and minimize/maximize/close
buttons, styled to match ``style.STYLE_SHEET`` instead of the OS theme.

Dragging and the maximize button use ``QWindow.startSystemMove()`` /
``startSystemResize()`` (Qt 5.15+) so the platform's own window manager
performs the move/resize — this keeps behaviour correct under both X11 and
Wayland instead of hand-rolling geometry math.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QMouseEvent
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

_ASSETS = Path(__file__).parent / "assets"


class TitleBar(QWidget):
    """Custom-drawn replacement for the native window title bar."""

    def __init__(self, window: QWidget, title: str) -> None:
        super().__init__(window)
        self._window = window
        self.setObjectName("titleBar")
        self.setFixedHeight(38)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(8)

        icon_label = QLabel()
        icon_label.setPixmap(QIcon(str(_ASSETS / "app_icon.svg")).pixmap(QSize(18, 18)))
        layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setObjectName("titleBarLabel")
        layout.addWidget(title_label)

        layout.addStretch(1)

        self._minimize_button = self._make_button("titleBarMinButton", "−", self._window.showMinimized)
        self._maximize_button = self._make_button("titleBarMaxButton", "□", self._toggle_maximize)
        self._close_button = self._make_button("titleBarCloseButton", "✕", self._window.close)

        for button in (self._minimize_button, self._maximize_button, self._close_button):
            layout.addWidget(button)

    def _make_button(self, object_name: str, glyph: str, slot) -> QPushButton:
        button = QPushButton(glyph)
        button.setObjectName(object_name)
        button.setFixedSize(44, 38)
        button.setFocusPolicy(Qt.NoFocus)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button.clicked.connect(slot)
        return button

    def _toggle_maximize(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def refresh_maximize_glyph(self) -> None:
        self._maximize_button.setText("❐" if self._window.isMaximized() else "□")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            handle = self._window.windowHandle()
            if handle is not None:
                handle.startSystemMove()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._toggle_maximize()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
