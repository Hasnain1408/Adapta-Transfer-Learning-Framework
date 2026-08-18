"""Small reusable widgets the main window composes.

Plain ``QWidget`` subclasses with no knowledge of ``MainWindow`` — they
take what they need through their constructor, so they can be read (and
reused) without the window around them.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLineEdit, QMainWindow, QVBoxLayout, QWidget

from presentation.qt.title_bar import TitleBar

class ClickableLineEdit(QLineEdit):
    """Read-only line edit that emits ``clicked`` on any mouse press.

    Used for the dataset path fields so clicking anywhere in the field opens
    the file browser dialog, not just the adjacent "Browse…" button.
    """

    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.clicked.emit()
        super().mousePressEvent(event)


class FullScreenViewer(QMainWindow):
    """Generic standalone window that borrows the results tab widget.

    Reparents the actual tab widget in place — it keeps its live state
    (chart figure, populated table rows, filter checkboxes) and is handed
    back to its original layout slot when the window closes, so no data has
    to be duplicated or re-synced.
    """

    def __init__(
        self,
        title: str,
        content: QWidget,
        stylesheet: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{title} — Full Screen")
        self.setMinimumSize(800, 540)
        self.setStyleSheet(stylesheet)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        title_bar = TitleBar(self, f"{title} — Full Screen")

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(18, 14, 18, 14)
        body_layout.setSpacing(10)
        body_layout.addWidget(content, 1)

        container = QWidget()
        container.setObjectName("centralWidget")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(title_bar)
        outer.addWidget(body, 1)
        self.setCentralWidget(container)

        self.showMaximized()


__all__ = ["ClickableLineEdit", "FullScreenViewer"]
