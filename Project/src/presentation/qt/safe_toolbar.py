"""A NavigationToolbar2QT subclass that overrides the save dialog.

The default toolbar calls matplotlib's own save_figure(), which on Linux/Qt5Agg
opens the *native* OS file-picker.  That dialog renders as a completely black
(blank) window on many GTK-on-Wayland or theme-heavy setups, and can crash the
process.

SafeNavigationToolbar replaces save_figure() with a pure-Qt QFileDialog that
uses DontUseNativeDialog — the same option used everywhere else in this
codebase — so the user sees a properly-themed, working file chooser.
"""

from __future__ import annotations

import os

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


_SAVE_FILTER = (
    "PNG image (*.png);;"
    "JPEG image (*.jpg *.jpeg);;"
    "SVG vector (*.svg);;"
    "PDF document (*.pdf);;"
    "All files (*)"
)

_EXT_FORMAT = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".svg": "svg",
    ".pdf": "pdf",
}


class SafeNavigationToolbar(NavigationToolbar2QT):
    """NavigationToolbar with a non-native, always-visible save dialog."""

    def save_figure(self, *_args) -> None:  # type: ignore[override]
        """Open a Qt-native (non-OS-native) file dialog to save the figure."""
        # Suggest the last directory the user visited, or the home folder.
        start_dir = getattr(self, "_last_save_dir", os.path.expanduser("~"))

        dialog = QFileDialog(self, "Save Chart", start_dir, _SAVE_FILTER)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("png")

        if dialog.exec_() != QFileDialog.Accepted:
            return

        paths = dialog.selectedFiles()
        if not paths:
            return

        path = paths[0]
        # Remember the directory for next time.
        self._last_save_dir = os.path.dirname(path)

        # Derive the format from the extension; fall back to PNG.
        ext = os.path.splitext(path)[1].lower()
        fmt = _EXT_FORMAT.get(ext, "png")
        if not ext:
            path += ".png"

        try:
            self.canvas.figure.savefig(path, format=fmt, bbox_inches="tight", dpi=150)
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Could not save the chart:\n{exc}",
            )
