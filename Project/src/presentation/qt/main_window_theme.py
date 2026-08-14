"""Theme switching and the Learning Materials dialog for ``MainWindow``."""

from __future__ import annotations

from PyQt5.QtWidgets import QApplication

from presentation.qt.style import DARK_PALETTE, LIGHT_PALETTE, build_style_sheet


class MainWindowThemeMixin:
    """Dark/light theme toggling and the Learning Materials dialog."""

    def _apply_theme_stylesheet(self) -> None:
        stylesheet = build_style_sheet(self._palette)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet)
        self.setStyleSheet(stylesheet)

    def _toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self._palette = LIGHT_PALETTE if self._theme == "light" else DARK_PALETTE
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._apply_theme_stylesheet()
        self._theme_toggle_button.setText("🌙  Dark" if self._theme == "light" else "☀  Light")

        color = self._palette[self._status_key]
        self._status_dot.setStyleSheet(f"color: {color};")
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

        self._charts_widget.set_palette(self._palette)
        self._visualization_tab.set_palette(self._palette)
        self._history_tab.set_palette(self._palette)
        if self._last_result is not None:
            self._show_predictions(self._last_result)

        if hasattr(self, "_learning_materials_dialog") and self._learning_materials_dialog is not None:
            try:
                self._learning_materials_dialog.set_palette(self._palette)
            except RuntimeError:
                self._learning_materials_dialog = None

    def _open_learning_materials(self, category_index: int = 0) -> None:
        dialog = getattr(self, "_learning_materials_dialog", None)
        try:
            if dialog is not None:
                dialog.select_topic(category_index)
                dialog.raise_()
                dialog.activateWindow()
                return
        except RuntimeError:
            pass

        from presentation.qt.learning_materials_dialog import LearningMaterialsDialog

        self._learning_materials_dialog = LearningMaterialsDialog(
            self._palette, parent=self
        )
        self._learning_materials_dialog.select_topic(category_index)
        self._learning_materials_dialog.destroyed.connect(
            lambda: setattr(self, "_learning_materials_dialog", None)
        )
        self._learning_materials_dialog.show()


__all__ = ["MainWindowThemeMixin"]
