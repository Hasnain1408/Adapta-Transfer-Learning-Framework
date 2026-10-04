"""Widget, layout, and menu construction for MainWindow.

Everything under this heading only builds the window — no business logic, no
persistence, no run orchestration. MainWindowUIMixin is a facade
composing four narrower mixins, so main_window.py still names a single
construction concern while each stage stays separately readable:

- MainWindowWidgetsMixin — creating every widget the window owns.
- MainWindowMenuMixin — the menu bar and custom title bar.
- MainWindowLayoutMixin — arranging those widgets into the two pages.
- MainWindowSignalsMixin — connecting each control to its handler.

They run in that order from MainWindow.__init__: widgets must exist
before the layout can place them, and both before signals can be wired.

The option tables live in ui_options.py and the small reusable widget
classes in ui_widgets.py; both are re-exported here so existing imports
from presentation.qt.main_window_ui keep working.
"""

from __future__ import annotations

from presentation.qt.main_window_layout import MainWindowLayoutMixin
from presentation.qt.main_window_menu import MainWindowMenuMixin
from presentation.qt.main_window_signals import MainWindowSignalsMixin
from presentation.qt.main_window_widgets import MainWindowWidgetsMixin
from presentation.qt.ui_options import (
    CLASSIFIER_OPTIONS,
    DATASET_FILTER,
    FEATURE_PREVIEW_ROWS,
    KERNEL_OPTIONS,
    NORMALIZATION_OPTIONS,
    STANDARDIZATION_OPTIONS,
)
from presentation.qt.ui_widgets import ClickableLineEdit, FullScreenViewer

#: Kept as an alias: the class was private to this module before the split.
_ClickableLineEdit = ClickableLineEdit


class MainWindowUIMixin(
    MainWindowWidgetsMixin,
    MainWindowMenuMixin,
    MainWindowLayoutMixin,
    MainWindowSignalsMixin,
):
    """Widget/layout/menu construction and signal wiring for MainWindow.

    Composed into MainWindow alongside the other MainWindow*Mixin
    classes; every method below assumes the attributes MainWindow.__init__
    sets before calling into construction (self._use_case, self._catalog,
    self._palette, self._current_user, …).
    """


__all__ = [
    "CLASSIFIER_OPTIONS",
    "DATASET_FILTER",
    "FEATURE_PREVIEW_ROWS",
    "KERNEL_OPTIONS",
    "NORMALIZATION_OPTIONS",
    "STANDARDIZATION_OPTIONS",
    "ClickableLineEdit",
    "FullScreenViewer",
    "MainWindowLayoutMixin",
    "MainWindowMenuMixin",
    "MainWindowSignalsMixin",
    "MainWindowUIMixin",
    "MainWindowWidgetsMixin",
    "_ClickableLineEdit",
]
