"""The menu bar and the custom title bar above it."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAction, QVBoxLayout, QWidget

from domain.entities import UserRole
from presentation.qt.title_bar import TitleBar

class MainWindowMenuMixin:
    """The menu bar and the custom title bar above it."""

    def _build_menu(self) -> None:
        self._home_action = QAction("Go to &Home Page", self)
        self._home_action.setShortcut(QKeySequence("Ctrl+H"))
        self._open_source_action = QAction("Open &Source Dataset…", self)
        self._open_source_action.setShortcut(QKeySequence("Ctrl+S"))
        self._open_target_action = QAction("Open &Target Dataset…", self)
        self._open_target_action.setShortcut(QKeySequence("Ctrl+T"))
        self._swap_datasets_action = QAction("S&wap Source / Target", self)
        self._swap_datasets_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._export_action = QAction("&Export Report…", self)
        self._export_action.setShortcut(QKeySequence("Ctrl+R"))
        self._export_action.setEnabled(False)
        self._quit_action = QAction("E&xit System", self)
        self._quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self._run_action = QAction("&Run Experiment", self)
        self._run_action.setShortcut(QKeySequence("Ctrl+E"))
        self._about_action = QAction("&About", self)

        self._learning_materials_action = QAction("&Learning Materials…", self)
        self._learning_materials_action.setShortcut(QKeySequence("Ctrl+L"))
        self._tl_methods_action = QAction("Transfer Learning &Methods…", self)
        self._classifiers_action = QAction("&Classifiers…", self)
        self._metrics_action = QAction("Evaluation &Metrics…", self)
        self._preprocessing_action = QAction("&Normalization & Scaling…", self)

        # Database actions stay disabled without a connection rather than
        # opening a dialog that can only show an error.
        has_database = self._catalog is not None
        is_admin = self._current_user is not None and self._current_user.role == UserRole.ADMIN
        self._manage_users_action = QAction("Manage &Users…", self)
        self._manage_users_action.setShortcut(QKeySequence("Ctrl+U"))
        self._manage_users_action.setEnabled(has_database and is_admin)
        self._dataset_registry_action = QAction("&Dataset Registry…", self)
        self._dataset_registry_action.setShortcut(QKeySequence("Ctrl+D"))
        self._dataset_registry_action.setEnabled(has_database)
        self._refresh_history_action = QAction("Refresh &History", self)
        self._refresh_history_action.setShortcut(QKeySequence("F5"))
        self._refresh_history_action.setEnabled(has_database)
        self._database_info_action = QAction("Connection &Info", self)

        self.menuBar().setCornerWidget(self._home_button, Qt.TopLeftCorner)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._home_action)
        file_menu.addSeparator()
        file_menu.addAction(self._open_source_action)
        file_menu.addAction(self._open_target_action)
        file_menu.addAction(self._swap_datasets_action)
        file_menu.addSeparator()
        file_menu.addAction(self._export_action)
        file_menu.addSeparator()
        file_menu.addAction(self._quit_action)

        run_menu = self.menuBar().addMenu("&Run")
        run_menu.addAction(self._run_action)

        database_menu = self.menuBar().addMenu("&Database")
        database_menu.addAction(self._manage_users_action)
        database_menu.addAction(self._dataset_registry_action)
        database_menu.addSeparator()
        database_menu.addAction(self._refresh_history_action)
        database_menu.addAction(self._database_info_action)

        learning_menu = self.menuBar().addMenu("&Learning Materials")
        learning_menu.addAction(self._learning_materials_action)
        learning_menu.addSeparator()
        learning_menu.addAction(self._tl_methods_action)
        learning_menu.addAction(self._classifiers_action)
        learning_menu.addAction(self._metrics_action)
        learning_menu.addAction(self._preprocessing_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self._learning_materials_action)
        help_menu.addSeparator()
        help_menu.addAction(self._about_action)

    def _build_title_bar(self) -> None:
        """Stack a custom title bar above the real menu bar.

        The window is frameless, so ``TitleBar`` stands in for the OS
        decoration. ``QMainWindow.setMenuWidget`` accepts any widget as the
        menu-bar area, so the existing ``self.menuBar()`` is nested inside a
        small container alongside it rather than rebuilt.
        """
        self._title_bar = TitleBar(self, "Adapta: Transfer Learning Studio")

        # Held onto deliberately: once setMenuWidget() below installs the
        # container, a later self.menuBar() call would *create a second,
        # empty* menu bar and install that instead — destroying this
        # container, the title bar inside it, and the running process with
        # it. Anything that needs the menu bar afterwards uses this.
        self._menu_bar = self.menuBar()

        container = QWidget()
        container.setObjectName("titleBarContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._title_bar)
        layout.addWidget(self._menu_bar)
        self.setMenuWidget(container)


__all__ = ["MainWindowMenuMixin"]
