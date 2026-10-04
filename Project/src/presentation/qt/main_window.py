"""MainWindow composition root.

The class itself only owns window-level lifecycle (__init__,
changeEvent, closeEvent) and shared state (self._palette,
self._last_result, …). Everything else is implemented by the mixins
below, one per concern:

- MainWindowUIMixin — widget/layout/menu construction, signal wiring.
- MainWindowSessionMixin — auth, database dialogs, history persistence.
- MainWindowThemeMixin — dark/light theme, Learning Materials dialog.
- MainWindowRunMixin — dataset/batch picking, worker lifecycle, run state.
- MainWindowResultsMixin — metrics/predictions/features rendering.
- MainWindowReportMixin — PDF report export.

Read this file first to see the whole window at a glance, then follow into
whichever mixin covers the behaviour you're looking for.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow

from application.authentication_use_case import AuthenticationUseCase
from application.experiment_catalog_use_case import ExperimentCatalogUseCase
from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.entities import User
from domain.models import (
    BatchTransferLearningRequest,
    BatchTransferLearningResult,
    MultiMethodTransferLearningRequest,
    MultiMethodTransferLearningResult,
    TransferLearningRequest,
    TransferLearningResult,
)
from domain.run_control import RunControl
from presentation.qt import message_box
from presentation.qt.experiment_detail_view import ExperimentDetailView
from presentation.qt.main_window_report import MainWindowReportMixin
from presentation.qt.main_window_results import MainWindowResultsMixin
from presentation.qt.main_window_run import MainWindowRunMixin
from presentation.qt.main_window_session import MainWindowSessionMixin
from presentation.qt.main_window_theme import MainWindowThemeMixin
from presentation.qt.main_window_ui import MainWindowUIMixin
from presentation.qt.pair_detail_view import PairDetailView
from presentation.qt.style import DARK_PALETTE
from presentation.qt.worker import BatchTransferLearningWorker, TransferLearningWorker


class MainWindow(
    MainWindowUIMixin,
    MainWindowSessionMixin,
    MainWindowThemeMixin,
    MainWindowRunMixin,
    MainWindowResultsMixin,
    MainWindowReportMixin,
    QMainWindow,
):
    def __init__(
        self,
        use_case: RunTransferLearningUseCase,
        catalog: ExperimentCatalogUseCase | None = None,
        auth: AuthenticationUseCase | None = None,
        current_user: User | None = None,
        database_error: str | None = None,
    ) -> None:
        """*catalog* is None when the database could not be reached.

        The app deliberately stays usable in that state — experiments still
        run, they just are not recorded — so a Neon outage or a missing
        DATABASE_URL degrades the feature rather than blocking start-up.
        database_error carries the reason, shown once in the status bar.

        *current_user* is the account currently signed in when the window is
        created. It is usually None because sign-in is now optional and is
        only prompted from the header button when the user asks for it.
        """
        super().__init__()
        self._use_case = use_case
        self._catalog = catalog
        self._auth = auth
        self._database_error = database_error
        self._current_user = current_user
        self._worker: TransferLearningWorker | BatchTransferLearningWorker | None = None
        self._run_control: RunControl | None = None
        self._progress_color_index = 0
        self._last_result: TransferLearningResult | None = None
        self._last_request: TransferLearningRequest | None = None
        self._last_batch_request: BatchTransferLearningRequest | None = None
        self._last_batch_result: BatchTransferLearningResult | None = None
        self._last_multi_request: MultiMethodTransferLearningRequest | None = None
        self._last_multi_result: MultiMethodTransferLearningResult | None = None
        #: "single" | "batch" | "multi" | None — which of the _last_*_result
        #: fields the sidebar's Export Report should build a report from.
        self._last_run_kind: str | None = None
        self._method_results_tab_index = -1
        self._batch_results_tab_index = -1
        self._theme = "dark"
        self._palette = DARK_PALETTE
        self._status_key = "neutral"
        self._history_tab_index = -1
        #: experiment id -> its open detail tab widget, so a second click on
        #: the same row raises the existing tab instead of duplicating it.
        self._experiment_detail_tabs: dict[int, ExperimentDetailView] = {}
        #: id(pair) -> its open detail tab widget, same idea as above but for
        #: batch pairs, which have no persisted id to key on.
        self._pair_detail_tabs: dict[int, PairDetailView] = {}
        #: id(method entry) -> its open detail tab, for one method of a
        #: comparison. Keyed on the entry rather than the method name so two
        #: comparisons in a row cannot collide on "TCA".
        self._method_detail_tabs: dict[int, PairDetailView] = {}

        self.setWindowTitle("Adapta: Transfer Learning Studio")
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "app_icon.svg")))
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setMinimumSize(1000, 720)
        self.resize(1180, 820)
        self._apply_theme_stylesheet()

        self._build_widgets()
        self._build_menu()
        self._build_title_bar()
        self._build_layout()
        self._build_status_bar()
        self._connect_signals()
        self._sync_method_parameters()
        self._update_auth_ui()
        # The app opens on the landing page; the workspace is one click away
        # and is built and ready behind it.
        self._show_landing()

    # ------------------------------------------------------------------
    # Landing page navigation
    # ------------------------------------------------------------------
    def _show_landing(self) -> None:
        """Bring the landing page to the front.

        Safe at any time, including mid-run: the workspace page is only
        stacked behind, not torn down, and the status bar carrying the
        experiment's progress belongs to the window rather than either page.
        """
        self._update_landing_session()
        self._page_stack.setCurrentIndex(0)
        # A full application menu over a landing page reads like a leftover
        # from the screen behind it. Menu-action shortcuts keep working while
        # it is hidden (they are scoped to the window, not to the menu bar's
        # visibility), so Ctrl+H still comes back here.
        self._menu_bar.setVisible(False)

    def _show_workspace(self) -> None:
        self._page_stack.setCurrentIndex(1)
        self._menu_bar.setVisible(True)

    def _update_landing_session(self) -> None:
        """Push database/auth state onto the landing page's cards."""
        landing = getattr(self, "_landing_page", None)
        if landing is None:
            return
        landing.set_session(
            has_database=self._catalog is not None,
            signed_in=self._current_user is not None,
            user_label=self._user_label.text(),
        )
        landing.set_theme_label(self._theme_toggle_button.text())

    def _navigate_from_landing(self, key: str) -> None:
        """Route one landing-page choice to the thing it names.

        The whole mapping lives here so LandingPage stays a view that
        emits names and knows nothing about the window's internals.
        """
        if key == "experiment":
            self._show_workspace()
            if self._batch_mode_check.isEnabled():
                self._batch_mode_check.setChecked(False)
        elif key == "batch":
            self._show_workspace()
            if self._batch_mode_check.isEnabled():
                self._batch_mode_check.setChecked(True)
        elif key == "comparison":
            self._open_workspace_tab(self._comparison_tab)
        elif key == "history":
            self._open_history_from_landing()
        elif key == "registry":
            # Dialogs open over whichever page is showing, so the user comes
            # back to the landing page rather than being dropped into the
            # workspace by a side effect of closing a dialog. The database
            # check is here because _open_dataset_registry returns silently
            # without one, and a card that swallows the click explains
            # nothing.
            if self._catalog is None:
                self._explain_database_outage()
            else:
                self._open_dataset_registry()
        elif key == "learning":
            self._open_learning_materials(0)
        elif key == "auth":
            self._handle_auth_button_clicked()
            self._update_landing_session()
        elif key == "theme":
            self._toggle_theme()
        elif key == "exit":
            self._exit_application()

    def _explain_database_outage(self) -> None:
        """Say why a database-backed module is unreachable."""
        message_box.info(
            self,
            "Database Unavailable",
            "This module reads from the experiment database.",
            "Adapta could not connect, so there is nothing to show here yet. "
            "Experiments still run — they just aren't recorded.",
        )

    def _open_workspace_tab(self, tab) -> None:
        """Switch to the workspace with *tab* selected in the results area."""
        if self._catalog is None:
            self._explain_database_outage()
            return
        self._show_workspace()
        # The results area starts on its empty-state page; these tabs are
        # populated from the database, not from a run, so they are worth
        # showing before any experiment has finished.
        self._results_stack.setCurrentIndex(1)
        index = self._results_tabs.indexOf(tab)
        if index >= 0:
            self._results_tabs.setCurrentIndex(index)

    def _open_history_from_landing(self) -> None:
        """History needs both a database and a signed-in user."""
        if self._catalog is None:
            self._open_workspace_tab(self._history_tab)  # explains the outage
            return
        if self._current_user is None:
            self._show_login_dialog()
            self._update_landing_session()
            if self._current_user is None:
                return
        self._show_workspace()
        self._show_history()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            self._title_bar.refresh_maximize_glyph()

    def _stop_background_work(self) -> None:
        """Bring any in-flight run to a halt before the window goes away.

        Shared by closeEvent and _exit_application, which reach
        shutdown by different routes — the window manager's close button
        versus the app's own Exit System button — but must leave the same
        state behind: no QThread still touching objects Qt is about to
        destroy.
        """
        worker = self._worker
        try:
            still_running = worker is not None and worker.isRunning()
        except RuntimeError:
            # The QThread's C++ side is already gone (deleteLater ran);
            # nothing left to stop.
            return
        if still_running:
            # A paused run is blocked indefinitely on its checkpoint, so it
            # has to be cancelled before waiting on it, or this would hang
            # until the timeout with the thread still alive afterwards.
            if self._run_control is not None:
                self._run_control.cancel()
            worker.wait(3000)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._stop_background_work()
        super().closeEvent(event)

    def _exit_application(self) -> None:
        """Quit the whole app: the sidebar's Exit System button and Ctrl+Q.

        Not simply self.close(). Closing only this window leaves the app
        running whenever another top-level window is open — the full-screen
        results viewer and the Learning Materials dialog are both parented to
        None — so the process would linger with no way back to it. This
        ends the event loop outright, which lets main() run its
        finally and close the database connection.
        """
        running = self._is_worker_busy()
        if running:
            headline = f"{self._busy_description()} is still running."
            detail = (
                "Exiting now stops it partway and discards its results. "
                "Anything already saved to history is unaffected."
            )
        else:
            headline = "Exit Adapta?"
            detail = (
                "Results on screen will be discarded. Experiments already "
                "recorded in history are kept."
            )
        if not message_box.confirm(
            self,
            "Exit System",
            headline,
            detail,
            confirm_label="Exit",
            destructive=running,
        ):
            return

        self._stop_background_work()
        application = QApplication.instance()
        if application is not None:
            application.quit()
        else:  # pragma: no cover - only reachable outside a running app
            self.close()


__all__ = ["MainWindow"]
