"""``MainWindow`` composition root.

The class itself only owns window-level lifecycle (``__init__``,
``changeEvent``, ``closeEvent``) and shared state (``self._palette``,
``self._last_result``, …). Everything else is implemented by the mixins
below, one per concern:

- ``MainWindowUIMixin`` — widget/layout/menu construction, signal wiring.
- ``MainWindowSessionMixin`` — auth, database dialogs, history persistence.
- ``MainWindowThemeMixin`` — dark/light theme, Learning Materials dialog.
- ``MainWindowRunMixin`` — dataset/batch picking, worker lifecycle, run state.
- ``MainWindowResultsMixin`` — metrics/predictions/features rendering.
- ``MainWindowReportMixin`` — PDF report export.

Read this file first to see the whole window at a glance, then follow into
whichever mixin covers the behaviour you're looking for.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow

from application.authentication_use_case import AuthenticationUseCase
from application.experiment_catalog_use_case import ExperimentCatalogUseCase
from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.entities import User
from domain.models import (
    BatchTransferLearningRequest,
    BatchTransferLearningResult,
    TransferLearningRequest,
    TransferLearningResult,
)
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
        """*catalog* is ``None`` when the database could not be reached.

        The app deliberately stays usable in that state — experiments still
        run, they just are not recorded — so a Neon outage or a missing
        ``DATABASE_URL`` degrades the feature rather than blocking start-up.
        ``database_error`` carries the reason, shown once in the status bar.

        *current_user* is the account currently signed in when the window is
        created. It is usually ``None`` because sign-in is now optional and is
        only prompted from the header button when the user asks for it.
        """
        super().__init__()
        self._use_case = use_case
        self._catalog = catalog
        self._auth = auth
        self._database_error = database_error
        self._current_user = current_user
        self._worker: TransferLearningWorker | BatchTransferLearningWorker | None = None
        self._last_result: TransferLearningResult | None = None
        self._last_request: TransferLearningRequest | None = None
        self._last_batch_request: BatchTransferLearningRequest | None = None
        self._last_batch_result: BatchTransferLearningResult | None = None
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

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            self._title_bar.refresh_maximize_glyph()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(3000)
        super().closeEvent(event)


__all__ = ["MainWindow"]
