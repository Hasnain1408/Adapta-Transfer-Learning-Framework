"""Auth, database dialogs, and experiment-history persistence for ``MainWindow``.

Everything under this heading either talks to ``ExperimentCatalogUseCase`` /
``AuthenticationUseCase`` or manages the signed-in account and history state
that gates them. Kept apart from run orchestration (``main_window_run.py``)
and result rendering (``main_window_results.py``) because those work whether
or not a database is connected — this one doesn't.

``MainWindowSessionMixin`` is a facade composing four narrower mixins, so
``main_window.py`` still names a single session concern while each piece
stays separately readable:

- ``MainWindowAuthMixin`` — sign-in/out and auth-dependent chrome.
- ``MainWindowHistoryMixin`` — persisting runs, the History tab, config replay.
- ``MainWindowDatabaseMixin`` — the Database menu's dialogs.
- ``MainWindowDetailTabsMixin`` — closable experiment/pair detail tabs.
"""

from __future__ import annotations

from presentation.qt.main_window_auth import MainWindowAuthMixin
from presentation.qt.main_window_database import MainWindowDatabaseMixin
from presentation.qt.main_window_detail_tabs import MainWindowDetailTabsMixin
from presentation.qt.main_window_history import MainWindowHistoryMixin


class MainWindowSessionMixin(
    MainWindowAuthMixin,
    MainWindowHistoryMixin,
    MainWindowDatabaseMixin,
    MainWindowDetailTabsMixin,
):
    """Auth, database dialogs, and history persistence for ``MainWindow``."""


__all__ = [
    "MainWindowAuthMixin",
    "MainWindowDatabaseMixin",
    "MainWindowDetailTabsMixin",
    "MainWindowHistoryMixin",
    "MainWindowSessionMixin",
]
