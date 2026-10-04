"""Dataset selection and run orchestration for MainWindow.

MainWindowRunMixin is a facade composing five narrower mixins, so
main_window.py still names a single run concern while each piece stays
separately readable:

- MainWindowBusyMixin — the in-flight guard the other mixins check.
- MainWindowDatasetsMixin — picking source/target/combined/batch files.
- MainWindowProgressMixin — status bar, progress bar, pause/cancel.
- MainWindowExecutionMixin — starting runs, completion handlers.
- MainWindowWorkspaceMixin — clearing the workspace back to empty.

The worker-thread invariant lives in main_window_execution.py: every
experiment runs off the UI thread through TransferLearningWorker /
BatchTransferLearningWorker, never by calling the use case from a slot.
"""

from __future__ import annotations

from presentation.qt.main_window_busy import MainWindowBusyMixin
from presentation.qt.main_window_datasets import MainWindowDatasetsMixin
from presentation.qt.main_window_execution import MainWindowExecutionMixin
from presentation.qt.main_window_progress import MainWindowProgressMixin
from presentation.qt.main_window_workspace import MainWindowWorkspaceMixin


class MainWindowRunMixin(
    MainWindowBusyMixin,
    MainWindowDatasetsMixin,
    MainWindowProgressMixin,
    MainWindowExecutionMixin,
    MainWindowWorkspaceMixin,
):
    """Dataset picking, batch-file management, and run orchestration."""


__all__ = [
    "MainWindowBusyMixin",
    "MainWindowDatasetsMixin",
    "MainWindowExecutionMixin",
    "MainWindowProgressMixin",
    "MainWindowRunMixin",
    "MainWindowWorkspaceMixin",
]
