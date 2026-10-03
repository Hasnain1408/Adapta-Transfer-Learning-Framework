"""Dialogs for managing the persisted catalogue: users and datasets.

Both follow the same shape — a table on top, actions underneath, every write
routed through :class:`ExperimentCatalogUseCase` so no widget touches SQL.

This module is the facade; the dialogs live one per topic in
``user_dialogs.py`` and ``dataset_registry_dialog.py``.
"""

from __future__ import annotations

from presentation.qt.dataset_registry_dialog import DatasetRegistryDialog
from presentation.qt.user_dialogs import UserEditorDialog, UserManagerDialog

__all__ = ["DatasetRegistryDialog", "UserEditorDialog", "UserManagerDialog"]
