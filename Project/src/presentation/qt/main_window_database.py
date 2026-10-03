"""The Database menu: user management, dataset registry, connection info."""

from __future__ import annotations

from presentation.qt.database_dialogs import DatasetRegistryDialog, UserManagerDialog
from presentation.qt import message_box
from presentation.qt.style import build_style_sheet


class MainWindowDatabaseMixin:
    """Admin dialogs and the connection-status report."""

    def _open_user_manager(self) -> None:
        if self._reject_if_busy("opening User Management"):
            return
        if self._catalog is None:
            return
        dialog = UserManagerDialog(self._catalog, build_style_sheet(self._palette), self)
        dialog.exec_()
        # Names and roles may have changed, including the signed-in user's own.
        if self._current_user is not None and self._current_user.id is not None:
            refreshed = self._catalog.get_user(self._current_user.id)
            if refreshed is not None:
                self._current_user = refreshed
                self._update_user_label()
        self._history_tab.refresh()
        self._comparison_tab.refresh()

    def _open_dataset_registry(self) -> None:
        if self._reject_if_busy("opening the Dataset Registry"):
            return
        if self._catalog is None:
            return
        dialog = DatasetRegistryDialog(
            self._catalog, build_style_sheet(self._palette), self
        )
        dialog.exec_()

    def _show_database_info(self) -> None:
        if self._reject_if_busy("checking database info"):
            return
        if self._catalog is None:
            message_box.error(
                self,
                "Database Unavailable",
                "Not connected to the experiment database.",
                "Experiments still run normally — they just aren't saved to "
                "History, and logging in is unavailable.",
                technical=self._database_error
                or "No connection details were configured (DATABASE_URL is unset).",
            )
            return

        try:
            stats = self._catalog.statistics()
        except Exception as exc:  # noqa: BLE001
            message_box.error(
                self,
                "Database Error",
                "Connected, but couldn't read the record counts.",
                "The connection may have dropped just now. Try again in a moment.",
                technical=str(exc),
            )
            return

        message_box.success(
            self,
            "Database Connected",
            "PostgreSQL is connected and responding.",
            f"Stored records — {stats['users']} users, {stats['datasets']} datasets, "
            f"{stats['experiments']} experiments, {stats['evaluations']} evaluations.",
        )


__all__ = ["MainWindowDatabaseMixin"]
