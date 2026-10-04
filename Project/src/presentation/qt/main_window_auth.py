"""Sign-in/sign-out and the auth-dependent state of the window chrome.

Everything here is about *who* is using the app. What that account is then
allowed to see (History) or record (persistence) lives in
main_window_history.py.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QDialog

from domain.entities import UserRole
from presentation.qt.auth_dialogs import AuthDialog
from presentation.qt import message_box
from presentation.qt.style import build_style_sheet


class MainWindowAuthMixin:
    """Login/logout and the widgets whose enabled state depends on it."""

    def _update_user_label(self) -> None:
        if self._current_user is not None:
            self._user_label.setText(f"Signed in as {self._current_user.display_name}")
        elif self._catalog is None:
            self._user_label.setText("No database — experiments are not recorded")
        else:
            self._user_label.setText("Signed out")

    def _update_auth_ui(self) -> None:
        signed_in = self._current_user is not None
        can_authenticate = self._auth is not None

        if signed_in:
            self._logout_button.setText("Log Out")
            self._logout_button.setToolTip("Sign out and hide experiment history")
        else:
            self._logout_button.setText("Log In")
            self._logout_button.setToolTip("Log in to record and review experiments")

        self._logout_button.setEnabled(can_authenticate)
        self._manage_users_action.setEnabled(
            self._catalog is not None and signed_in and self._current_user is not None and self._current_user.role == UserRole.ADMIN
        )
        self._refresh_history_action.setEnabled(self._catalog is not None and signed_in)
        if hasattr(self, "_browse_history_button"):
            self._browse_history_button.setVisible(self._catalog is not None and signed_in)

        self._history_tab.set_current_user(self._current_user)
        self._comparison_tab.set_current_user(self._current_user)
        self._set_history_visible(signed_in)
        self._update_landing_session()

    def _handle_auth_button_clicked(self) -> None:
        # A run records itself under whoever is signed in when it *finishes*
        # (see _persist_run), not who started it — switching accounts
        # mid-run would credit it to the wrong person, so both directions
        # wait like every other database-touching action.
        if self._reject_if_busy("signing in or out"):
            return
        if self._current_user is None:
            self._show_login_dialog()
        else:
            self._sign_out()

    def _show_login_dialog(self) -> None:
        if self._auth is None:
            message_box.info(
                self,
                "Log In Unavailable",
                "Signing in needs a database connection.",
                "You can still run experiments — they just won't be saved to "
                "History until a connection is available.",
            )
            return

        dialog = AuthDialog(self._auth, build_style_sheet(self._palette), parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        self._current_user = dialog.authenticated_user()
        self._update_user_label()
        self._update_context_label()
        self._update_auth_ui()
        if self._current_user is not None:
            self._set_status(f"Signed in as {self._current_user.display_name}", "success")

    def _sign_out(self) -> None:
        self._current_user = None
        self._update_user_label()
        self._update_context_label()
        self._update_auth_ui()
        self._set_status("You have been logged out.", "neutral")
        message_box.success(
            self,
            "Logged Out",
            "You're signed out.",
            "You can keep running experiments. History and saved results stay "
            "hidden until you log in again.",
        )

    def _current_user_id(self) -> int | None:
        return self._current_user.id if self._current_user else None


__all__ = ["MainWindowAuthMixin"]
