"""Login and registration, shown on demand from the main window.

A single modal with two pages (login / register) toggled by a link-style
button, rather than two separate dialogs, so switching between them keeps the
window in place instead of popping a new one on top.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from application.authentication_use_case import AuthenticationError, AuthenticationUseCase
from domain.entities import User
from presentation.qt import message_box
from presentation.qt.auth_pages import AuthPagesMixin
from presentation.qt.title_bar import install_title_bar


class AuthDialog(AuthPagesMixin, QDialog):
    """Blocks until the user logs in, registers, or cancels.

    authenticated_user returns the resulting User once
    exec_ returns QDialog.Accepted; it is None otherwise
    (including after a plain cancel).
    """

    def __init__(
        self,
        auth: AuthenticationUseCase,
        stylesheet: str,
        *,
        start_on_register: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._auth = auth
        self._user: User | None = None

        self.setWindowTitle("Sign In — Transfer Learning Studio")
        self.setStyleSheet(stylesheet)
        self.setMinimumWidth(380)

        self._build_login_page()
        self._build_register_page()

        self._stack = QStackedWidget()
        self._stack.addWidget(self._login_page)
        self._stack.addWidget(self._register_page)

        title = QLabel("Transfer Learning Studio")
        title.setObjectName("appTitle")
        subtitle = QLabel("Sign in to record and review your experiments")
        subtitle.setObjectName("appSubtitle")
        subtitle.setWordWrap(True)

        card = QWidget()
        card.setObjectName("authCard")
        card.setFixedWidth(380)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(26, 22, 26, 20)
        card_layout.setSpacing(4)
        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(14)
        card_layout.addWidget(self._stack)

        # Full screen leaves a lot of empty space around a small form, so the
        # card is centered in whatever room the window has instead of
        # sitting pinned to the top-left corner.
        body = QWidget()
        body_row = QHBoxLayout(body)
        body_row.addStretch(1)
        body_column = QVBoxLayout()
        body_column.addStretch(1)
        body_column.addWidget(card)
        body_column.addStretch(1)
        body_row.addLayout(body_column)
        body_row.addStretch(1)

        install_title_bar(self, "Sign In — Transfer Learning Studio", body)

        self._stack.setCurrentIndex(1 if start_on_register else 0)

    def _do_login(self) -> None:
        email = self._login_email.text()
        password = self._login_password.text()
        if not email.strip() or not password:
            missing = "email address" if not email.strip() else "password"
            message_box.warn(
                self,
                "Details Needed",
                f"Enter your {missing} to log in.",
            )
            return
        try:
            self._user = self._auth.login(email, password)
        except AuthenticationError as exc:
            # The use case's own wording ("Incorrect email or password") is
            # already the user-facing sentence, and it is deliberately vague
            # about *which* was wrong — restating it here would either leak
            # that or contradict it.
            message_box.warn(self, "Login Failed", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            message_box.error(
                self,
                "Login Unavailable",
                "Couldn't reach the account database.",
                "Your details weren't checked. Try again in a moment.",
                technical=str(exc),
            )
            return
        self.accept()

    def _do_register(self) -> None:
        password = self._register_password.text()
        if password != self._register_confirm.text():
            message_box.warn(
                self,
                "Passwords Don't Match",
                "The two passwords aren't the same.",
                "Re-type the same password in both fields.",
            )
            return
        try:
            self._user = self._auth.register(
                self._register_name.text(),
                self._register_email.text(),
                password,
                role=self._register_role.currentData(),
            )
        except AuthenticationError as exc:
            # Same reasoning as login: the use case raises the sentence meant
            # for the user ("An account with that email already exists").
            message_box.warn(self, "Couldn't Create Account", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            message_box.error(
                self,
                "Registration Unavailable",
                "Couldn't reach the account database.",
                "Your account wasn't created. Try again in a moment.",
                technical=str(exc),
            )
            return
        self.accept()

    # ------------------------------------------------------------------
    def authenticated_user(self) -> User | None:
        return self._user


__all__ = ["AuthDialog"]


__all__ = ["AuthDialog"]
