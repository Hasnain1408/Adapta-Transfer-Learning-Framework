"""Login and registration, shown on demand from the main window.

A single modal with two pages (login / register) toggled by a link-style
button, rather than two separate dialogs, so switching between them keeps the
window in place instead of popping a new one on top.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from application.authentication_use_case import AuthenticationError, AuthenticationUseCase
from domain.entities import User, UserRole
from presentation.qt import message_box
from presentation.qt.title_bar import install_title_bar


def _field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


class AuthDialog(QDialog):
    """Blocks until the user logs in, registers, or cancels.

    :meth:`authenticated_user` returns the resulting :class:`User` once
    :meth:`exec_` returns ``QDialog.Accepted``; it is ``None`` otherwise
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

    # ------------------------------------------------------------------
    def _build_login_page(self) -> None:
        page = QWidget()
        self._login_email = QLineEdit()
        self._login_email.setPlaceholderText("name@example.com")
        self._login_password = QLineEdit()
        self._login_password.setEchoMode(QLineEdit.Password)
        self._login_password.setPlaceholderText("Password")
        self._login_password.returnPressed.connect(self._do_login)

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow(_field_label("Email"), self._login_email)
        form.addRow(_field_label("Password"), self._login_password)

        login_button = QPushButton("Log In")
        login_button.setObjectName("runButton")
        login_button.setCursor(Qt.PointingHandCursor)
        login_button.setDefault(True)
        login_button.clicked.connect(self._do_login)

        switch_button = QPushButton("New here? Create an account")
        switch_button.setObjectName("linkButton")
        switch_button.setCursor(Qt.PointingHandCursor)
        switch_button.clicked.connect(lambda: self._stack.setCurrentIndex(1))

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(14)
        layout.addLayout(form)
        layout.addWidget(login_button)
        layout.addWidget(switch_button, 0, Qt.AlignHCenter)
        self._login_page = page

    def _build_register_page(self) -> None:
        page = QWidget()
        self._register_name = QLineEdit()
        self._register_name.setPlaceholderText("Full name")
        self._register_email = QLineEdit()
        self._register_email.setPlaceholderText("name@example.com")
        self._register_password = QLineEdit()
        self._register_password.setEchoMode(QLineEdit.Password)
        self._register_password.setPlaceholderText("At least 8 characters")
        self._register_confirm = QLineEdit()
        self._register_confirm.setEchoMode(QLineEdit.Password)
        self._register_confirm.setPlaceholderText("Confirm password")
        self._register_confirm.returnPressed.connect(self._do_register)

        self._register_role = QComboBox()
        for role in UserRole:
            if role is UserRole.ADMIN:
                # The first admin account is provisioned separately (Manage
                # Users, or directly in the database) rather than self-served,
                # so registration alone can never mint one.
                continue
            self._register_role.addItem(role.label, role.value)

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow(_field_label("Name"), self._register_name)
        form.addRow(_field_label("Email"), self._register_email)
        form.addRow(_field_label("Password"), self._register_password)
        form.addRow(_field_label("Confirm Password"), self._register_confirm)
        form.addRow(_field_label("Role"), self._register_role)

        register_button = QPushButton("Create Account")
        register_button.setObjectName("runButton")
        register_button.setCursor(Qt.PointingHandCursor)
        register_button.clicked.connect(self._do_register)

        switch_button = QPushButton("Already have an account? Log in")
        switch_button.setObjectName("linkButton")
        switch_button.setCursor(Qt.PointingHandCursor)
        switch_button.clicked.connect(lambda: self._stack.setCurrentIndex(0))

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(14)
        layout.addLayout(form)
        layout.addWidget(register_button)
        layout.addWidget(switch_button, 0, Qt.AlignHCenter)
        self._register_page = page

    # ------------------------------------------------------------------
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
