"""The login and registration pages of ``AuthDialog``.

Two pages of one ``QStackedWidget`` rather than two dialogs, so switching
between them keeps the window in place. Split from the dialog itself so
the form layouts stay readable next to the submit logic that reads them.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from domain.entities import UserRole


def field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


class AuthPagesMixin:
    """Builds the two stacked pages and the widgets they own."""

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
        form.addRow(field_label("Email"), self._login_email)
        form.addRow(field_label("Password"), self._login_password)

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
        form.addRow(field_label("Name"), self._register_name)
        form.addRow(field_label("Email"), self._register_email)
        form.addRow(field_label("Password"), self._register_password)
        form.addRow(field_label("Confirm Password"), self._register_confirm)
        form.addRow(field_label("Role"), self._register_role)

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


__all__ = ["AuthPagesMixin", "field_label"]
