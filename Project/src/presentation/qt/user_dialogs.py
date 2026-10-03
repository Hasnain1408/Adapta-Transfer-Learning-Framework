"""Dialogs for managing user accounts.

A table of accounts with add/edit/delete underneath, and the single-record
editor behind those actions. Every write is routed through
``ExperimentCatalogUseCase`` so no widget touches SQL.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from application.experiment_catalog_use_case import ExperimentCatalogUseCase
from domain.auth import hash_password
from domain.entities import User, UserRole
from presentation.qt import message_box
from presentation.qt.dialog_utils import format_timestamp as _format_timestamp
from presentation.qt.title_bar import install_title_bar


class UserEditorDialog(QDialog):
    """Create or edit a single user.

    Returns the edited :class:`User` from :meth:`user`; ``id`` and
    ``created_at`` are carried through untouched so the caller can pass the
    result straight to an update.
    """

    def __init__(
        self,
        stylesheet: str,
        user: User | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._original = user
        self.setWindowTitle("Edit User" if user else "New User")
        self.setStyleSheet(stylesheet)
        self.setMinimumWidth(400)

        self._name_input = QLineEdit(user.name if user else "")
        self._name_input.setPlaceholderText("Full name")

        self._email_input = QLineEdit(user.email if user else "")
        self._email_input.setPlaceholderText("name@example.com")

        self._password_input = QLineEdit()
        self._password_input.setEchoMode(QLineEdit.Password)
        self._password_input.setPlaceholderText(
            "Leave blank to keep current password" if user else "Optional — set an initial password"
        )

        self._role_combo = QComboBox()
        for role in UserRole:
            self._role_combo.addItem(role.label, role.value)
        if user is not None:
            index = self._role_combo.findData(user.role.value)
            if index >= 0:
                self._role_combo.setCurrentIndex(index)

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow(self._field_label("Name"), self._name_input)
        form.addRow(self._field_label("Email"), self._email_input)
        form.addRow(self._field_label("Password"), self._password_input)
        form.addRow(self._field_label("Role"), self._role_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(14)
        layout.addLayout(form)
        layout.addWidget(buttons)
        install_title_bar(self, "Edit User" if user else "New User", body)

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _validate_and_accept(self) -> None:
        """Reject empty or obviously malformed input before hitting the database.

        The schema enforces both rules too (``CHECK (length(btrim(...)) > 0)``
        and the unique email index); catching them here turns a driver
        exception into a readable message.
        """
        if not self._name_input.text().strip():
            message_box.warn(self, "Name Needed", "Enter a name for this user.")
            return
        email = self._email_input.text().strip()
        if not email:
            message_box.warn(self, "Email Needed", "Enter an email address for this user.")
            return
        if "@" not in email or email.startswith("@") or email.endswith("@"):
            message_box.warn(
                self,
                "Check the Email",
                f"“{email}” doesn't look like an email address.",
                "It should be in the form name@example.com.",
            )
            return
        password = self._password_input.text()
        if password and len(password) < 8:
            message_box.warn(
                self,
                "Password Too Short",
                "Choose a password of at least 8 characters.",
                "Leave the field empty to keep this user's current password.",
            )
            return
        self.accept()

    def user(self) -> User:
        password = self._password_input.text()
        if password:
            password_hash = hash_password(password)
        elif self._original is not None:
            password_hash = self._original.password_hash
        else:
            password_hash = ""
        return User(
            id=self._original.id if self._original else None,
            name=self._name_input.text().strip(),
            email=self._email_input.text().strip(),
            role=UserRole.from_value(self._role_combo.currentData()),
            password_hash=password_hash,
            created_at=self._original.created_at if self._original else None,
        )


class UserManagerDialog(QDialog):
    """Full CRUD over the ``users`` table."""

    def __init__(
        self,
        catalog: ExperimentCatalogUseCase,
        stylesheet: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._catalog = catalog
        self._stylesheet = stylesheet
        self._users: list[User] = []

        self.setWindowTitle("Manage Users")
        self.setStyleSheet(stylesheet)
        self.setMinimumSize(720, 420)

        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["ID", "Name", "Email", "Role", "Created"])
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(30)
        self._table.horizontalHeader().setHighlightSections(False)
        self._table.itemSelectionChanged.connect(self._sync_buttons)
        self._table.doubleClicked.connect(self._edit_selected)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        self._table.setColumnWidth(0, 56)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self._table.setColumnWidth(3, 110)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        self._table.setColumnWidth(4, 140)

        self._add_button = QPushButton("＋  Add User")
        self._add_button.setCursor(Qt.PointingHandCursor)
        self._add_button.clicked.connect(self._add)

        self._edit_button = QPushButton("Edit")
        self._edit_button.setCursor(Qt.PointingHandCursor)
        self._edit_button.setEnabled(False)
        self._edit_button.clicked.connect(self._edit_selected)

        self._delete_button = QPushButton("Delete")
        self._delete_button.setObjectName("dangerButton")
        self._delete_button.setCursor(Qt.PointingHandCursor)
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self._delete_selected)

        close_button = QPushButton("Close")
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.clicked.connect(self.accept)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self._add_button)
        actions.addWidget(self._edit_button)
        actions.addWidget(self._delete_button)
        actions.addStretch(1)
        actions.addWidget(close_button)

        self._caption = QLabel()
        self._caption.setObjectName("captionLabel")

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._caption)
        layout.addLayout(actions)
        install_title_bar(self, "Manage Users", body)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            self._users = self._catalog.list_users()
        except Exception as exc:  # noqa: BLE001 - surfaced in the caption
            self._users = []
            self._caption.setText(f"Could not load users: {exc}")
            self._table.setRowCount(0)
            return

        self._table.setRowCount(len(self._users))
        for row, user in enumerate(self._users):
            cells = [
                str(user.id or ""),
                user.name,
                user.email,
                user.role.label,
                _format_timestamp(user.created_at),
            ]
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if column in (0, 3, 4):
                    item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(row, column, item)

        count = len(self._users)
        self._caption.setText(f"{count} user{'s' if count != 1 else ''}.")
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        has_selection = self._selected_user() is not None
        self._edit_button.setEnabled(has_selection)
        self._delete_button.setEnabled(has_selection)

    def _selected_user(self) -> User | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._users):
            return self._users[row]
        return None

    # ------------------------------------------------------------------
    def _add(self) -> None:
        dialog = UserEditorDialog(self._stylesheet, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        user = dialog.user()
        try:
            self._catalog.create_user(
                user.name, user.email, user.role, password_hash=user.password_hash
            )
        except ValueError as exc:
            # Duplicate email — the repository translates the unique-index
            # violation into this already-readable sentence.
            message_box.warn(self, "Couldn't Add User", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            message_box.error(
                self,
                "Couldn't Add User",
                f"{user.name} wasn't added.",
                "The database rejected the change. Try again in a moment.",
                technical=str(exc),
            )
            return
        self.refresh()

    def _edit_selected(self) -> None:
        user = self._selected_user()
        if user is None:
            return
        dialog = UserEditorDialog(self._stylesheet, user=user, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        try:
            self._catalog.update_user(dialog.user())
        except ValueError as exc:
            message_box.warn(self, "Couldn't Save User", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            message_box.error(
                self,
                "Couldn't Save User",
                f"Changes to {user.name} weren't saved.",
                "The database rejected the change. Try again in a moment.",
                technical=str(exc),
            )
            return
        self.refresh()

    def _delete_selected(self) -> None:
        user = self._selected_user()
        if user is None or user.id is None:
            return
        confirmed = message_box.confirm(
            self,
            "Delete User",
            f"Delete {user.name}?",
            f"{user.email}\n\nTheir experiments are kept, but stop being "
            "attributed to anyone. This can't be undone.",
            confirm_label="Delete",
            destructive=True,
        )
        if not confirmed:
            return
        try:
            self._catalog.delete_user(user.id)
        except Exception as exc:  # noqa: BLE001
            message_box.error(
                self,
                "Delete Failed",
                f"{user.name} wasn't deleted.",
                "The account is unchanged. Try again in a moment.",
                technical=str(exc),
            )
            return
        self.refresh()


__all__ = ["UserEditorDialog", "UserManagerDialog"]
