"""Auth, database dialogs, and experiment-history persistence for ``MainWindow``.

Everything here either talks to ``ExperimentCatalogUseCase`` /
``AuthenticationUseCase`` or manages the signed-in account and history state
that gates them. Kept apart from run orchestration (``main_window_run.py``)
and result rendering (``main_window_results.py``) because those work whether
or not a database is connected — this mixin is the part that doesn't.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QDialog, QTabBar

from domain.entities import ExperimentHistoryEntry, ExperimentRecord, UserRole
from domain.models import (
    BatchTransferLearningResult,
    TransferLearningRequest,
    TransferLearningResult,
)
from presentation.qt.auth_dialogs import AuthDialog
from presentation.qt.database_dialogs import DatasetRegistryDialog, UserManagerDialog
from presentation.qt.experiment_detail_view import ExperimentDetailView
from presentation.qt import message_box
from presentation.qt.pair_detail_view import PairDetailView
from presentation.qt.style import build_style_sheet


class MainWindowSessionMixin:
    """Auth, database dialogs, and history persistence for ``MainWindow``."""

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
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

    def _set_history_visible(self, visible: bool) -> None:
        if not hasattr(self, "_results_tabs"):
            return

        index = self._results_tabs.indexOf(self._history_tab)
        if visible:
            if index < 0:
                insert_at = min(self._history_tab_index, self._results_tabs.count())
                self._results_tabs.insertTab(insert_at, self._history_tab, "History")
                # insertTab always gives a fresh tab a close button when the
                # widget is tabsClosable — History is permanent like the other
                # five fixed tabs, so strip it (as _build_results_tabs does
                # for the initial six) or it would show a close "x" that,
                # if clicked, deletes the singleton HistoryTab widget.
                self._results_tabs.tabBar().setTabButton(
                    insert_at, QTabBar.RightSide, None
                )
        elif index >= 0:
            if self._results_tabs.currentIndex() == index:
                self._results_tabs.setCurrentIndex(0)
            self._results_tabs.removeTab(index)

    def _handle_auth_button_clicked(self) -> None:
        if self._current_user is None:
            self._show_login_dialog()
        else:
            self._sign_out()

    def _show_login_dialog(self) -> None:
        if self._auth is None:
            message_box.information(
                self,
                "Log In Unavailable",
                "Sign-in is only available when the database connection is active.",
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
        message_box.information(
            self,
            "Logged Out",
            "You have been logged out. You can keep running experiments, but history stays hidden until you log in again.",
        )

    def _current_user_id(self) -> int | None:
        return self._current_user.id if self._current_user else None

    def _persist_run(
        self,
        request: TransferLearningRequest | None,
        result: TransferLearningResult | None,
        error_message: str | None,
    ) -> None:
        """Record one run's datasets and experiment. No history refresh.

        Unauthenticated runs (without log in) are NOT recorded into the database.
        Shared by both the single-run and batch completion handlers, so every
        pair in a batch is catalogued exactly like a standalone run.
        """
        if (
            self._catalog is None
            or request is None
            or self._current_user is None
            or self._current_user.id is None
        ):
            return

        supported = self._use_case.supported_parameters(request.method_name)

        try:
            if error_message is None and result is not None:
                self._catalog.record_success(
                    request, result, self._current_user_id(), supported
                )
            else:
                self._catalog.record_failure(
                    request, self._current_user_id(), error_message or "Unknown error", supported
                )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Run not recorded: {exc}", "warning")

    def _record_run(
        self, result: TransferLearningResult | None, error_message: str | None
    ) -> None:
        """Persist the single run that just finished and refresh History."""
        self._persist_run(self._last_request, result, error_message)
        self._history_tab.refresh()
        self._comparison_tab.refresh()

    def _record_batch_run(self, batch_result: BatchTransferLearningResult) -> None:
        """Persist every pair of a batch run as its own experiment/dataset pair."""
        shared = self._last_batch_request
        if shared is None:
            return

        for pair in batch_result.pairs:
            pair_request = TransferLearningRequest(
                method_name=shared.method_name,
                source_path=pair.source_path,
                target_path=pair.target_path,
                normalization=shared.normalization,
                standardization=shared.standardization,
                kernel_type=shared.kernel_type,
                regularization=shared.regularization,
                latent_dim=shared.latent_dim,
                classifier=shared.classifier,
            )
            self._persist_run(pair_request, pair.result, pair.error)

        self._history_tab.refresh()
        self._comparison_tab.refresh()

    def _show_history(self) -> None:
        """Bring the History tab to the front and reload user-specific history."""
        if self._catalog is None or self._current_user is None:
            return
        self._history_tab.set_current_user(self._current_user)
        self._comparison_tab.set_current_user(self._current_user)
        self._set_history_visible(True)
        self._results_stack.setCurrentIndex(1)
        index = self._results_tabs.indexOf(self._history_tab)
        if index >= 0:
            self._results_tabs.setCurrentIndex(index)

    def _open_user_manager(self) -> None:
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
        if self._catalog is None:
            return
        dialog = DatasetRegistryDialog(
            self._catalog, build_style_sheet(self._palette), self
        )
        dialog.exec_()

    def _show_database_info(self) -> None:
        if self._catalog is None:
            message_box.warning(
                self,
                "Database Unavailable",
                "No PostgreSQL connection.\n\n"
                + (self._database_error or "No connection details were configured."),
            )
            return

        try:
            stats = self._catalog.statistics()
        except Exception as exc:  # noqa: BLE001
            message_box.critical(self, "Database Error", str(exc))
            return

        message_box.information(
            self,
            "Database Connection",
            "<b>PostgreSQL connected</b>"
            "<p>Stored records:</p>"
            f"<ul><li>Users: {stats['users']}</li>"
            f"<li>Datasets: {stats['datasets']}</li>"
            f"<li>Experiments: {stats['experiments']}</li>"
            f"<li>Evaluations: {stats['evaluations']}</li></ul>",
        )

    def _apply_configuration(self, experiment: ExperimentRecord) -> None:
        """Copy a stored experiment's settings back into the run form.

        Only the settings — not the dataset paths, which may since have moved
        or been deleted. Selections are applied by ``findData``/``findText`` so
        a value the current build no longer offers is skipped rather than
        silently landing on the wrong entry.
        """
        method_index = self._method_combo.findText(experiment.tl_algorithm)
        if method_index >= 0:
            self._method_combo.setCurrentIndex(method_index)

        for combo, value in (
            (self._classifier_combo, experiment.classifier),
            (self._norm_combo, experiment.normalization_type),
            (self._std_combo, experiment.standardization_type),
            (self._kernel_combo, experiment.kernel_type),
        ):
            if value is None:
                continue
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)

        if experiment.regularization_value is not None:
            self._reg_input.setValue(float(experiment.regularization_value))
        if experiment.latent_feature is not None:
            self._dim_input.setValue(int(experiment.latent_feature))

        self._sync_method_parameters()
        self._set_status(
            f"Loaded configuration from experiment #{experiment.id}", "neutral"
        )

    def _open_experiment_detail_tab(self, entry: ExperimentHistoryEntry) -> None:
        """Open (or raise) a closable tab with every field of *entry*.

        Joins in the source/target ``DatasetRecord`` rows, which
        ``ExperimentHistoryEntry`` doesn't carry (it only denormalises the
        domain *names*) — this is the one place that needs the full rows.
        """
        experiment_id = entry.experiment.id
        if experiment_id is None or self._catalog is None:
            return

        existing = self._experiment_detail_tabs.get(experiment_id)
        if existing is not None:
            index = self._results_tabs.indexOf(existing)
            if index >= 0:
                self._results_tabs.setCurrentIndex(index)
                return
            # The tab was closed without going through _close_results_tab
            # (shouldn't normally happen); fall through and rebuild it.
            del self._experiment_detail_tabs[experiment_id]

        try:
            source_dataset = self._catalog.get_dataset(entry.experiment.source_dataset_id)
            target_dataset = self._catalog.get_dataset(entry.experiment.target_dataset_id)
        except Exception:  # noqa: BLE001 - the detail view just shows "—" instead
            source_dataset = None
            target_dataset = None

        view = ExperimentDetailView(entry, source_dataset, target_dataset, self._palette)
        view.export_requested.connect(
            lambda: self._export_history_report(entry, source_dataset, target_dataset)
        )
        index = self._results_tabs.addTab(view, f"Experiment #{experiment_id}")
        self._experiment_detail_tabs[experiment_id] = view
        self._results_tabs.setCurrentIndex(index)

    def _export_history_entry(self, entry: ExperimentHistoryEntry) -> None:
        """Export straight from the History table's toolbar, without going
        through a detail tab first — same dataset lookup as opening one."""
        if self._catalog is None:
            return
        try:
            source_dataset = self._catalog.get_dataset(entry.experiment.source_dataset_id)
            target_dataset = self._catalog.get_dataset(entry.experiment.target_dataset_id)
        except Exception:  # noqa: BLE001 - the report just shows "—" instead
            source_dataset = None
            target_dataset = None
        self._export_history_report(entry, source_dataset, target_dataset)

    def _open_pair_detail_tab(self) -> None:
        """Open (or raise) a closable tab for the row selected in Batch Results.

        Mirrors ``_open_experiment_detail_tab``'s "select a row, get a tab"
        pattern. Fires on every selection change, so re-selecting an
        already-open pair's row raises its tab instead of duplicating it.
        Batch pairs have no persisted id (unlike an ``ExperimentRecord``), so
        the pair object's identity is the key instead.
        """
        if self._last_batch_result is None:
            return
        row = self._batch_results_table.currentRow()
        if not (0 <= row < len(self._last_batch_result.pairs)):
            return
        pair = self._last_batch_result.pairs[row]
        pair_key = id(pair)

        existing = self._pair_detail_tabs.get(pair_key)
        if existing is not None:
            index = self._results_tabs.indexOf(existing)
            if index >= 0:
                self._results_tabs.setCurrentIndex(index)
                return
            del self._pair_detail_tabs[pair_key]

        view = PairDetailView(pair, self._palette)
        view.export_requested.connect(lambda: self._export_pair_report(pair, view.chart_html()))
        source_name = Path(pair.source_path).stem
        target_name = Path(pair.target_path).stem
        index = self._results_tabs.addTab(view, f"{source_name} → {target_name}")
        self._pair_detail_tabs[pair_key] = view
        self._results_tabs.setCurrentIndex(index)

    def _close_results_tab(self, index: int) -> None:
        """Handle the tab strip's close button — only detail tabs have one.

        Guards against ever deleting one of the six permanent tabs (their
        close buttons are stripped, but this is a second line of defence):
        only a widget actually tracked in ``_experiment_detail_tabs`` or
        ``_pair_detail_tabs`` gets removed and deleted.
        """
        widget = self._results_tabs.widget(index)
        experiment_id = next(
            (eid for eid, view in self._experiment_detail_tabs.items() if view is widget),
            None,
        )
        if experiment_id is not None:
            del self._experiment_detail_tabs[experiment_id]
            self._results_tabs.removeTab(index)
            widget.deleteLater()
            return

        pair_key = next(
            (key for key, view in self._pair_detail_tabs.items() if view is widget),
            None,
        )
        if pair_key is None:
            return
        del self._pair_detail_tabs[pair_key]
        self._results_tabs.removeTab(index)
        widget.deleteLater()


__all__ = ["MainWindowSessionMixin"]
