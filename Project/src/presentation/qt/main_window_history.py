"""Recording finished runs to the catalogue, and the History tab around them.

``_persist_run`` is the single write path — both the single-run and the
batch completion handlers funnel through it, so every pair of a batch is
catalogued exactly like a standalone run. The rest of this mixin is about
showing that history back: the tab's visibility, and replaying a stored
experiment's settings into the run form.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QTabBar

from domain.entities import ExperimentRecord
from domain.models import (
    BatchTransferLearningResult,
    MultiMethodTransferLearningResult,
    TransferLearningRequest,
    TransferLearningResult,
)


class MainWindowHistoryMixin:
    """Experiment persistence, the History tab, and configuration replay."""

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

    def _record_multi_method_run(
        self, multi_result: MultiMethodTransferLearningResult
    ) -> None:
        """Persist each method of a comparison as its own experiment row.

        Same treatment a batch gives its pairs: a comparison is a
        convenience for running several experiments under identical
        conditions, not a different kind of thing to store. Recording them
        individually is also what lets the database-backed Comparison tab
        pick them up afterwards.
        """
        shared = self._last_multi_request
        if shared is None:
            return

        for entry in multi_result.methods:
            method_request = TransferLearningRequest(
                method_name=entry.method_name,
                source_path=multi_result.source_path,
                target_path=multi_result.target_path,
                normalization=shared.normalization,
                standardization=shared.standardization,
                kernel_type=shared.kernel_type,
                regularization=shared.regularization,
                latent_dim=shared.latent_dim,
                mu=shared.mu,
                classifier=shared.classifier,
            )
            self._persist_run(method_request, entry.result, entry.error)

        self._history_tab.refresh()
        self._comparison_tab.refresh()

    def _show_history(self) -> None:
        """Bring the History tab to the front and reload user-specific history."""
        if self._reject_if_busy("refreshing history"):
            return
        if self._catalog is None or self._current_user is None:
            return
        self._history_tab.set_current_user(self._current_user)
        self._comparison_tab.set_current_user(self._current_user)
        self._set_history_visible(True)
        self._results_stack.setCurrentIndex(1)
        index = self._results_tabs.indexOf(self._history_tab)
        if index >= 0:
            self._results_tabs.setCurrentIndex(index)

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


__all__ = ["MainWindowHistoryMixin"]
