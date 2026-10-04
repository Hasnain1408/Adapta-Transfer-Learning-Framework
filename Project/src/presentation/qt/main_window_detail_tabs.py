"""Closable per-experiment and per-pair detail tabs.

Both follow the same "select a row, get a tab" pattern and are tracked in
their own dict so a second click raises the existing tab instead of
duplicating it. _close_results_tab is the only path that removes one,
and it refuses to touch anything not in those dicts — the permanent tabs
must survive it.
"""

from __future__ import annotations

from pathlib import Path

from domain.entities import ExperimentHistoryEntry
from domain.models import PairTransferLearningResult
from presentation.qt.experiment_detail_view import ExperimentDetailView
from presentation.qt.pair_detail_view import PairDetailView


class MainWindowDetailTabsMixin:
    """Opening, exporting from, and closing the dynamic detail tabs."""

    def _open_experiment_detail_tab(self, entry: ExperimentHistoryEntry) -> None:
        """Open (or raise) a closable tab with every field of *entry*.

        Joins in the source/target DatasetRecord rows, which
        ExperimentHistoryEntry doesn't carry (it only denormalises the
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

        Mirrors _open_experiment_detail_tab's "select a row, get a tab"
        pattern. Fires on every selection change, so re-selecting an
        already-open pair's row raises its tab instead of duplicating it.
        Batch pairs have no persisted id (unlike an ExperimentRecord), so
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

    def _open_method_detail_tab(self) -> None:
        """Open (or raise) a tab for the method row selected in Method Results.

        The table is ordered by rank, not by the order the methods ran, so
        the row index is resolved through the same ranking the table used
        rather than indexing into methods directly — otherwise clicking
        the winner would open whichever method happened to run first.
        """
        multi_result = self._last_multi_result
        if multi_result is None:
            return
        row = self._method_results_table.currentRow()
        ordered = self._ranked_method_entries(multi_result)
        if not (0 <= row < len(ordered)):
            return
        entry = ordered[row]
        entry_key = id(entry)

        existing = self._method_detail_tabs.get(entry_key)
        if existing is not None:
            index = self._results_tabs.indexOf(existing)
            if index >= 0:
                self._results_tabs.setCurrentIndex(index)
                return
            del self._method_detail_tabs[entry_key]

        # PairDetailView already renders exactly this (metrics, predictions,
        # charts, dataset summary) for a batch pair; a comparison entry is
        # the same shape with a different identity, so it is wrapped rather
        # than duplicated into a near-identical second view class.
        pair = PairTransferLearningResult(
            source_path=multi_result.source_path,
            target_path=multi_result.target_path,
            result=entry.result,
            error=entry.error,
        )
        source_name = Path(multi_result.source_path).name
        target_name = Path(multi_result.target_path).name
        view = PairDetailView(
            pair,
            self._palette,
            title=entry.method_name,
            subtitle_prefix=f"{source_name} → {target_name}",
        )
        view.export_requested.connect(
            lambda: self._export_method_report(entry, pair, view.chart_html())
        )
        index = self._results_tabs.addTab(view, entry.method_name)
        self._method_detail_tabs[entry_key] = view
        self._results_tabs.setCurrentIndex(index)

    @staticmethod
    def _ranked_method_entries(multi_result):
        """Method entries in the order populate_method_results_table rows them."""
        ranked = multi_result.ranked()
        ranked_names = {entry.method_name for entry in ranked}
        return ranked + [
            entry for entry in multi_result.methods if entry.method_name not in ranked_names
        ]

    def _close_results_tab(self, index: int) -> None:
        """Handle the tab strip's close button — only detail tabs have one.

        Guards against ever deleting one of the six permanent tabs (their
        close buttons are stripped, but this is a second line of defence):
        only a widget actually tracked in _experiment_detail_tabs or
        _pair_detail_tabs gets removed and deleted.
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

        for tracker in (self._pair_detail_tabs, self._method_detail_tabs):
            key = next((k for k, view in tracker.items() if view is widget), None)
            if key is not None:
                del tracker[key]
                self._results_tabs.removeTab(index)
                widget.deleteLater()
                return


__all__ = ["MainWindowDetailTabsMixin"]
