"""Result rendering for ``MainWindow``: metrics, predictions, features, fullscreen.

Populates the widgets ``MainWindowUIMixin`` builds once a run has finished —
called from ``MainWindowRunMixin``'s success handlers. Kept separate from run
orchestration because rendering has no opinion on *how* a result was
produced (single run vs. one pair of a batch).
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem

from domain.models import EvaluationMetrics, TransferLearningResult
from presentation.qt.main_window_ui import FEATURE_PREVIEW_ROWS, FullScreenViewer
from presentation.qt.result_tables import populate_predictions_table
from presentation.qt.style import build_style_sheet


class MainWindowResultsMixin:
    """Metrics/predictions/dataset-summary/transformed-features rendering."""

    def _show_metrics(
        self, metrics: EvaluationMetrics | None, timing_note: str = ""
    ) -> None:
        """Fill the four metric cards and the caption beneath them.

        *timing_note* is appended to the caption — the execution time is not
        derivable from ``metrics``, and an unlabelled run still has a timing
        worth showing even though it has no metrics at all.
        """
        suffix = f"  ·  {timing_note}" if timing_note else ""
        if metrics is None:
            for card in self._metric_cards:
                card.set_value(None)
            self._counts_label.setText(
                "Metrics unavailable — the source or target dataset has no labels."
                + suffix
            )
            return

        self._accuracy_card.set_value(f"{metrics.accuracy:.4f}")
        self._precision_card.set_value(f"{metrics.precision:.4f}")
        self._recall_card.set_value(f"{metrics.recall:.4f}")
        self._f1_card.set_value(f"{metrics.f1_score:.4f}")
        self._counts_label.setText(
            f"{metrics.total_samples} target samples evaluated  ·  "
            f"{metrics.correct_predictions} correct  ·  "
            f"{metrics.wrong_predictions} incorrect" + suffix
        )

    def _show_dataset_summary(self, result: TransferLearningResult) -> None:
        self._summary_view.update_result(result)

    def _show_predictions(self, result: TransferLearningResult) -> None:
        if result.target_predictions is None:
            self._predictions_table.clearContents()
            self._predictions_table.setRowCount(0)
            self._incorrect_only_check.setEnabled(False)
            self._correct_only_check.setEnabled(False)
            self._results_tabs.setTabEnabled(0, False)
            self._results_tabs.setCurrentIndex(1)
            return

        self._results_tabs.setTabEnabled(0, True)
        self._incorrect_only_check.setEnabled(True)
        self._correct_only_check.setEnabled(True)
        self._results_tabs.setCurrentIndex(0)

        populate_predictions_table(self._predictions_table, result, self._palette)
        self._apply_prediction_filter()

    def _on_prediction_filter_toggled(self) -> None:
        sender = self.sender()
        if sender is None or not sender.isChecked():
            self._apply_prediction_filter()
            return
        other = self._incorrect_only_check if sender is self._correct_only_check else self._correct_only_check
        other.blockSignals(True)
        other.setChecked(False)
        other.blockSignals(False)
        self._apply_prediction_filter()

    def _apply_prediction_filter(self) -> None:
        """Hide rows according to the active filter."""
        result = self._last_result
        if result is None or result.prediction_correctness is None:
            return

        correct_only = self._correct_only_check.isChecked()
        incorrect_only = self._incorrect_only_check.isChecked()
        for row, correctness in enumerate(result.prediction_correctness):
            hide = (
                incorrect_only and correctness == "Right"
            ) or (correct_only and correctness == "Wrong")
            self._predictions_table.setRowHidden(row, hide)

    def _open_results_fullscreen(self) -> None:
        """Move the whole results tab widget into a maximized window.

        All tabs (Predictions, Visualizations, Transformed Features, Dataset
        Summary, Batch Results, History) move together and stay switchable —
        this is one full-screen view over the tab strip, not a separate one
        per tab. ``self._results_tabs`` is reparented in place (not copied),
        so it keeps its live state, and is handed back to the same slot in
        ``self._results_stack`` when the window closes.
        """
        win = getattr(self, "_results_window", None)
        try:
            if win is not None:
                win.raise_()
                win.activateWindow()
                return
        except RuntimeError:
            # The user already closed the previous window (WA_DeleteOnClose).
            pass

        origin_index = self._results_stack.indexOf(self._results_tabs)
        self._results_stack.removeWidget(self._results_tabs)
        # The corner button travels with the tab widget (it's the same
        # instance, just reparented) — hide it so a window already in full
        # screen doesn't show a button to open full screen again.
        self._results_fullscreen_button.hide()
        win = FullScreenViewer(
            "Results", self._results_tabs, build_style_sheet(self._palette), parent=None
        )
        # QStackedWidget.removeWidget() leaves the widget hidden (it was only
        # ever shown as the "current" stack page) — without this it silently
        # stays invisible in its new window.
        self._results_tabs.show()
        self._results_window = win

        def _restore() -> None:
            self._results_stack.insertWidget(origin_index, self._results_tabs)
            self._results_tabs.show()
            self._results_stack.setCurrentWidget(self._results_tabs)
            self._results_fullscreen_button.show()
            self._results_window = None

        win.destroyed.connect(_restore)

    def _show_transformed_features(self, result: TransferLearningResult) -> None:
        matrix = result.transformed_target
        table = self._features_table
        table.clearContents()

        headers = ["Sample"] + list(result.output_feature_names)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        self._align_headers(table, [Qt.AlignLeft] + [Qt.AlignRight] * (len(headers) - 1))

        total_rows = matrix.shape[0]
        shown_rows = min(total_rows, FEATURE_PREVIEW_ROWS)
        table.setRowCount(shown_rows)

        for row in range(shown_rows):
            name_item = QTableWidgetItem(f"Sample {row + 1}")
            table.setItem(row, 0, name_item)
            for column in range(matrix.shape[1]):
                item = QTableWidgetItem(f"{matrix[row, column]:.6f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row, column + 1, item)

        if shown_rows < total_rows:
            self._features_caption.setText(
                f"Showing the first {shown_rows} of {total_rows} transformed target rows."
            )
        else:
            self._features_caption.setText(
                f"{total_rows} transformed target rows × {matrix.shape[1]} components."
            )
        self._features_caption.show()

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)


__all__ = ["MainWindowResultsMixin"]
