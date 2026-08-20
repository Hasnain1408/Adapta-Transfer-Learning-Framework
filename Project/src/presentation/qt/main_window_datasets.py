"""Choosing the datasets a run will use.

Covers all three input modes the Datasets group offers — a source/target
pair, one combined file carrying both halves, and a batch file list — plus
the toggles that switch between them, including collapsing the whole group
once its inputs have been consumed by a finished run. Nothing here starts a
run; it only decides what a run would be given.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QDialog, QFileDialog, QLineEdit

from presentation.qt.main_window_ui import DATASET_FILTER
from presentation.qt.style import build_style_sheet
from presentation.qt.title_bar import install_title_bar_on_grid


class MainWindowDatasetsMixin:
    """File pickers, the combined-file toggle, and the batch file list."""

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pick_source(self) -> None:
        self._pick_file(self._source_input, "Select Source Dataset")

    def _pick_target(self) -> None:
        self._pick_file(self._target_input, "Select Target Dataset")

    def _swap_datasets(self) -> None:
        if self._combined_file_check.isChecked():
            # Nothing to swap: the target field is hidden and unused in
            # combined-file mode. Guarded here too since the Ctrl+Shift+S
            # shortcut and menu action stay reachable even while it's hidden.
            return
        source_text = self._source_input.text()
        target_text = self._target_input.text()
        self._source_input.setText(target_text)
        self._target_input.setText(source_text)

    def _pick_file(self, target_input: QLineEdit, title: str) -> None:
        start_dir = target_input.text().strip() or str(Path.cwd())
        dialog = QFileDialog(self, title, start_dir, DATASET_FILTER)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setStyleSheet(build_style_sheet(self._palette))
        install_title_bar_on_grid(dialog, title)
        if dialog.exec_() == QDialog.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files:
                target_input.setText(selected_files[0])

    # ------------------------------------------------------------------
    # Combined source/target file
    # ------------------------------------------------------------------
    def _on_combined_file_toggled(self, checked: bool) -> None:
        self._source_field_label.setText("Dataset File" if checked else "Source")
        self._target_field_label.setVisible(not checked)
        self._target_input.setVisible(not checked)
        self._swap_button.setVisible(not checked)

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Collapsing the section
    # ------------------------------------------------------------------
    def _toggle_datasets_collapsed(self) -> None:
        self._set_datasets_collapsed(self._dataset_body.isVisible())

    def _set_datasets_collapsed(self, collapsed: bool) -> None:
        """Hide or show the dataset fields, leaving the group's header.

        Collapsed, the group keeps a one-line summary of what is selected —
        a section that vanished into a bare title would leave the user
        unable to confirm which files a result came from without expanding
        it again.
        """
        self._dataset_body.setVisible(not collapsed)
        self._dataset_summary_label.setVisible(collapsed)
        # The group's 16px QSS padding is sized for a form; around a single
        # line of summary it reads as a mostly-empty box. A dynamic property
        # lets the sheet trim it, the same way the processing strip switches
        # its chip states — Qt only re-evaluates property selectors after an
        # unpolish/polish, hence the pair of calls.
        self._dataset_group.setProperty("collapsed", "true" if collapsed else "false")
        self._dataset_group.style().unpolish(self._dataset_group)
        self._dataset_group.style().polish(self._dataset_group)
        if collapsed:
            self._dataset_summary_label.setText(self._dataset_summary_text())
        self._dataset_toggle_button.setText("Show" if collapsed else "Hide")
        self._dataset_toggle_button.setToolTip(
            "Show the dataset fields again"
            if collapsed
            else "Collapse the dataset fields to give the results more room"
        )

    def _dataset_summary_text(self) -> str:
        """One line naming whatever the current mode has selected."""
        if self._batch_mode_check.isChecked():
            count = self._batch_files_list.count()
            if count < 2:
                return "Batch mode — not enough files selected"
            return f"Batch mode — {count} files, {count * (count - 1)} ordered pairs"

        source = self._source_input.text().strip()
        if not source:
            return "No datasets selected"
        if self._combined_file_check.isChecked():
            return f"{Path(source).name}  (combined source + target)"

        target = self._target_input.text().strip()
        if not target:
            return f"{Path(source).name}  →  no target selected"
        return f"{Path(source).name}  →  {Path(target).name}"

    def _on_batch_mode_toggled(self, checked: bool) -> None:
        self._single_dataset_widget.setVisible(not checked)
        self._batch_dataset_widget.setVisible(checked)
        # Batch varies the datasets, multi-method varies the method; doing
        # both at once is methods x pairs, which is a different feature with
        # a different results shape. Batch wins the toggle because it is the
        # one the user just asked for.
        if checked and self._multi_method_check.isChecked():
            self._multi_method_check.setChecked(False)
        self._multi_method_check.setEnabled(not checked)
        self._sync_run_button_text()

    def _add_batch_files(self) -> None:
        start_dir = str(Path.cwd())
        dialog = QFileDialog(self, "Select Dataset Files", start_dir, DATASET_FILTER)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setStyleSheet(build_style_sheet(self._palette))
        install_title_bar_on_grid(dialog, "Select Dataset Files")
        file_paths = []
        if dialog.exec_() == QDialog.Accepted:
            file_paths = dialog.selectedFiles()
        existing = {
            self._batch_files_list.item(i).text() for i in range(self._batch_files_list.count())
        }
        for path in file_paths:
            if path not in existing:
                self._batch_files_list.addItem(path)
                existing.add(path)
        self._update_batch_caption()

    def _remove_selected_batch_files(self) -> None:
        for item in self._batch_files_list.selectedItems():
            self._batch_files_list.takeItem(self._batch_files_list.row(item))
        self._update_batch_caption()

    def _clear_batch_files(self) -> None:
        self._batch_files_list.clear()
        self._update_batch_caption()

    def _update_batch_caption(self) -> None:
        count = self._batch_files_list.count()
        if count < 2:
            self._batch_files_caption.setText("Add 2+ files to enable batch mode.")
        else:
            pair_count = count * (count - 1)
            self._batch_files_caption.setText(
                f"{count} files selected → {pair_count} ordered pairs will be run."
            )


__all__ = ["MainWindowDatasetsMixin"]
