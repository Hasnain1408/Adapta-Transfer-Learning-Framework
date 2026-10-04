"""Live run feedback: the status bar, the progress bar, and pause/cancel.

Everything that tells the user what a run is *doing right now*, as opposed
to starting it (main_window_execution.py) or showing what it produced
(main_window_results.py). _on_progress is the UI-thread end of the
worker's progress signal and must never block.
"""

from __future__ import annotations

from domain.progress import ProgressUpdate


class MainWindowProgressMixin:
    """Status text, progress-bar state, and the pause/cancel controls."""

    # ------------------------------------------------------------------
    # Live processing status
    # ------------------------------------------------------------------
    def _on_progress(self, update: ProgressUpdate) -> None:
        """Relay one worker-thread progress report to the UI.

        Arrives on the UI thread: RunControl calls the worker's
        progress.emit from the worker thread, and Qt queues the slot
        because the worker object's affinity is this thread. Nothing here
        may block — it runs between paint events.

        A pause is not overwritten: _toggle_pause has already put
        "Paused" on the status bar, and a report that was in flight when
        the user hit pause would otherwise flip it straight back to
        "Running".
        """
        self._processing_strip.apply(update)
        if self._run_control is not None and self._run_control.is_paused:
            return

        self._set_status(update.headline, "warning")

        # A batch knows how many pairs are left, so its bar can be honest
        # about how far along it is; a single run genuinely does not know
        # how long an eigendecomposition will take, and a fabricated
        # percentage would be worse than the indeterminate sweep.
        fraction = update.fraction if update.pair_total > 0 else None
        if fraction is None:
            if self._progress_bar.maximum() != 0:
                self._progress_bar.setRange(0, 0)
        else:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(int(round(fraction * 100)))

    def _set_status(self, text: str, color_key: str) -> None:
        """Show a status message. color_key names a semantic palette entry
        (e.g. "success", "warning") rather than a literal colour, so a theme
        switch can recolour the currently displayed status."""
        self._status_key = color_key
        color = self._palette[color_key]
        self._status_dot.setStyleSheet(f"color: {color};")
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    # ------------------------------------------------------------------
    # Running-state chrome
    # ------------------------------------------------------------------
    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._run_action.setEnabled(not running)
        if running:
            self._run_button.setText("Running…")
        else:
            # Through _sync_run_button_text rather than a second copy of the
            # rule here: the label depends on both the batch and comparison
            # toggles now, and a duplicate would silently go stale the next
            # time a mode is added.
            self._sync_run_button_text()
        self._progress_bar.setVisible(running)
        self._batch_mode_check.setEnabled(not running)
        self._combined_file_check.setEnabled(not running)
        self._add_batch_files_button.setEnabled(not running)
        self._remove_batch_file_button.setEnabled(not running)
        self._clear_batch_files_button.setEnabled(not running)
        self._clear_button.setEnabled(not running)
        # Locked for the run's duration like the batch controls: the request
        # was built from these, and letting them change mid-run would leave
        # the sidebar describing something other than what is executing.
        self._multi_method_check.setEnabled(
            not running and not self._batch_mode_check.isChecked()
        )
        self._method_list.setEnabled(not running)
        self._select_all_methods_button.setEnabled(not running)
        self._clear_methods_button.setEnabled(not running)

        self._pause_button.setEnabled(running)
        self._pause_button.setText("Pause")
        self._cancel_button.setEnabled(running)
        self._set_progress_bar_paused(False)

        if running:
            # Restart the cycle fresh on every run rather than continuing
            # from wherever a previous run left off.
            self._progress_color_index = 0
            self._progress_bar.setProperty("cycleColor", "0")
            self._progress_bar.style().unpolish(self._progress_bar)
            self._progress_bar.style().polish(self._progress_bar)
            self._progress_color_timer.start()
        else:
            self._progress_color_timer.stop()
            self._run_control = None

    def _set_progress_bar_paused(self, paused: bool) -> None:
        """Freeze the bar (static, no motion) and switch its colour while
        paused; restore the animated indeterminate sweep otherwise. A
        QProgressBar's built-in "busy" animation runs off its own Qt timer,
        independent of whether the underlying work is actually paused, so
        this has to be done explicitly rather than falling out of RunControl."""
        self._progress_bar.setRange(0, 1 if paused else 0)
        if paused:
            self._progress_bar.setValue(1)
        self._progress_bar.setProperty("paused", paused)
        self._progress_bar.style().unpolish(self._progress_bar)
        self._progress_bar.style().polish(self._progress_bar)

    def _advance_progress_bar_color(self) -> None:
        """Ticks once a second (see the QTimer wired up in
        MainWindowUIMixin._build_status_bar) while a run is active and not
        paused, stepping the chunk through cycleColor 0-3 so it visibly
        changes colour over time rather than sitting on one shade."""
        self._progress_color_index = (self._progress_color_index + 1) % 4
        self._progress_bar.setProperty("cycleColor", str(self._progress_color_index))
        self._progress_bar.style().unpolish(self._progress_bar)
        self._progress_bar.style().polish(self._progress_bar)

    # ------------------------------------------------------------------
    # Pause / cancel
    # ------------------------------------------------------------------
    def _toggle_pause(self) -> None:
        if self._run_control is None:
            return
        if self._run_control.is_paused:
            self._run_control.resume()
            self._pause_button.setText("Pause")
            self._set_progress_bar_paused(False)
            self._progress_color_timer.start()
            self._processing_strip.set_paused(False)
            # The stage in flight is the truest thing to show on resume; it
            # beats a generic "Running…" that the next report overwrites
            # anyway, and a stage blocked in numpy may not report for a while.
            last = self._run_control.last_update
            self._set_status(
                last.headline if last is not None else "Running…", "warning"
            )
        else:
            self._run_control.pause()
            self._pause_button.setText("Resume")
            self._set_progress_bar_paused(True)
            self._progress_color_timer.stop()
            self._processing_strip.set_paused(True)
            last = self._run_control.last_update
            self._set_status(
                f"Paused — {last.stage}" if last is not None else "Paused", "neutral"
            )

    def _cancel_run(self) -> None:
        if self._run_control is None:
            return
        self._run_control.cancel()
        # Disabled immediately so a second click can't fire while the worker
        # thread is still unwinding to its next checkpoint.
        self._cancel_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._processing_strip.set_paused(False)
        self._set_status("Cancelling…", "warning")

    def _on_cancelled(self) -> None:
        """A single-pair run was cancelled before it produced any result --
        batch cancellation instead comes back through _on_batch_success
        with cancelled=True, since a batch has partial results worth
        keeping. Nothing to record here: there's no run to show."""
        self._set_status("Cancelled", "neutral")
        # Left on screen rather than hidden: the stage still marked active
        # is the answer to "how far did it get before I stopped it?".
        self._processing_strip.finish("Cancelled")


__all__ = ["MainWindowProgressMixin"]
