"""The "is a run in flight?" guard every blocking action checks first.

Its own module because almost every other mixin depends on it — the session
mixins call ``_reject_if_busy`` before any database round-trip, the report
mixin before rendering a PDF, and the run mixin before starting a second
run. Keeping it separate makes that shared dependency explicit rather than
having the others reach into run orchestration for it.
"""

from __future__ import annotations

from presentation.qt import message_box


class MainWindowBusyMixin:
    """Detecting an in-flight run and refusing to stack work on top of it."""

    def _is_worker_busy(self) -> bool:
        """Whether a background run is currently in flight.

        Also clears a stale ``self._worker`` left over from a QThread whose
        C++ object was already deleted by ``deleteLater`` — the Python
        wrapper raises ``RuntimeError`` on any further method call once
        that happens, so this is the one safe place to check it.
        """
        try:
            return self._worker is not None and self._worker.isRunning()
        except RuntimeError:
            self._worker = None
            return False

    def _busy_description(self) -> str:
        """Name whatever background run is in flight, for the message
        ``_reject_if_busy`` shows. Reads the batch checkbox rather than
        ``_last_run_kind`` because it must describe the run *in progress*,
        and the checkbox is disabled (not reset) for the run's duration —
        it still reflects what was actually started."""
        if self._batch_mode_check.isChecked():
            return "The batch experiment"
        request = self._last_request
        return f"The {request.method_name} experiment" if request is not None else "The experiment"

    def _reject_if_busy(self, action: str) -> bool:
        """Block *action* while a run is in flight, telling the user why.

        Every entry point that does meaningful synchronous work on the UI
        thread — a database round-trip, PDF rendering, starting a second
        run — calls this first. Stacking that on top of the worker
        thread's CPU-bound computation is what turns "a bit slow" into
        "the whole app looks hung" (and, under enough load, can make the
        rest of the desktop feel unresponsive too); saying so explicitly
        beats a silent no-op the user is left to guess at.
        """
        if not self._is_worker_busy():
            return False
        message_box.info(
            self,
            "Experiment Running",
            f"{self._busy_description()} is still running.",
            f"Wait for it to finish before {action}, or press Cancel to stop it.",
        )
        return True


__all__ = ["MainWindowBusyMixin"]
