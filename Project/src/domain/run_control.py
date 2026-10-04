"""Execution control and thread synchronization for active experiment runs."""

from __future__ import annotations

import threading
from time import perf_counter
from typing import Callable

from domain.progress import ProgressUpdate

# Minimum interval between duplicate progress notifications to prevent event flooding.
_MIN_REPEAT_INTERVAL = 0.05

# Maximum iteration updates emitted per iterative method to bound logging overhead.
_MAX_ITERATION_REPORTS = 50


class RunCancelled(Exception):
    """Raised from RunControl.checkpoint when a running experiment has been cancelled."""


class RunControl:
    """Thread-safe synchronization channel between worker and UI threads.

    Provides cooperative pause, resume, and cancellation mechanisms downwards to
    workers, while streaming progress notifications upwards to user interfaces.
    """

    def __init__(self) -> None:
        self._running = threading.Event()
        self._running.set()
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._started = perf_counter()
        self._listener: Callable[[ProgressUpdate], None] | None = None
        self._last_update: ProgressUpdate | None = None
        self._last_emit = 0.0
        self._pair_index = 0
        self._pair_total = 0
        self._unit_label = "Pair"
        # Total paused duration in seconds, excluding open pauses.
        self._paused_total = 0.0
        self._paused_since: float | None = None

    def pause(self) -> None:
        """Signal the running experiment to pause at the next checkpoint."""
        with self._lock:
            if self._paused_since is None:
                self._paused_since = perf_counter()
        self._running.clear()

    def resume(self) -> None:
        """Signal a paused experiment to resume execution."""
        self._close_pause_window()
        self._running.set()

    def cancel(self) -> None:
        """Signal the running experiment to abort at the next checkpoint."""
        self._cancelled.set()
        self._close_pause_window()
        # Wake up a paused checkpoint so cancellation is processed immediately.
        self._running.set()

    def _close_pause_window(self) -> None:
        with self._lock:
            if self._paused_since is not None:
                self._paused_total += perf_counter() - self._paused_since
                self._paused_since = None

    @property
    def paused_seconds(self) -> float:
        """Total duration spent paused in seconds, including any active pause."""
        with self._lock:
            total = self._paused_total
            if self._paused_since is not None:
                total += perf_counter() - self._paused_since
            return total

    @property
    def is_paused(self) -> bool:
        """True if the controller is currently in a paused state."""
        return not self._running.is_set()

    @property
    def is_cancelled(self) -> bool:
        """True if cancellation has been requested."""
        return self._cancelled.is_set()

    def checkpoint(self) -> None:
        """Cooperative synchronization checkpoint called between execution steps.

        Blocks execution while paused. Raises RunCancelled if cancellation was requested.
        """
        self._running.wait()
        if self._cancelled.is_set():
            raise RunCancelled()

    # ------------------------------------------------------------------
    # Progress Reporting Helpers
    # ------------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        """Active compute time in seconds since start, excluding paused durations."""
        return max(0.0, perf_counter() - self._started - self.paused_seconds)

    def set_listener(self, listener: Callable[[ProgressUpdate], None] | None) -> None:
        """Register a callback listener to receive progress update snapshots."""
        with self._lock:
            self._listener = listener

    def set_pair(self, index: int, total: int, label: str = "Pair") -> None:
        """Set multi-item batch execution context (e.g. Pair 2 of 5)."""
        with self._lock:
            self._pair_index = index
            self._pair_total = total
            self._unit_label = label

    def report(self, stage: str, detail: str = "", completed: int = 0, total: int = 0) -> None:
        """Emit a progress update notification to the registered listener."""
        now = perf_counter()
        with self._lock:
            listener = self._listener
            previous = self._last_update
            if (
                previous is not None
                and previous.stage == stage
                and previous.detail == detail
                and now - self._last_emit < _MIN_REPEAT_INTERVAL
            ):
                return
            update = ProgressUpdate(
                stage=stage,
                detail=detail,
                completed=completed,
                total=total,
                pair_index=self._pair_index,
                pair_total=self._pair_total,
                unit_label=self._unit_label,
                elapsed_seconds=max(0.0, now - self._started - self._paused_total),
            )
            self._last_update = update
            self._last_emit = now

        if listener is not None:
            try:
                listener(update)
            except Exception:  # Guard against listener failures
                pass

    def report_iteration(self, label: str, index: int, total: int) -> None:
        """Emit an iteration progress update for iterative optimization methods."""
        from domain.progress import STAGE_ADAPTING

        if total > _MAX_ITERATION_REPORTS:
            stride = total // _MAX_ITERATION_REPORTS
            if index % stride and index != total:
                return

        self.report(
            STAGE_ADAPTING,
            f"{label} iteration {index} of {total}",
            completed=index,
            total=total,
        )

    @property
    def last_update(self) -> ProgressUpdate | None:
        """Return the most recently emitted progress update snapshot."""
        with self._lock:
            return self._last_update
