from __future__ import annotations

import threading
from time import perf_counter


class RunCancelled(Exception):
    """Raised from ``RunControl.checkpoint`` once a run has been cancelled."""


class RunControl:
    """Cooperative pause/cancel signal shared between the UI thread and the
    worker thread actually running a (possibly long) experiment.

    There is no way to safely interrupt a single blocking numpy call (an
    eigendecomposition mid-flight can't be paused), so this only takes
    effect at *checkpoints* the run calls between logical steps -- the top
    of ``execute()``, each iteration of an iterative method, and each pair
    of a batch run. A fast one-shot method (CORAL, GFK, TCA) may simply
    finish before a pause/cancel has a chance to land.
    """

    def __init__(self) -> None:
        self._running = threading.Event()
        self._running.set()
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        #: Total seconds spent paused so far, so the timing the use case
        #: reports is compute cost and not "how long the user left it
        #: paused". Only whole pause/resume cycles land here; a pause still
        #: in progress is added on by the ``paused_seconds`` property.
        self._paused_total = 0.0
        self._paused_since: float | None = None

    def pause(self) -> None:
        with self._lock:
            if self._paused_since is None:
                self._paused_since = perf_counter()
        self._running.clear()

    def resume(self) -> None:
        self._close_pause_window()
        self._running.set()

    def cancel(self) -> None:
        self._cancelled.set()
        self._close_pause_window()
        # Wake up a paused checkpoint so it can observe the cancellation
        # instead of blocking forever.
        self._running.set()

    def _close_pause_window(self) -> None:
        with self._lock:
            if self._paused_since is not None:
                self._paused_total += perf_counter() - self._paused_since
                self._paused_since = None

    @property
    def paused_seconds(self) -> float:
        """Seconds spent paused, including a pause still open right now."""
        with self._lock:
            total = self._paused_total
            if self._paused_since is not None:
                total += perf_counter() - self._paused_since
            return total

    @property
    def is_paused(self) -> bool:
        return not self._running.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def checkpoint(self) -> None:
        """Call between logical steps. Blocks while paused; raises
        ``RunCancelled`` once cancelled, whether that happened while paused
        or not."""
        self._running.wait()
        if self._cancelled.is_set():
            raise RunCancelled()
