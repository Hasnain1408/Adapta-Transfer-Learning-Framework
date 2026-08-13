from __future__ import annotations

import threading


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

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def cancel(self) -> None:
        self._cancelled.set()
        # Wake up a paused checkpoint so it can observe the cancellation
        # instead of blocking forever.
        self._running.set()

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
