from __future__ import annotations

import threading
from time import perf_counter
from typing import Callable

from domain.progress import ProgressUpdate

#: An update identical to the one before it is dropped if it arrives
#: faster than this. Only *literal* repeats are coalesced: anything the
#: user would see as new text goes through immediately, because dropping
#: on stage alone silently hid every preprocessing sub-step and nine of
#: JDA's ten iterations on a fast dataset.
_MIN_REPEAT_INTERVAL = 0.05

#: Upper bound on how many progress reports one iterative method emits.
#: Every method in the app today runs 10-20 iterations and reports all of
#: them; this only thins the stream if some future method loops thousands
#: of times, where per-iteration updates would be unreadable anyway.
_MAX_ITERATION_REPORTS = 50


class RunCancelled(Exception):
    """Raised from ``RunControl.checkpoint`` once a run has been cancelled."""


class RunControl:
    """The channel between the UI thread and the worker thread actually
    running a (possibly long) experiment: pause/cancel downwards, progress
    upwards.

    There is no way to safely interrupt a single blocking numpy call (an
    eigendecomposition mid-flight can't be paused), so pause/cancel only
    take effect at *checkpoints* the run calls between logical steps -- the
    top of ``execute()``, each iteration of an iterative method, and each
    pair of a batch run. A fast one-shot method (CORAL, GFK, TCA) may
    simply finish before a pause/cancel has a chance to land.

    Progress travels the same path for the same reason: ``control`` is
    already handed to every method's ``run()``, so an iterative method can
    say which iteration it is on without a second parameter threaded
    through ten call sites.
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

    # ------------------------------------------------------------------
    # Progress reporting (worker thread -> UI thread)
    # ------------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        """Compute seconds since the run started, paused time excluded."""
        return max(0.0, perf_counter() - self._started - self.paused_seconds)

    def set_listener(self, listener: Callable[[ProgressUpdate], None] | None) -> None:
        """Register who hears progress. Set from the UI thread *before* the
        worker starts; the listener is then called on the worker thread, so
        it must do nothing but hand the update across (a queued Qt signal
        emit, in this app)."""
        with self._lock:
            self._listener = listener

    def set_pair(self, index: int, total: int) -> None:
        """Batch context stamped onto every subsequent update.

        ``execute()`` has no idea it is one pair of a batch, so
        ``execute_batch`` sets this instead of passing it down.
        """
        with self._lock:
            self._pair_index = index
            self._pair_total = total

    def report(self, stage: str, detail: str = "", completed: int = 0, total: int = 0) -> None:
        """Announce the step now starting. Never raises.

        A run must not die because the UI's handler threw -- progress is
        commentary on the work, not part of it -- so the listener call is
        guarded.
        """
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
                # _paused_total, not the paused_seconds property: the lock
                # is non-reentrant and already held here. Any open pause is
                # excluded, which is harmless -- report() only runs after a
                # checkpoint has let the thread through, so no pause is open.
                elapsed_seconds=max(0.0, now - self._started - self._paused_total),
            )
            self._last_update = update
            self._last_emit = now

        if listener is not None:
            try:
                listener(update)
            except Exception:  # noqa: BLE001 - a broken listener must not sink the run
                pass

    def report_iteration(self, label: str, index: int, total: int) -> None:
        """Convenience for the iterative methods: "JDA iteration 3 of 10".

        *index* is 1-based (the iteration about to run).
        """
        from domain.progress import STAGE_ADAPTING

        if total > _MAX_ITERATION_REPORTS:
            # Report roughly _MAX_ITERATION_REPORTS times, evenly spaced,
            # always including the last iteration so the count still lands
            # on "n of n" rather than stopping short.
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
        with self._lock:
            return self._last_update
