"""Live "what is it doing right now" strip for a running experiment.

Sits above the results area and is hidden whenever nothing is running. It
renders the stage list from ``domain.progress`` as a left-to-right
pipeline — done / running / pending — so a run that takes half a minute is
legible rather than being one indeterminate bar and a wait.

Horizontal, not a sidebar checklist: the sidebar is already full at the
window's default height (its parameter group alone wants 590px of 950),
so a seven-row column there overlaps the method controls. Width is the
dimension this layout has spare.

Everything here is driven by ``ProgressUpdate`` values arriving on the UI
thread via ``TransferLearningWorker.progress``. The strip keeps no timer
of its own for the *work*; only the elapsed clock ticks locally, because
a stage that blocks for eight seconds sends nothing during those eight
seconds and a frozen counter reads as a frozen app.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from domain.progress import LABELLED_ONLY_STAGES, SINGLE_RUN_STAGES, ProgressUpdate
from presentation.qt.duration import format_duration

#: Chip captions. Short so all seven fit across the content column even on
#: a narrow window; the full stage name stays available as the tooltip.
SHORT_LABELS = {
    "Loading datasets": "Load",
    "Preprocessing": "Preprocess",
    "Adapting": "Adapt",
    "Classifying": "Classify",
    "Baseline": "Baseline",
    "Measuring alignment": "Align",
    "Finalising": "Finish",
}

#: Markers. Deliberately plain glyphs rather than emoji — emoji render at
#: a different baseline and width across the platforms this runs on, and
#: these sit inline with body text.
MARKER_PENDING = "○"
MARKER_ACTIVE = "▸"
MARKER_DONE = "✓"
MARKER_SKIPPED = "–"

STATE_PENDING = "pending"
STATE_ACTIVE = "active"
STATE_DONE = "done"
STATE_SKIPPED = "skipped"

_MARKERS = {
    STATE_PENDING: MARKER_PENDING,
    STATE_ACTIVE: MARKER_ACTIVE,
    STATE_DONE: MARKER_DONE,
    STATE_SKIPPED: MARKER_SKIPPED,
}


class _StageChip(QLabel):
    """One stage in the pipeline: a marker and a short name."""

    def __init__(self, stage: str) -> None:
        super().__init__()
        self.stage = stage
        self._state = STATE_PENDING
        self.setObjectName("stageChip")
        self.setToolTip(stage)
        self.set_state(STATE_PENDING)

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state
        self.setText(f"{_MARKERS[state]}  {SHORT_LABELS.get(self.stage, self.stage)}")
        self.setProperty("stageState", state)
        # Qt does not re-evaluate property selectors on its own; without the
        # unpolish/polish pair the chip keeps the colour it was first
        # painted with for the whole run.
        self.style().unpolish(self)
        self.style().polish(self)


class ProcessingStatusStrip(QFrame):
    """The strip itself. Call ``start`` / ``apply`` / ``finish`` / ``clear``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("processingStrip")

        self._chips: dict[str, _StageChip] = {}
        pipeline = QHBoxLayout()
        pipeline.setContentsMargins(0, 0, 0, 0)
        pipeline.setSpacing(0)
        for position, stage in enumerate(SINGLE_RUN_STAGES):
            if position:
                arrow = QLabel("›")
                arrow.setObjectName("stageArrow")
                pipeline.addWidget(arrow)
            chip = _StageChip(stage)
            self._chips[stage] = chip
            pipeline.addWidget(chip)
        pipeline.addStretch(1)

        self._elapsed = QLabel()
        self._elapsed.setObjectName("processingElapsed")
        self._elapsed.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pipeline.addWidget(self._elapsed, 0)

        self._caption = QLabel()
        self._caption.setObjectName("processingCaption")
        # No word wrap: this label changes on every update, and a wrapping
        # label that grows to two lines mid-run makes the whole results
        # area below it jump. Elide long text instead by letting the layout
        # clip it — the same text is on the status bar in full.
        self._caption.setWordWrap(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)
        layout.addLayout(pipeline)
        layout.addWidget(self._caption)

        #: Ticks the elapsed clock while a stage is mid-computation and
        #: sending nothing. 200ms looks live without repainting constantly.
        self._clock = QTimer(self)
        self._clock.setInterval(200)
        self._clock.timeout.connect(self._tick)

        self._elapsed_seconds = 0.0
        self._scope = ""
        self._outcome = ""
        self._activity = ""
        self._paused = False
        self._active_stage: str | None = None
        self.hide()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, scope: str, labelled: bool = True) -> None:
        """Show the strip with every stage pending and the clock running.

        *labelled* is the caller's best guess at whether metrics will be
        produced; the classifier and baseline stages are marked skipped up
        front when it is false, so they don't sit looking stuck forever on
        an unlabelled dataset.
        """
        self._elapsed_seconds = 0.0
        self._scope = scope
        self._outcome = ""
        self._activity = ""
        self._paused = False
        self._active_stage = None
        for stage, chip in self._chips.items():
            skipped = not labelled and stage in LABELLED_ONLY_STAGES
            chip.set_state(STATE_SKIPPED if skipped else STATE_PENDING)
        self._render_elapsed()
        self._render_caption("Starting…")
        self.show()
        self._clock.start()

    def apply(self, update: ProgressUpdate) -> None:
        """Advance the pipeline to the stage *update* announces."""
        self._elapsed_seconds = update.elapsed_seconds
        self._render_elapsed()

        detail = update.detail
        if update.pair_total > 0:
            prefix = f"{update.unit_label} {update.pair_index} of {update.pair_total}"
            detail = f"{prefix}  ·  {detail}" if detail else prefix
        self._render_caption(f"{update.stage} — {detail}" if detail else update.stage)

        if update.stage not in self._chips:
            return

        # Every stage listed before this one has necessarily finished, and
        # settling them here rather than on completion means a stage skipped
        # at runtime (no labels, so no baseline) still resolves instead of
        # being stranded as "pending" for the whole run.
        reached = False
        for stage in SINGLE_RUN_STAGES:
            chip = self._chips[stage]
            if stage == update.stage:
                reached = True
                chip.set_state(STATE_ACTIVE)
            elif not reached:
                if chip.state != STATE_SKIPPED:
                    chip.set_state(STATE_DONE)
            elif update.pair_total > 0 and chip.state == STATE_DONE:
                # A batch loops back to stage one on the next pair, so the
                # later chips must reset rather than stay ticked.
                chip.set_state(STATE_PENDING)
        self._active_stage = update.stage

    def set_paused(self, paused: bool) -> None:
        """Freeze the clock and say so. The pipeline is left as-is: the run
        really is halfway through that stage, and blanking it would lose the
        one piece of information a paused user came to read."""
        self._paused = paused
        if paused:
            self._clock.stop()
        else:
            self._clock.start()
        self._render_caption()

    def finish(self, outcome: str) -> None:
        """Stop the clock and settle the pipeline.

        The strip stays visible after a run so the last state is readable —
        including, after a failure, which stage it died in.
        """
        self._clock.stop()
        self._paused = False
        self._outcome = outcome
        if outcome.lower().startswith("completed"):
            for chip in self._chips.values():
                if chip.state != STATE_SKIPPED:
                    chip.set_state(STATE_DONE)
        self._render_caption()

    def clear(self) -> None:
        self._clock.stop()
        self._active_stage = None
        self._scope = ""
        self._outcome = ""
        self._activity = ""
        self._paused = False
        self._caption.clear()
        self._elapsed.clear()
        for chip in self._chips.values():
            chip.set_state(STATE_PENDING)
        self.hide()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        self._elapsed_seconds += self._clock.interval() / 1000.0
        self._render_elapsed()

    def _render_elapsed(self) -> None:
        self._elapsed.setText(format_duration(self._elapsed_seconds) or "0.000s")

    def _render_caption(self, activity: str | None = None) -> None:
        """Caption line: what is running, on what. *activity* is remembered
        so a later pause/finish can re-render around the same text."""
        if activity is not None:
            self._activity = activity

        # A terminal outcome ("Completed", "Failed") replaces the activity —
        # "Completed · Finalising — assembling results" is just noise. A
        # pause keeps it, because *which* stage it stopped in is the whole
        # reason someone reads this line after hitting Pause.
        if self._outcome:
            lead = self._outcome
        elif self._paused:
            # "at", not another dash: the activity already contains an em
            # dash of its own, and "Paused — Adapting — MEDA on 2000+1800
            # samples" reads as three unrelated fragments.
            lead = f"Paused at {self._activity}" if self._activity else "Paused"
        else:
            lead = self._activity
        self._caption.setText("  ·  ".join(p for p in (lead, self._scope) if p))


__all__ = ["ProcessingStatusStrip"]
