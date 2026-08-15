"""Vocabulary for reporting what a run is doing *while* it is doing it.

A run is one long blocking call from the UI's point of view, so without
this the only observable states are "started" and "finished" — an
indeterminate bar that says nothing about whether the app is reading a
40 MB ``.mat`` file, grinding through a 10-iteration eigendecomposition,
or fitting the baseline classifier. This module names the steps so the
window can say which one is in flight.

Stage names are the domain's, not the UI's: the presentation layer maps
them to icons and orders them, but does not invent its own list, so a new
step added to ``RunTransferLearningUseCase`` shows up in the status panel
without a second edit somewhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The steps a single run passes through, in the order ``execute`` runs
#: them. The UI renders this as a checklist, so the order matters here.
STAGE_LOADING = "Loading datasets"
STAGE_PREPROCESSING = "Preprocessing"
STAGE_ADAPTING = "Adapting"
STAGE_CLASSIFYING = "Classifying"
STAGE_BASELINE = "Baseline"
STAGE_ALIGNMENT = "Measuring alignment"
STAGE_FINALIZING = "Finalising"

SINGLE_RUN_STAGES: tuple[str, ...] = (
    STAGE_LOADING,
    STAGE_PREPROCESSING,
    STAGE_ADAPTING,
    STAGE_CLASSIFYING,
    STAGE_BASELINE,
    STAGE_ALIGNMENT,
    STAGE_FINALIZING,
)

#: Stages that only happen when both datasets carry labels. An unlabelled
#: run skips them entirely, so the UI greys them rather than leaving them
#: looking stuck.
LABELLED_ONLY_STAGES = frozenset({STAGE_CLASSIFYING, STAGE_BASELINE})


@dataclass(frozen=True)
class ProgressUpdate:
    """One "here is what I am doing now" message from a running experiment.

    Immutable so it can cross the worker/UI thread boundary as a plain
    value — the receiving slot reads a snapshot, never a live object the
    worker is still mutating underneath it.
    """

    stage: str
    #: What specifically, within the stage: "JDA iteration 3 of 10",
    #: "source.csv", "5-NN on 1800 target samples". May be empty.
    detail: str = ""
    #: Sub-steps done / expected inside this stage. ``total == 0`` means
    #: the stage has no countable sub-steps (it is one blocking call).
    completed: int = 0
    total: int = 0
    #: Which pair of a batch this belongs to; both 0 for a single run.
    pair_index: int = 0
    pair_total: int = 0
    #: Seconds of compute since the run started, excluding paused time.
    elapsed_seconds: float = 0.0

    @property
    def fraction(self) -> float | None:
        """Completion in 0..1, or ``None`` when nothing countable is known.

        For a batch this is pairs finished plus how far the current pair
        has got through the stage list. Counting whole pairs alone would
        read 100% the instant the *last* pair started rather than when it
        finished, which is the one moment a progress bar must not lie
        about; the stage term is what carries it across that pair.

        ``pair_index`` is 1-based, so ``pair_index - 1`` is the number of
        pairs actually behind us.
        """
        if self.pair_total > 0:
            try:
                stage_position = SINGLE_RUN_STAGES.index(self.stage) + 1
            except ValueError:
                stage_position = 0
            within_pair = stage_position / len(SINGLE_RUN_STAGES)
            done = (self.pair_index - 1) + within_pair
            return min(1.0, max(0.0, done / self.pair_total))
        if self.total > 0:
            return min(1.0, max(0.0, self.completed / self.total))
        return None

    @property
    def headline(self) -> str:
        """One line for the status bar: stage, then detail if there is one."""
        return f"{self.stage} — {self.detail}" if self.detail else self.stage


__all__ = [
    "LABELLED_ONLY_STAGES",
    "ProgressUpdate",
    "SINGLE_RUN_STAGES",
    "STAGE_ADAPTING",
    "STAGE_ALIGNMENT",
    "STAGE_BASELINE",
    "STAGE_CLASSIFYING",
    "STAGE_FINALIZING",
    "STAGE_LOADING",
    "STAGE_PREPROCESSING",
]
