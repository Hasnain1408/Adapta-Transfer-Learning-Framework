"""Domain vocabulary and models for reporting experiment execution progress.

Defines standardized stage identifiers and snapshot update models passed across
worker and UI thread boundaries during long-running experiment executions.
"""

from __future__ import annotations

from dataclasses import dataclass

# Standardized sequential pipeline stages for experiment execution.
STAGE_LOADING = "Loading datasets"
STAGE_PREPROCESSING = "Preprocessing"
STAGE_ADAPTING = "Adapting"
STAGE_CLASSIFYING = "Classifying"
STAGE_BASELINE = "Baseline"
STAGE_ALIGNMENT = "Measuring alignment"
STAGE_FINALIZING = "Finalising"

# Ordered execution stages for a single transfer learning run.
SINGLE_RUN_STAGES: tuple[str, ...] = (
    STAGE_LOADING,
    STAGE_PREPROCESSING,
    STAGE_BASELINE,
    STAGE_ADAPTING,
    STAGE_CLASSIFYING,
    STAGE_ALIGNMENT,
    STAGE_FINALIZING,
)

# Stages executed exclusively when supervised ground-truth labels are present.
LABELLED_ONLY_STAGES = frozenset({STAGE_CLASSIFYING, STAGE_BASELINE})


@dataclass(frozen=True)
class ProgressUpdate:
    """Immutable progress snapshot emitted by an active experiment run."""

    stage: str
    # Specific detail text (such as JDA iteration 3 of 10 or source.csv).
    detail: str = ""
    # Sub-steps completed and total within the current stage.
    completed: int = 0
    total: int = 0
    # Multi-item run counter (index of current pair/method and total count).
    pair_index: int = 0
    pair_total: int = 0
    # Unit label for multi-item runs (such as Pair or Method).
    unit_label: str = "Pair"
    # Elapsed compute time in seconds, excluding paused duration.
    elapsed_seconds: float = 0.0

    @property
    def fraction(self) -> float | None:
        """Normalized completion progress between 0.0 and 1.0, or None if indeterminate."""
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
        """Formatted status bar line with stage name and optional detail."""
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
