"""Persisted entities.

These mirror the four database tables (``users``, ``datasets``,
``experiments``, ``evaluations``) and are the only shape the application and
presentation layers deal in — nothing outside ``infrastructure/persistence``
sees a ``sqlite3.Row``.

Distinct from ``domain.models``: those describe a *run in flight* (feature
matrices, predictions), these describe a *record of a run* that outlives the
process. In particular ``DatasetRecord`` here is the catalogued metadata for a
dataset file, whereas ``models.Dataset`` holds the loaded matrix itself.

Every entity carries ``id: int | None`` — ``None`` before the row exists, set
by the repository on insert. All are frozen, so repositories return a new
instance via :func:`dataclasses.replace` rather than mutating in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """Who is running the experiment.

    Inherits from ``str`` so the value round-trips through SQLite (which has
    no enum type) and compares equal to the plain string in queries.
    """

    TEACHER = "teacher"
    RESEARCHER = "researcher"
    STUDENT = "student"
    ADMIN = "admin"

    @property
    def label(self) -> str:
        """Title-cased form for display in the UI."""
        return self.value.capitalize()

    @classmethod
    def from_value(cls, value: str) -> "UserRole":
        """Parse a stored/UI string, falling back to STUDENT for unknown input.

        Rows written by an older build (or hand-edited) should not crash the
        history view, so this never raises.
        """
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.STUDENT


class ExperimentStatus(str, Enum):
    """Lifecycle of a single experiment row."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def label(self) -> str:
        return self.value.capitalize()

    @classmethod
    def from_value(cls, value: str) -> "ExperimentStatus":
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.PENDING


class DatasetFormat(str, Enum):
    """File formats the loader accepts, stored verbatim including the dot."""

    CSV = ".csv"
    XLSX = ".xlsx"
    MAT = ".mat"

    @classmethod
    def from_path(cls, path: str) -> "DatasetFormat":
        """Derive the format from a file name's suffix.

        Raises ``ValueError`` for anything the loader could not read anyway,
        so a bad format is caught before a row is written.
        """
        from pathlib import Path

        suffix = Path(path).suffix.lower()
        try:
            return cls(suffix)
        except ValueError as exc:
            supported = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unsupported dataset format {suffix!r}. Supported: {supported}"
            ) from exc


@dataclass(frozen=True)
class User:
    """A person who owns experiments.

    ``email`` is unique (case-insensitively) in the database and acts as the
    natural key when importing or de-duplicating users.
    """

    name: str
    email: str
    role: UserRole = UserRole.STUDENT
    #: PBKDF2 hash from ``domain.auth.hash_password``, never the plaintext.
    #: Empty for a legacy/admin-provisioned row that has not set one yet —
    #: such an account simply cannot pass ``AuthenticationUseCase.login``.
    password_hash: str = ""
    id: int | None = None
    created_at: datetime | None = None

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.role.label})"


@dataclass(frozen=True)
class DatasetRecord:
    """Catalogued metadata for one dataset file.

    Populated from a loaded ``models.Dataset`` by
    ``ExperimentCatalogUseCase.register_dataset`` — the app never asks the user
    to type these in. ``source_path`` is the natural key: re-running against
    the same file updates the existing row instead of adding a duplicate.
    """

    name: str
    file_format: DatasetFormat
    num_samples: int
    num_features: int
    label_name: str = ""
    num_classes: int = 0
    feature_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    #: Compact summary of feature column dtypes, e.g. ``"float64 x24, int64 x2"``.
    dtype_summary: str = ""
    #: Feature cells that were missing or non-numeric at load time and got
    #: filled with 0.0 — see ``models.Dataset.missing_value_count``.
    missing_value_count: int = 0
    source_path: str = ""
    id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class ExperimentRecord:
    """One configured run: who ran what, against which domains, and how it ended.

    ``source_dataset_id`` / ``target_dataset_id`` link to the catalogued
    ``datasets`` rows; ``source_domain`` / ``target_domain`` keep the
    human-readable domain names denormalised so history stays readable even
    after a dataset row is deleted.

    The three hyper-parameters are nullable because not every method consumes
    them — ``None`` means "not applicable to this algorithm" and renders as a
    dash in the history table, which is different from a stored 0.
    """

    tl_algorithm: str
    classifier: str
    source_domain: str
    target_domain: str
    date: datetime
    status: ExperimentStatus = ExperimentStatus.PENDING
    normalization_type: str = "None"
    standardization_type: str = "None"
    latent_feature: int | None = None
    regularization_value: float | None = None
    kernel_type: str | None = None
    user_id: int | None = None
    source_dataset_id: int | None = None
    target_dataset_id: int | None = None
    error_message: str | None = None
    id: int | None = None


@dataclass(frozen=True)
class EvaluationRecord:
    """Scores for one experiment. At most one row per experiment.

    ``total_samples`` / ``correct_predictions`` / ``wrong_predictions`` go
    beyond the four headline scores so a stored run round-trips back into the
    UI's counts caption without re-running the experiment.
    """

    experiment_id: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int = 0
    correct_predictions: int = 0
    wrong_predictions: int = 0
    #: Everything below is nullable and stays that way: rows written before
    #: the results-comparison module existed have no such measurement, and
    #: the charts must show a gap rather than invent a zero for them.
    auc: float | None = None
    mmd_before: float | None = None
    mmd_after: float | None = None
    alignment_status: str | None = None
    execution_seconds: float | None = None
    adaptation_seconds: float | None = None
    baseline_accuracy: float | None = None
    baseline_f1: float | None = None
    overlap_before: float | None = None
    overlap_after: float | None = None
    id: int | None = None

    @property
    def mmd_change_percent(self) -> float | None:
        """``((before - after) / before) * 100``; None if either is missing."""
        if self.mmd_before is None or self.mmd_after is None:
            return None
        if abs(self.mmd_before) < 1e-12:
            return 0.0
        return (self.mmd_before - self.mmd_after) / self.mmd_before * 100.0

    @property
    def overlap_change_points(self) -> float | None:
        """After minus before, in percentage *points*; None if either is missing.

        Points rather than a ratio: overlap is already a percentage, so
        40% -> 60% is "+20 points". See ``domain/overlap.py``.
        """
        if self.overlap_before is None or self.overlap_after is None:
            return None
        return self.overlap_after - self.overlap_before

    @property
    def accuracy_delta(self) -> float | None:
        """Adapted minus baseline accuracy; None without a stored baseline."""
        if self.baseline_accuracy is None:
            return None
        return self.accuracy - self.baseline_accuracy

    @property
    def f1_delta(self) -> float | None:
        if self.baseline_f1 is None:
            return None
        return self.f1_score - self.baseline_f1


@dataclass(frozen=True)
class ExperimentHistoryEntry:
    """An experiment joined with its owner and its scores, for the history view.

    A read model rather than a table: the history table needs the user's name
    and the accuracy in one row, and issuing a query per row would be wasteful.
    ``user`` and ``evaluation`` are ``None`` when the experiment has no owner
    (the user was deleted) or produced no metrics (unlabelled data, or it
    failed).
    """

    experiment: ExperimentRecord
    user: User | None = None
    evaluation: EvaluationRecord | None = None
