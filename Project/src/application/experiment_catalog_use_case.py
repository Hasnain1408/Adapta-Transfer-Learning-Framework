"""Recording and querying experiments.

Sits alongside ``RunTransferLearningUseCase``: that one *executes* an
experiment, this one *remembers* it. Keeping them separate means a failure to
reach the database can never prevent a run from completing — the UI records
opportunistically after the fact.

Everything the presentation layer needs from persistence goes through this
class, so no widget imports a repository or touches SQL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from domain.entities import (
    DatasetFormat,
    DatasetRecord,
    EvaluationRecord,
    ExperimentHistoryEntry,
    ExperimentRecord,
    ExperimentStatus,
    User,
    UserRole,
)
from domain.models import Dataset, TransferLearningRequest, TransferLearningResult
from domain.repositories import (
    DatasetRepository,
    EvaluationRepository,
    ExperimentRepository,
    UserRepository,
)

#: Hyper-parameters are stored as NULL when the chosen algorithm ignores them,
#: which the UI renders as "—". Storing 0.0 instead would be a lie: it would
#: read as "this run used a regularisation of zero".
_UNUSED = None


class ExperimentCatalogUseCase:
    """Persist experiment runs and expose the catalogue to the UI."""

    def __init__(
        self,
        users: UserRepository,
        datasets: DatasetRepository,
        experiments: ExperimentRepository,
        evaluations: EvaluationRepository,
    ) -> None:
        self._users = users
        self._datasets = datasets
        self._experiments = experiments
        self._evaluations = evaluations

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def create_user(
        self, name: str, email: str, role: str | UserRole, password_hash: str = ""
    ) -> User:
        return self._users.add(
            User(
                name=name.strip(),
                email=email.strip(),
                role=UserRole.from_value(str(role)),
                password_hash=password_hash,
            )
        )

    def update_user(self, user: User) -> User:
        return self._users.update(user)

    def delete_user(self, user_id: int) -> bool:
        return self._users.delete(user_id)

    def get_user(self, user_id: int) -> User | None:
        return self._users.get(user_id)

    def list_users(self) -> list[User]:
        return self._users.list_all()

    def ensure_default_user(self) -> User:
        """Return some usable user, creating a starter account if the table is empty.

        A brand-new database would otherwise leave every experiment
        unattributed, and force the user through a dialog before their first
        run. The account is a normal row and can be renamed or deleted.
        """
        existing = self._users.list_all()
        if existing:
            return existing[0]
        return self._users.add(
            User(name="Default User", email="default@local", role=UserRole.RESEARCHER)
        )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def register_dataset(self, path: str, dataset: Dataset) -> DatasetRecord | None:
        """Catalogue the file at *path* using the statistics of the loaded *dataset*.

        Returns ``None`` for a path whose extension the catalogue does not
        model, rather than raising — the run itself already succeeded, so a
        bookkeeping gap must not surface as an error.
        """
        try:
            file_format = DatasetFormat.from_path(path)
        except ValueError:
            return None

        class_names = _class_names(dataset.labels)
        record = DatasetRecord(
            name=Path(path).name,
            file_format=file_format,
            num_samples=int(dataset.features.shape[0]),
            num_features=int(dataset.features.shape[1]),
            feature_names=[str(name) for name in dataset.feature_names],
            label_name=str(dataset.label_name or ""),
            num_classes=len(class_names),
            class_names=class_names,
            source_path=str(path),
        )
        return self._datasets.upsert_by_path(record)

    def list_datasets(self) -> list[DatasetRecord]:
        return self._datasets.list_all()

    def delete_dataset(self, dataset_id: int) -> bool:
        return self._datasets.delete(dataset_id)

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------
    def record_success(
        self,
        request: TransferLearningRequest,
        result: TransferLearningResult,
        user_id: int | None,
        supported_parameters: frozenset[str] | None = None,
    ) -> ExperimentRecord | None:
        """Store a completed run: both datasets, the experiment, and its scores.

        *supported_parameters* comes from
        ``RunTransferLearningUseCase.supported_parameters`` and decides which
        hyper-parameters are meaningful for the chosen algorithm; the rest are
        stored as NULL. Passing ``None`` records all three.
        """
        if user_id is None:
            return None

        source_record = self.register_dataset(request.source_path, result.source_dataset)
        target_record = self.register_dataset(request.target_path, result.target_dataset)

        experiment = self._experiments.add(
            self._build_record(
                request,
                user_id=user_id,
                status=ExperimentStatus.COMPLETED,
                source_dataset_id=source_record.id if source_record else None,
                target_dataset_id=target_record.id if target_record else None,
                supported_parameters=supported_parameters,
            )
        )

        metrics = result.metrics
        if metrics is not None and experiment.id is not None:
            self._evaluations.add(
                EvaluationRecord(
                    experiment_id=experiment.id,
                    accuracy=float(metrics.accuracy),
                    precision=float(metrics.precision),
                    recall=float(metrics.recall),
                    f1_score=float(metrics.f1_score),
                    total_samples=int(metrics.total_samples),
                    correct_predictions=int(metrics.correct_predictions),
                    wrong_predictions=int(metrics.wrong_predictions),
                )
            )
        return experiment

    def record_failure(
        self,
        request: TransferLearningRequest,
        user_id: int | None,
        error_message: str,
        supported_parameters: frozenset[str] | None = None,
    ) -> ExperimentRecord | None:
        """Store a run that raised, so failed configurations are not lost.

        No datasets are catalogued: the failure may well have been the loader
        refusing the file, in which case there are no statistics to record.
        """
        if user_id is None:
            return None

        return self._experiments.add(
            self._build_record(
                request,
                user_id=user_id,
                status=ExperimentStatus.FAILED,
                error_message=error_message,
                supported_parameters=supported_parameters,
            )
        )

    def _build_record(
        self,
        request: TransferLearningRequest,
        *,
        user_id: int | None,
        status: ExperimentStatus,
        source_dataset_id: int | None = None,
        target_dataset_id: int | None = None,
        error_message: str | None = None,
        supported_parameters: frozenset[str] | None = None,
    ) -> ExperimentRecord:
        supported = supported_parameters
        uses = (lambda field: True) if supported is None else (lambda field: field in supported)

        return ExperimentRecord(
            user_id=user_id,
            source_dataset_id=source_dataset_id,
            target_dataset_id=target_dataset_id,
            source_domain=Path(request.source_path).stem or request.source_path,
            target_domain=Path(request.target_path).stem or request.target_path,
            tl_algorithm=request.method_name,
            classifier=request.classifier,
            normalization_type=request.normalization,
            standardization_type=request.standardization,
            # Timezone-aware: the column is TIMESTAMPTZ, and a naive value
            # would be silently interpreted as the server's zone.
            date=datetime.now(timezone.utc).astimezone(),
            status=status,
            latent_feature=int(request.latent_dim) if uses("latent_dim") else _UNUSED,
            regularization_value=(
                float(request.regularization) if uses("regularization") else _UNUSED
            ),
            kernel_type=request.kernel_type if uses("kernel_type") else _UNUSED,
            error_message=error_message,
        )

    def list_experiments(self, user_id: int | None = None) -> list[ExperimentRecord]:
        return self._experiments.list_all(user_id)

    def history(self, user_id: int | None = None) -> list[ExperimentHistoryEntry]:
        return self._experiments.history(user_id)

    def get_experiment(self, experiment_id: int) -> ExperimentRecord | None:
        return self._experiments.get(experiment_id)

    def delete_experiment(self, experiment_id: int) -> bool:
        return self._experiments.delete(experiment_id)

    def get_evaluation(self, experiment_id: int) -> EvaluationRecord | None:
        return self._evaluations.get_for_experiment(experiment_id)

    # ------------------------------------------------------------------
    def statistics(self) -> dict[str, int]:
        """Row counts per table, for the status bar and the About box."""
        return {
            "users": _safe_count(self._users),
            "datasets": _safe_count(self._datasets),
            "experiments": _safe_count(self._experiments),
            "evaluations": _safe_count(self._evaluations),
        }


def _class_names(labels: np.ndarray | None) -> list[str]:
    """Distinct label values, sorted, as strings.

    ``np.unique`` on a mixed-type object array raises, so the fallback keeps
    catalogue registration working for oddly typed label columns instead of
    failing the whole recording step.
    """
    if labels is None:
        return []
    try:
        return [str(value) for value in np.unique(labels)]
    except (TypeError, ValueError):
        return sorted({str(value) for value in labels})


def _safe_count(repository: object) -> int:
    """Read ``repository.count()`` when it exists, else 0.

    ``count`` is an implementation convenience on the Postgres repositories
    rather than part of the domain protocols, so this stays tolerant of an
    implementation that omits it.
    """
    counter = getattr(repository, "count", None)
    if callable(counter):
        try:
            return int(counter())
        except Exception:
            return 0
    return 0
