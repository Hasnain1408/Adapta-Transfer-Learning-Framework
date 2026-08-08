"""PostgreSQL implementations of the domain repository protocols.

Row-to-entity mapping happens here and nowhere else: everything above this
module deals in the dataclasses from ``domain.entities``.

Two conventions worth knowing when reading the SQL:

- Placeholders are ``%s`` (psycopg2's paramstyle), never f-strings. Values are
  always passed as parameters so nothing user-supplied is interpolated into a
  statement.
- Inserts and updates end with ``RETURNING *`` and rebuild the entity from
  what the database actually stored, so defaults filled in by the schema
  (``created_at``, ``date``, ``status``) come back populated rather than being
  guessed at on the Python side.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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

from .database import Database, PSYCOPG2_AVAILABLE

if PSYCOPG2_AVAILABLE:  # pragma: no cover - import-time branch
    from psycopg2 import errors as pg_errors
else:  # pragma: no cover
    pg_errors = None  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Row mappers
# ----------------------------------------------------------------------
def _as_list(value: Any) -> list[str]:
    """Normalise a ``TEXT[]`` column to a list of strings.

    psycopg2 already returns a Python list for array columns; this guards the
    ``NULL`` case and anything hand-inserted as a scalar.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _to_user(row: dict[str, Any]) -> User:
    return User(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        role=UserRole.from_value(row["role"]),
        password_hash=row.get("password_hash") or "",
        created_at=row.get("created_at"),
    )


def _to_dataset(row: dict[str, Any]) -> DatasetRecord:
    return DatasetRecord(
        id=row["id"],
        name=row["name"],
        file_format=DatasetFormat(row["file_format"]),
        num_samples=row["num_samples"],
        num_features=row["num_features"],
        feature_names=_as_list(row.get("feature_names")),
        label_name=row.get("label_name") or "",
        num_classes=row.get("num_classes") or 0,
        class_names=_as_list(row.get("class_names")),
        dtype_summary=row.get("dtype_summary") or "",
        missing_value_count=row.get("missing_value_count") or 0,
        source_path=row.get("source_path") or "",
        created_at=row.get("created_at"),
    )


def _to_experiment(row: dict[str, Any]) -> ExperimentRecord:
    return ExperimentRecord(
        id=row["id"],
        user_id=row.get("user_id"),
        source_dataset_id=row.get("source_dataset_id"),
        target_dataset_id=row.get("target_dataset_id"),
        source_domain=row.get("source_domain") or "",
        target_domain=row.get("target_domain") or "",
        tl_algorithm=row["tl_algorithm"],
        classifier=row.get("classifier") or "",
        normalization_type=row.get("normalization_type") or "None",
        standardization_type=row.get("standardization_type") or "None",
        date=row["date"],
        status=ExperimentStatus.from_value(row["status"]),
        latent_feature=row.get("latent_feature"),
        regularization_value=row.get("regularization_value"),
        kernel_type=row.get("kernel_type"),
        error_message=row.get("error_message"),
    )


def _to_evaluation(row: dict[str, Any]) -> EvaluationRecord:
    return EvaluationRecord(
        id=row["id"],
        experiment_id=row["experiment_id"],
        accuracy=row["accuracy"],
        precision=row["precision"],
        recall=row["recall"],
        f1_score=row["f1_score"],
        total_samples=row.get("total_samples") or 0,
        correct_predictions=row.get("correct_predictions") or 0,
        wrong_predictions=row.get("wrong_predictions") or 0,
    )


def _is_unique_violation(exc: Exception) -> bool:
    """Whether *exc* is a duplicate-key error rather than any other failure."""
    if pg_errors is None:
        return False
    return isinstance(exc, pg_errors.UniqueViolation)


# ----------------------------------------------------------------------
# Users
# ----------------------------------------------------------------------
class PostgresUserRepository:
    """CRUD for ``users``."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def add(self, user: User) -> User:
        try:
            row = self._db.execute_returning(
                """
                INSERT INTO users (name, email, password_hash, role)
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (
                    user.name.strip(),
                    user.email.strip(),
                    user.password_hash,
                    UserRole(user.role).value,
                ),
            )
        except Exception as exc:
            # The unique index is on lower(email), so this is the only way a
            # duplicate can arrive; translate it into a message the UI can show.
            if _is_unique_violation(exc):
                raise ValueError(
                    f"A user with the email {user.email!r} already exists."
                ) from exc
            raise
        assert row is not None  # RETURNING on a successful INSERT always yields a row
        return _to_user(row)

    def update(self, user: User) -> User:
        if user.id is None:
            raise ValueError("Cannot update a user that has no id.")
        try:
            row = self._db.execute_returning(
                """
                UPDATE users
                   SET name = %s, email = %s, password_hash = %s, role = %s
                 WHERE id = %s
                RETURNING *
                """,
                (
                    user.name.strip(),
                    user.email.strip(),
                    user.password_hash,
                    UserRole(user.role).value,
                    user.id,
                ),
            )
        except Exception as exc:
            if _is_unique_violation(exc):
                raise ValueError(
                    f"A user with the email {user.email!r} already exists."
                ) from exc
            raise
        if row is None:
            raise ValueError(f"No user with id {user.id}.")
        return _to_user(row)

    def delete(self, user_id: int) -> bool:
        # experiments.user_id is ON DELETE SET NULL, so the owner's history
        # survives this and simply becomes unattributed.
        return self._db.execute("DELETE FROM users WHERE id = %s", (user_id,)) > 0

    def get(self, user_id: int) -> User | None:
        row = self._db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
        return _to_user(row) if row else None

    def get_by_email(self, email: str) -> User | None:
        row = self._db.fetch_one(
            "SELECT * FROM users WHERE lower(email) = lower(%s)", (email.strip(),)
        )
        return _to_user(row) if row else None

    def list_all(self) -> list[User]:
        rows = self._db.fetch_all("SELECT * FROM users ORDER BY name, id")
        return [_to_user(row) for row in rows]

    def count(self) -> int:
        row = self._db.fetch_one("SELECT count(*) AS total FROM users")
        return int(row["total"]) if row else 0


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------
class PostgresDatasetRepository:
    """CRUD for ``datasets``, the catalogue of files that have been used."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def add(self, dataset: DatasetRecord) -> DatasetRecord:
        try:
            row = self._db.execute_returning(
                """
                INSERT INTO datasets (
                    name, file_format, num_samples, num_features,
                    feature_names, label_name, num_classes, class_names,
                    dtype_summary, missing_value_count, source_path
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    dataset.name,
                    DatasetFormat(dataset.file_format).value,
                    dataset.num_samples,
                    dataset.num_features,
                    list(dataset.feature_names),
                    dataset.label_name,
                    dataset.num_classes,
                    list(dataset.class_names),
                    dataset.dtype_summary,
                    dataset.missing_value_count,
                    dataset.source_path,
                ),
            )
        except Exception as exc:
            if _is_unique_violation(exc):
                raise ValueError(
                    f"A dataset is already registered for {dataset.source_path!r}."
                ) from exc
            raise
        assert row is not None
        return _to_dataset(row)

    def upsert_by_path(self, dataset: DatasetRecord) -> DatasetRecord:
        """Insert, or refresh the row that already claims this ``source_path``.

        Uses ``ON CONFLICT`` so the read and the write are one atomic
        statement — checking for an existing row first and then inserting
        would race against a concurrent run of the same file.
        """
        row = self._db.execute_returning(
            """
            INSERT INTO datasets (
                name, file_format, num_samples, num_features,
                feature_names, label_name, num_classes, class_names,
                dtype_summary, missing_value_count, source_path
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_path) DO UPDATE SET
                name                 = EXCLUDED.name,
                file_format          = EXCLUDED.file_format,
                num_samples          = EXCLUDED.num_samples,
                num_features         = EXCLUDED.num_features,
                feature_names        = EXCLUDED.feature_names,
                label_name           = EXCLUDED.label_name,
                num_classes          = EXCLUDED.num_classes,
                class_names          = EXCLUDED.class_names,
                dtype_summary        = EXCLUDED.dtype_summary,
                missing_value_count  = EXCLUDED.missing_value_count
            RETURNING *
            """,
            (
                dataset.name,
                DatasetFormat(dataset.file_format).value,
                dataset.num_samples,
                dataset.num_features,
                list(dataset.feature_names),
                dataset.label_name,
                dataset.num_classes,
                list(dataset.class_names),
                dataset.dtype_summary,
                dataset.missing_value_count,
                dataset.source_path,
            ),
        )
        assert row is not None
        return _to_dataset(row)

    def update(self, dataset: DatasetRecord) -> DatasetRecord:
        if dataset.id is None:
            raise ValueError("Cannot update a dataset that has no id.")
        row = self._db.execute_returning(
            """
            UPDATE datasets
               SET name = %s, file_format = %s, num_samples = %s, num_features = %s,
                   feature_names = %s, label_name = %s, num_classes = %s,
                   class_names = %s, dtype_summary = %s, missing_value_count = %s,
                   source_path = %s
             WHERE id = %s
            RETURNING *
            """,
            (
                dataset.name,
                DatasetFormat(dataset.file_format).value,
                dataset.num_samples,
                dataset.num_features,
                list(dataset.feature_names),
                dataset.label_name,
                dataset.num_classes,
                list(dataset.class_names),
                dataset.dtype_summary,
                dataset.missing_value_count,
                dataset.source_path,
                dataset.id,
            ),
        )
        if row is None:
            raise ValueError(f"No dataset with id {dataset.id}.")
        return _to_dataset(row)

    def delete(self, dataset_id: int) -> bool:
        # experiments.*_dataset_id is ON DELETE SET NULL: the experiment keeps
        # its denormalised domain names and only loses the catalogue link.
        return self._db.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,)) > 0

    def get(self, dataset_id: int) -> DatasetRecord | None:
        row = self._db.fetch_one("SELECT * FROM datasets WHERE id = %s", (dataset_id,))
        return _to_dataset(row) if row else None

    def get_by_path(self, source_path: str) -> DatasetRecord | None:
        row = self._db.fetch_one(
            "SELECT * FROM datasets WHERE source_path = %s", (source_path,)
        )
        return _to_dataset(row) if row else None

    def list_all(self) -> list[DatasetRecord]:
        rows = self._db.fetch_all("SELECT * FROM datasets ORDER BY created_at DESC, id DESC")
        return [_to_dataset(row) for row in rows]

    def count(self) -> int:
        row = self._db.fetch_one("SELECT count(*) AS total FROM datasets")
        return int(row["total"]) if row else 0


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
_EXPERIMENT_COLUMNS = """
    user_id, source_dataset_id, target_dataset_id, source_domain, target_domain,
    tl_algorithm, classifier, normalization_type, standardization_type,
    date, status, latent_feature, regularization_value, kernel_type, error_message
"""


class PostgresExperimentRepository:
    """CRUD for ``experiments`` plus the joined history read model."""

    def __init__(self, database: Database) -> None:
        self._db = database

    @staticmethod
    def _values(experiment: ExperimentRecord) -> tuple[Any, ...]:
        return (
            experiment.user_id,
            experiment.source_dataset_id,
            experiment.target_dataset_id,
            experiment.source_domain,
            experiment.target_domain,
            experiment.tl_algorithm,
            experiment.classifier,
            experiment.normalization_type,
            experiment.standardization_type,
            experiment.date or datetime.now().astimezone(),
            ExperimentStatus(experiment.status).value,
            experiment.latent_feature,
            experiment.regularization_value,
            experiment.kernel_type,
            experiment.error_message,
        )

    def add(self, experiment: ExperimentRecord) -> ExperimentRecord:
        row = self._db.execute_returning(
            f"""
            INSERT INTO experiments ({_EXPERIMENT_COLUMNS})
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            self._values(experiment),
        )
        assert row is not None
        return _to_experiment(row)

    def update(self, experiment: ExperimentRecord) -> ExperimentRecord:
        if experiment.id is None:
            raise ValueError("Cannot update an experiment that has no id.")
        row = self._db.execute_returning(
            """
            UPDATE experiments
               SET user_id = %s, source_dataset_id = %s, target_dataset_id = %s,
                   source_domain = %s, target_domain = %s, tl_algorithm = %s,
                   classifier = %s, normalization_type = %s, standardization_type = %s,
                   date = %s, status = %s, latent_feature = %s,
                   regularization_value = %s, kernel_type = %s, error_message = %s
             WHERE id = %s
            RETURNING *
            """,
            (*self._values(experiment), experiment.id),
        )
        if row is None:
            raise ValueError(f"No experiment with id {experiment.id}.")
        return _to_experiment(row)

    def set_status(
        self, experiment_id: int, status: str, error_message: str | None = None
    ) -> None:
        """Move an experiment to a new state without rewriting the whole row."""
        self._db.execute(
            "UPDATE experiments SET status = %s, error_message = %s WHERE id = %s",
            (ExperimentStatus(status).value, error_message, experiment_id),
        )

    def delete(self, experiment_id: int) -> bool:
        # evaluations.experiment_id is ON DELETE CASCADE, so the scores go too.
        return (
            self._db.execute("DELETE FROM experiments WHERE id = %s", (experiment_id,)) > 0
        )

    def get(self, experiment_id: int) -> ExperimentRecord | None:
        row = self._db.fetch_one(
            "SELECT * FROM experiments WHERE id = %s", (experiment_id,)
        )
        return _to_experiment(row) if row else None

    def list_all(self, user_id: int | None = None) -> list[ExperimentRecord]:
        if user_id is None:
            rows = self._db.fetch_all(
                "SELECT * FROM experiments ORDER BY date DESC, id DESC"
            )
        else:
            rows = self._db.fetch_all(
                "SELECT * FROM experiments WHERE user_id = %s ORDER BY date DESC, id DESC",
                (user_id,),
            )
        return [_to_experiment(row) for row in rows]

    def history(self, user_id: int | None = None) -> list[ExperimentHistoryEntry]:
        """Experiments with their owner and scores, newest first.

        One query with two LEFT JOINs rather than N+1 lookups while filling the
        table. Columns are aliased because ``users.name`` and ``datasets.name``
        would otherwise collide with each other in the flat result row.
        """
        sql = """
            SELECT e.*,
                   u.id         AS u_id,
                   u.name       AS u_name,
                   u.email      AS u_email,
                   u.role       AS u_role,
                   u.created_at AS u_created_at,
                   v.id                  AS v_id,
                   v.accuracy            AS v_accuracy,
                   v."precision"         AS v_precision,
                   v.recall              AS v_recall,
                   v.f1_score            AS v_f1_score,
                   v.total_samples       AS v_total_samples,
                   v.correct_predictions AS v_correct_predictions,
                   v.wrong_predictions   AS v_wrong_predictions
              FROM experiments e
              LEFT JOIN users       u ON u.id = e.user_id
              LEFT JOIN evaluations v ON v.experiment_id = e.id
        """
        params: tuple[Any, ...] = ()
        if user_id is not None:
            sql += " WHERE e.user_id = %s"
            params = (user_id,)
        sql += " ORDER BY e.date DESC, e.id DESC"

        entries: list[ExperimentHistoryEntry] = []
        for row in self._db.fetch_all(sql, params):
            user = (
                _to_user(
                    {
                        "id": row["u_id"],
                        "name": row["u_name"],
                        "email": row["u_email"],
                        "role": row["u_role"],
                        "created_at": row["u_created_at"],
                    }
                )
                if row.get("u_id") is not None
                else None
            )
            evaluation = (
                _to_evaluation(
                    {
                        "id": row["v_id"],
                        "experiment_id": row["id"],
                        "accuracy": row["v_accuracy"],
                        "precision": row["v_precision"],
                        "recall": row["v_recall"],
                        "f1_score": row["v_f1_score"],
                        "total_samples": row["v_total_samples"],
                        "correct_predictions": row["v_correct_predictions"],
                        "wrong_predictions": row["v_wrong_predictions"],
                    }
                )
                if row.get("v_id") is not None
                else None
            )
            entries.append(
                ExperimentHistoryEntry(
                    experiment=_to_experiment(row), user=user, evaluation=evaluation
                )
            )
        return entries

    def count(self) -> int:
        row = self._db.fetch_one("SELECT count(*) AS total FROM experiments")
        return int(row["total"]) if row else 0


# ----------------------------------------------------------------------
# Evaluations
# ----------------------------------------------------------------------
class PostgresEvaluationRepository:
    """CRUD for ``evaluations`` — at most one row per experiment."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def add(self, evaluation: EvaluationRecord) -> EvaluationRecord:
        """Insert scores, replacing any existing row for the same experiment.

        ``ON CONFLICT (experiment_id)`` upholds the 1:1 relationship: re-running
        and re-saving an experiment overwrites its scores rather than failing
        on the unique constraint.
        """
        row = self._db.execute_returning(
            """
            INSERT INTO evaluations (
                experiment_id, accuracy, "precision", recall, f1_score,
                total_samples, correct_predictions, wrong_predictions
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id) DO UPDATE SET
                accuracy            = EXCLUDED.accuracy,
                "precision"         = EXCLUDED."precision",
                recall              = EXCLUDED.recall,
                f1_score            = EXCLUDED.f1_score,
                total_samples       = EXCLUDED.total_samples,
                correct_predictions = EXCLUDED.correct_predictions,
                wrong_predictions   = EXCLUDED.wrong_predictions
            RETURNING *
            """,
            (
                evaluation.experiment_id,
                evaluation.accuracy,
                evaluation.precision,
                evaluation.recall,
                evaluation.f1_score,
                evaluation.total_samples,
                evaluation.correct_predictions,
                evaluation.wrong_predictions,
            ),
        )
        assert row is not None
        return _to_evaluation(row)

    def get(self, evaluation_id: int) -> EvaluationRecord | None:
        row = self._db.fetch_one(
            "SELECT * FROM evaluations WHERE id = %s", (evaluation_id,)
        )
        return _to_evaluation(row) if row else None

    def get_for_experiment(self, experiment_id: int) -> EvaluationRecord | None:
        row = self._db.fetch_one(
            "SELECT * FROM evaluations WHERE experiment_id = %s", (experiment_id,)
        )
        return _to_evaluation(row) if row else None

    def delete(self, evaluation_id: int) -> bool:
        return (
            self._db.execute("DELETE FROM evaluations WHERE id = %s", (evaluation_id,)) > 0
        )

    def list_all(self) -> list[EvaluationRecord]:
        rows = self._db.fetch_all("SELECT * FROM evaluations ORDER BY id DESC")
        return [_to_evaluation(row) for row in rows]

    def count(self) -> int:
        row = self._db.fetch_one("SELECT count(*) AS total FROM evaluations")
        return int(row["total"]) if row else 0


__all__ = [
    "PostgresDatasetRepository",
    "PostgresEvaluationRepository",
    "PostgresExperimentRepository",
    "PostgresUserRepository",
]
