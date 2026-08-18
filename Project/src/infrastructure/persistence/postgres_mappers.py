"""Row-to-entity mapping shared by the ``Postgres*Repository`` classes.

Row-to-entity mapping happens here and nowhere else: everything above this
module deals in the dataclasses from ``domain.entities``.
"""

from __future__ import annotations

from typing import Any

from domain.entities import (
    DatasetFormat,
    DatasetRecord,
    EvaluationRecord,
    ExperimentRecord,
    ExperimentStatus,
    User,
    UserRole,
)

from .database import PSYCOPG2_AVAILABLE

if PSYCOPG2_AVAILABLE:  # pragma: no cover - import-time branch
    from psycopg2 import errors as pg_errors
else:  # pragma: no cover
    pg_errors = None  # type: ignore[assignment]


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
        # .get(), not "or": these are legitimately NULL on older rows, and
        # "or" would turn a real stored 0.0 into None.
        auc=row.get("auc"),
        overlap_before=row.get("overlap_before"),
        overlap_after=row.get("overlap_after"),
        mmd_before=row.get("mmd_before"),
        mmd_after=row.get("mmd_after"),
        alignment_status=row.get("alignment_status"),
        execution_seconds=row.get("execution_seconds"),
        adaptation_seconds=row.get("adaptation_seconds"),
        baseline_accuracy=row.get("baseline_accuracy"),
        baseline_f1=row.get("baseline_f1"),
    )


def _is_unique_violation(exc: Exception) -> bool:
    """Whether *exc* is a duplicate-key error rather than any other failure."""
    if pg_errors is None:
        return False
    return isinstance(exc, pg_errors.UniqueViolation)


__all__ = [
    "_as_list",
    "_is_unique_violation",
    "_to_dataset",
    "_to_evaluation",
    "_to_experiment",
    "_to_user",
]
