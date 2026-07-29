"""End-to-end check of the PostgreSQL layer against the configured database.

Run from ``Project/`` once ``DATABASE_URL`` is set (env var or ``.env``)::

    source venv/bin/activate
    python3 verify_database.py

Exercises every table and every repository method, then deletes everything it
created. It writes to real tables, so point it at a scratch Neon branch rather
than a database whose contents you care about — it is only additive, but the
cleanup step does issue DELETEs.

Pass ``--keep`` to leave the sample rows behind so you can inspect them in the
Neon console or the app's History tab.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Same trick as verify_jda.py: put src/ on the path so the domain/application
# imports resolve when running from the Project/ root.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from application.experiment_catalog_use_case import ExperimentCatalogUseCase  # noqa: E402
from domain.entities import (  # noqa: E402
    DatasetFormat,
    DatasetRecord,
    EvaluationRecord,
    ExperimentRecord,
    ExperimentStatus,
    User,
    UserRole,
)
from infrastructure.persistence import (  # noqa: E402
    Database,
    DatabaseConfigError,
    DatabaseUnavailable,
    PostgresDatasetRepository,
    PostgresEvaluationRepository,
    PostgresExperimentRepository,
    PostgresUserRepository,
)

PASS = "  \033[32mPASS\033[0m"
FAIL = "  \033[31mFAIL\033[0m"

_failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    if condition:
        print(f"{PASS}  {label}")
    else:
        _failures += 1
        print(f"{FAIL}  {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true", help="Leave the sample rows in the database."
    )
    args = parser.parse_args()

    # ── Connect ──────────────────────────────────────────────────────────
    try:
        database = Database()
        database.connect()
    except (DatabaseConfigError, DatabaseUnavailable) as exc:
        print(f"\nCould not connect:\n\n{exc}\n")
        return 2

    print(f"\nConnected to {database.dsn}\n")

    users = PostgresUserRepository(database)
    datasets = PostgresDatasetRepository(database)
    experiments = PostgresExperimentRepository(database)
    evaluations = PostgresEvaluationRepository(database)
    catalog = ExperimentCatalogUseCase(users, datasets, experiments, evaluations)

    created_user_ids: list[int] = []
    created_dataset_ids: list[int] = []
    created_experiment_ids: list[int] = []

    try:
        # ── Schema ───────────────────────────────────────────────────────
        print("Schema")
        tables = database.table_names()
        for expected in ("users", "datasets", "experiments", "evaluations"):
            check(f"table {expected!r} exists", expected in tables, f"found: {tables}")

        # ── Users ────────────────────────────────────────────────────────
        print("\nUsers")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        user = users.add(
            User(name="Verify Bot", email=f"verify+{stamp}@example.com", role=UserRole.RESEARCHER)
        )
        created_user_ids.append(user.id)
        check("add() returns an id", user.id is not None)
        check("add() returns the stored role", user.role is UserRole.RESEARCHER)
        check("add() populates created_at", isinstance(user.created_at, datetime))

        fetched = users.get(user.id)
        check("get() round-trips", fetched is not None and fetched.email == user.email)

        by_email = users.get_by_email(user.email.upper())
        check(
            "get_by_email() is case-insensitive",
            by_email is not None and by_email.id == user.id,
        )

        renamed = users.update(User(id=user.id, name="Verify Bot II", email=user.email, role=UserRole.ADMIN))
        check("update() changes name and role", renamed.name == "Verify Bot II" and renamed.role is UserRole.ADMIN)

        duplicate_rejected = False
        try:
            users.add(User(name="Clash", email=user.email, role=UserRole.STUDENT))
        except ValueError:
            duplicate_rejected = True
        check("duplicate email rejected", duplicate_rejected)

        check("list_all() includes the new user", any(u.id == user.id for u in users.list_all()))

        # ── Datasets ─────────────────────────────────────────────────────
        print("\nDatasets")
        record = DatasetRecord(
            name="verify_source.csv",
            file_format=DatasetFormat.CSV,
            num_samples=120,
            num_features=8,
            feature_names=[f"f{i}" for i in range(8)],
            label_name="label",
            num_classes=2,
            class_names=["0", "1"],
            source_path=f"/tmp/verify_source_{stamp}.csv",
        )
        source = datasets.upsert_by_path(record)
        created_dataset_ids.append(source.id)
        check("upsert_by_path() inserts", source.id is not None)
        check("TEXT[] feature_names round-trips", source.feature_names == record.feature_names)
        check("TEXT[] class_names round-trips", source.class_names == ["0", "1"])

        # Re-upserting the same path must update, not duplicate.
        updated = datasets.upsert_by_path(
            DatasetRecord(
                name="verify_source.csv",
                file_format=DatasetFormat.CSV,
                num_samples=999,
                num_features=8,
                feature_names=record.feature_names,
                label_name="label",
                num_classes=2,
                class_names=["0", "1"],
                source_path=record.source_path,
            )
        )
        check("upsert_by_path() updates in place", updated.id == source.id)
        check("upsert_by_path() refreshes stats", updated.num_samples == 999)

        target = datasets.upsert_by_path(
            DatasetRecord(
                name="verify_target.csv",
                file_format=DatasetFormat.CSV,
                num_samples=80,
                num_features=8,
                label_name="label",
                num_classes=2,
                class_names=["0", "1"],
                source_path=f"/tmp/verify_target_{stamp}.csv",
            )
        )
        created_dataset_ids.append(target.id)
        check("get_by_path() finds the row", datasets.get_by_path(target.source_path) is not None)

        # ── Experiments ──────────────────────────────────────────────────
        print("\nExperiments")
        experiment = experiments.add(
            ExperimentRecord(
                user_id=user.id,
                source_dataset_id=source.id,
                target_dataset_id=target.id,
                source_domain="verify_source",
                target_domain="verify_target",
                tl_algorithm="TCA",
                classifier="k-NN",
                normalization_type="Min-Max Normalization",
                standardization_type="None",
                date=datetime.now(timezone.utc).astimezone(),
                status=ExperimentStatus.COMPLETED,
                latent_feature=10,
                regularization_value=1.0,
                kernel_type="linear",
            )
        )
        created_experiment_ids.append(experiment.id)
        check("add() returns an id", experiment.id is not None)
        check("status round-trips as an enum", experiment.status is ExperimentStatus.COMPLETED)
        check("date round-trips as a datetime", isinstance(experiment.date, datetime))

        experiments.set_status(experiment.id, ExperimentStatus.FAILED.value, "verification")
        reread = experiments.get(experiment.id)
        check("set_status() updates status", reread.status is ExperimentStatus.FAILED)
        check("set_status() stores the message", reread.error_message == "verification")
        experiments.set_status(experiment.id, ExperimentStatus.COMPLETED.value, None)

        # NULL hyper-parameters must stay NULL, not become 0.
        minimal = experiments.add(
            ExperimentRecord(
                user_id=user.id,
                source_domain="a",
                target_domain="b",
                tl_algorithm="CORAL",
                classifier="SVM",
                date=datetime.now(timezone.utc).astimezone(),
                status=ExperimentStatus.COMPLETED,
                latent_feature=None,
                regularization_value=None,
                kernel_type=None,
            )
        )
        created_experiment_ids.append(minimal.id)
        check(
            "unused hyper-parameters stay NULL",
            minimal.latent_feature is None
            and minimal.regularization_value is None
            and minimal.kernel_type is None,
        )

        # ── Evaluations ──────────────────────────────────────────────────
        print("\nEvaluations")
        evaluation = evaluations.add(
            EvaluationRecord(
                experiment_id=experiment.id,
                accuracy=0.9125,
                precision=0.88,
                recall=0.91,
                f1_score=0.895,
                total_samples=80,
                correct_predictions=73,
                wrong_predictions=7,
            )
        )
        check("add() returns an id", evaluation.id is not None)
        check("scores round-trip", abs(evaluation.accuracy - 0.9125) < 1e-9)

        # The 1:1 constraint means a second add must overwrite, not raise.
        replaced = evaluations.add(
            EvaluationRecord(
                experiment_id=experiment.id,
                accuracy=0.5,
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                total_samples=80,
                correct_predictions=40,
                wrong_predictions=40,
            )
        )
        check("re-adding overwrites (1:1 upheld)", replaced.id == evaluation.id)
        check("overwrite changed the score", abs(replaced.accuracy - 0.5) < 1e-9)

        out_of_range_rejected = False
        try:
            evaluations.add(
                EvaluationRecord(
                    experiment_id=minimal.id, accuracy=1.7, precision=0, recall=0, f1_score=0
                )
            )
        except Exception:
            out_of_range_rejected = True
        check("CHECK constraint rejects accuracy > 1", out_of_range_rejected)

        # ── History join ─────────────────────────────────────────────────
        print("\nHistory read model")
        entries = experiments.history(user.id)
        check("history() returns the user's rows", len(entries) >= 2)
        joined = next((e for e in entries if e.experiment.id == experiment.id), None)
        check("history() joins the user", joined is not None and joined.user is not None)
        check("history() joins the evaluation", joined is not None and joined.evaluation is not None)
        unscored = next((e for e in entries if e.experiment.id == minimal.id), None)
        check(
            "history() tolerates a missing evaluation",
            unscored is not None and unscored.evaluation is None,
        )

        # ── Referential integrity ────────────────────────────────────────
        print("\nReferential integrity")
        datasets.delete(source.id)
        created_dataset_ids.remove(source.id)
        after = experiments.get(experiment.id)
        check("deleting a dataset SET NULLs the link", after.source_dataset_id is None)
        check("experiment survives dataset deletion", after is not None)
        check("denormalised domain name survives", after.source_domain == "verify_source")

        evaluation_id = evaluation.id
        experiments.delete(minimal.id)
        created_experiment_ids.remove(minimal.id)
        check("delete() removes the experiment", experiments.get(minimal.id) is None)

        experiments.delete(experiment.id)
        created_experiment_ids.remove(experiment.id)
        check(
            "deleting an experiment CASCADEs its evaluation",
            evaluations.get(evaluation_id) is None,
        )

        # ── Use-case surface ─────────────────────────────────────────────
        print("\nCatalog use case")
        stats = catalog.statistics()
        check(
            "statistics() reports all four tables",
            set(stats) == {"users", "datasets", "experiments", "evaluations"},
            str(stats),
        )
        default_user = catalog.ensure_default_user()
        check("ensure_default_user() returns a user", default_user.id is not None)

    finally:
        if not args.keep:
            print("\nCleaning up")
            for experiment_id in created_experiment_ids:
                experiments.delete(experiment_id)
            for dataset_id in created_dataset_ids:
                datasets.delete(dataset_id)
            for user_id in created_user_ids:
                users.delete(user_id)
            print("  removed the rows this script created")
        else:
            print("\n--keep: sample rows left in place")
        database.close()

    print()
    if _failures:
        print(f"{_failures} check(s) FAILED\n")
        return 1
    print("All checks passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
