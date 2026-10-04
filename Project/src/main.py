from __future__ import annotations

import argparse
import os
import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from application.authentication_use_case import AuthenticationUseCase
from application.experiment_catalog_use_case import ExperimentCatalogUseCase
from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from infrastructure.data.arff_dataset_loader import ARFFDatasetLoader
from infrastructure.data.csv_dataset_loader import CSVDatasetLoader
from infrastructure.data.dataset_loader import DatasetLoader
from infrastructure.data.mat_dataset_loader import MATDatasetLoader
from infrastructure.data.xlsx_dataset_loader import XLSXDatasetLoader
from infrastructure.methods.tca_method import TCAMethod
from infrastructure.methods.jda_method import JDAMethod
from infrastructure.methods.coral_method import CORALMethod
from infrastructure.methods.bda_method import BDAMethod
from infrastructure.methods.ocmtl_method import OCMTLMethod
from infrastructure.methods.dali_method import DALIMethod
from infrastructure.methods.gfk_method import GFKMethod
from infrastructure.methods.jgsa_method import JGSAMethod
from infrastructure.methods.jcsl_method import JCSLMethod
from infrastructure.methods.meda_method import MEDAMethod
from infrastructure.methods.proposed_method import ProposedMethod
from infrastructure.persistence import (
    Database,
    DatabaseConfigError,
    DatabaseUnavailable,
    PostgresDatasetRepository,
    PostgresEvaluationRepository,
    PostgresExperimentRepository,
    PostgresUserRepository,
)
from presentation.qt.main_window import MainWindow
from presentation.qt.style import DARK_PALETTE, build_style_sheet

_APP_ICON_PATH = os.path.join(
    os.path.dirname(__file__), "presentation", "qt", "assets", "app_icon.svg"
)


def build_catalog(
    dsn: str | None = None,
) -> tuple[ExperimentCatalogUseCase | None, AuthenticationUseCase | None, Database | None, str | None]:
    """Connect to PostgreSQL and wire up the catalogue and auth use cases.

    Returns ``(catalog, auth, database, error_message)``. On any failure the
    first three are ``None`` and the fourth explains why — the app then runs
    without persistence (and without login, since there is nowhere to check
    credentials against) rather than refusing to start, so a missing
    ``DATABASE_URL`` or a Neon outage costs you the history/account features,
    not the whole tool.
    """
    try:
        database = Database(dsn)
        # Opens the connection and applies the schema; both can fail, and
        # both should fail here rather than on the user's first run.
        database.connect()
    except (DatabaseConfigError, DatabaseUnavailable) as exc:
        return None, None, None, str(exc)
    except Exception as exc:  # noqa: BLE001 - unexpected driver/network faults
        return None, None, None, f"Unexpected database error: {exc}"

    users = PostgresUserRepository(database)
    catalog = ExperimentCatalogUseCase(
        users=users,
        datasets=PostgresDatasetRepository(database),
        experiments=PostgresExperimentRepository(database),
        evaluations=PostgresEvaluationRepository(database),
    )
    auth = AuthenticationUseCase(users)
    return catalog, auth, database, None


def build_window(
    catalog: ExperimentCatalogUseCase | None = None,
    auth: AuthenticationUseCase | None = None,
    database_error: str | None = None,
) -> MainWindow:
    loader = DatasetLoader(
        [CSVDatasetLoader(), XLSXDatasetLoader(), MATDatasetLoader(), ARFFDatasetLoader()]
    )
    methods = [
        TCAMethod(latent_dim=5, regularization=1.0, kernel_type="linear"),
        JDAMethod(latent_dim=5, regularization=1.0, kernel_type="linear", iterations=10),
        CORALMethod(),
        BDAMethod(mode="W-BDA", mu=1.0, regularization=0.1, latent_dim=10, kernel_type="linear", iterations=10),
        OCMTLMethod(latent_dim=2, max_iterations=20),
        DALIMethod(latent_dim=10, regularization=1.0, beta=0.4, delta=0.2, epsilon=0.1, iterations=10),
        GFKMethod(latent_dim=10),
        JGSAMethod(latent_dim=10, regularization=1.0, beta=1.0, mu=1.0, iterations=10),
        JCSLMethod(latent_dim=10, regularization=0.01, subspace_weight=0.5, iterations=200),
        MEDAMethod(latent_dim=20, regularization=1.0, kernel_type="rbf", rho=1.0, eta=0.1, iterations=10),
        ProposedMethod(),
    ]
    use_case = RunTransferLearningUseCase(loader=loader, methods=methods)
    return MainWindow(
        use_case,
        catalog=catalog,
        auth=auth,
        database_error=database_error,
    )


def main() -> int:
    # cofiguration code
    parser = argparse.ArgumentParser(description="Transfer Learning Desktop App")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Start UI in offscreen mode and auto-exit to verify startup.",
    )
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="Skip the PostgreSQL connection and run without recording experiments.",
    )
    args = parser.parse_args()

    if args.self_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # /cofiguration code

    app = QApplication(sys.argv)
    app.setApplicationName("Adapta")
    app.setApplicationDisplayName("Adapta: Transfer Learning Studio")
    app.setOrganizationName("Adapta")
    app.setWindowIcon(QIcon(_APP_ICON_PATH))
    app.setStyleSheet(build_style_sheet(DARK_PALETTE))

    if args.no_database:
        catalog, auth, database, error = None, None, None, "Started with --no-database."
    else:
        catalog, auth, database, error = build_catalog()
        if error:
            # Visible on the console; the window also reports it in the status
            # bar and under Database ▸ Connection Info.
            print(f"[database] {error}", file=sys.stderr)

    if auth is None and not args.self_test:
        print(
            "[auth] No database connection — starting without sign-in; "
            "experiments will not be recorded.",
            file=sys.stderr,
        )

    window = build_window(catalog=catalog, auth=auth, database_error=error)
    window.show()

 # cofiguration code
    if args.self_test:
        QTimer.singleShot(400, app.quit)
 #  /cofiguration code

    try:
        return app.exec_()
    finally:
        if database is not None:
            database.close()


if __name__ == "__main__":
    raise SystemExit(main())
