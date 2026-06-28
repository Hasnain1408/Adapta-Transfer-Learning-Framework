from __future__ import annotations

import argparse
import os
import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from infrastructure.data.csv_dataset_loader import CSVDatasetLoader
from infrastructure.methods.tca_method import TCAMethod
from presentation.qt.main_window import MainWindow


def build_window() -> MainWindow:
    loader = CSVDatasetLoader()
    methods = [TCAMethod(latent_dim=5, regularization=1.0)]
    use_case = RunTransferLearningUseCase(loader=loader, methods=methods)
    return MainWindow(use_case)


def main() -> int:
    # cofiguration code
    parser = argparse.ArgumentParser(description="Transfer Learning Desktop App")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Start UI in offscreen mode and auto-exit to verify startup.",
    )
    args = parser.parse_args()

    if args.self_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# /cofiguration code

    app = QApplication(sys.argv)
    window = build_window()
    window.show()
    
 # cofiguration code
    if args.self_test:
        QTimer.singleShot(400, app.quit)
 #  /cofiguration code

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
