from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.models import TransferLearningRequest


class TransferLearningWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        use_case: RunTransferLearningUseCase,
        request: TransferLearningRequest,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._request = request

    def run(self) -> None:
        try:
            result = self._use_case.execute(self._request)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
