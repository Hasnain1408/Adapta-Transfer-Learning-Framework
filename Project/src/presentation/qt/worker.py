from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.models import BatchTransferLearningRequest, TransferLearningRequest
from domain.run_control import RunCancelled, RunControl


class TransferLearningWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        use_case: RunTransferLearningUseCase,
        request: TransferLearningRequest,
        control: RunControl | None = None,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._request = request
        self._control = control

    def run(self) -> None:
        try:
            result = self._use_case.execute(self._request, control=self._control)
            self.succeeded.emit(result)
        except RunCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class BatchTransferLearningWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        use_case: RunTransferLearningUseCase,
        request: BatchTransferLearningRequest,
        control: RunControl | None = None,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._request = request
        self._control = control

    def run(self) -> None:
        try:
            result = self._use_case.execute_batch(self._request, control=self._control)
            self.succeeded.emit(result)
        except RunCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
