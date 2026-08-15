from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from application.run_transfer_learning_use_case import RunTransferLearningUseCase
from domain.models import BatchTransferLearningRequest, TransferLearningRequest
from domain.run_control import RunCancelled, RunControl


class _ProgressReportingWorker(QThread):
    """Shared plumbing for the two workers: signals plus progress relay.

    ``progress`` carries a ``ProgressUpdate``. It is emitted from *this*
    thread, but the worker object itself lives on the UI thread, so Qt's
    automatic connection type queues each emission onto the UI event loop —
    which is the only reason it is safe to touch widgets in the slot.
    Relaying through a signal is also why ``RunControl``'s listener may be
    called from the worker thread at all: the emit itself is thread-safe
    and returns immediately without running the slot inline.
    """

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(object)

    def __init__(
        self,
        use_case: RunTransferLearningUseCase,
        request,
        control: RunControl | None = None,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._request = request
        self._control = control
        if control is not None:
            control.set_listener(self.progress.emit)

    def _execute(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self) -> None:
        try:
            result = self._execute()
            self.succeeded.emit(result)
        except RunCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            # The C++ object is about to be deleteLater'd; a late report
            # from a straggling thread would emit from a dead signal.
            if self._control is not None:
                self._control.set_listener(None)


class TransferLearningWorker(_ProgressReportingWorker):
    def __init__(
        self,
        use_case: RunTransferLearningUseCase,
        request: TransferLearningRequest,
        control: RunControl | None = None,
    ) -> None:
        super().__init__(use_case, request, control)

    def _execute(self):
        return self._use_case.execute(self._request, control=self._control)


class BatchTransferLearningWorker(_ProgressReportingWorker):
    def __init__(
        self,
        use_case: RunTransferLearningUseCase,
        request: BatchTransferLearningRequest,
        control: RunControl | None = None,
    ) -> None:
        super().__init__(use_case, request, control)

    def _execute(self):
        return self._use_case.execute_batch(self._request, control=self._control)
