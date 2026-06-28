from __future__ import annotations

from domain.interfaces import DatasetLoader, TransferLearningMethod
from domain.models import TransferLearningRequest, TransferLearningResult


class RunTransferLearningUseCase:
    def __init__(self, loader: DatasetLoader, methods: list[TransferLearningMethod]) -> None:
        self._loader = loader
        self._methods = {method.name: method for method in methods}

    def execute(self, request: TransferLearningRequest) -> TransferLearningResult:
        if request.method_name not in self._methods:
            supported = ", ".join(sorted(self._methods.keys()))
            raise ValueError(f"Unsupported method: {request.method_name}. Supported: {supported}")

        source_dataset = self._loader.load(request.source_path)
        target_dataset = self._loader.load(request.target_path)

        if source_dataset.features.size == 0 or target_dataset.features.size == 0:
            raise ValueError("Source and target datasets must contain at least one feature row.")

        method = self._methods[request.method_name]
        transformed_target = method.run(source_dataset.features, target_dataset.features)

        output_feature_names = [
            f"{request.method_name.lower()}_feature_{index + 1}"
            for index in range(transformed_target.shape[1])
        ]

        return TransferLearningResult(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            transformed_target=transformed_target,
            output_feature_names=output_feature_names,
        )
