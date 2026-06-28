from __future__ import annotations

from domain.interfaces import DatasetLoader, TransferLearningMethod
from domain.models import TransferLearningRequest, TransferLearningResult, EvaluationMetrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        transformed_source, transformed_target = method.run(source_dataset.features, target_dataset.features)

        output_feature_names = [
            f"{request.method_name.lower()}_feature_{index + 1}"
            for index in range(transformed_target.shape[1])
        ]

        metrics = None
        if source_dataset.labels is not None and target_dataset.labels is not None:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(transformed_source, source_dataset.labels)
            predictions = knn.predict(transformed_target)

            accuracy = float(accuracy_score(target_dataset.labels, predictions))
            try:
                precision = float(precision_score(target_dataset.labels, predictions, average="binary", zero_division=0))
                recall = float(recall_score(target_dataset.labels, predictions, average="binary", zero_division=0))
                f1 = float(f1_score(target_dataset.labels, predictions, average="binary", zero_division=0))
            except ValueError:
                precision = float(precision_score(target_dataset.labels, predictions, average="macro", zero_division=0))
                recall = float(recall_score(target_dataset.labels, predictions, average="macro", zero_division=0))
                f1 = float(f1_score(target_dataset.labels, predictions, average="macro", zero_division=0))

            metrics = EvaluationMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
            )

        return TransferLearningResult(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            transformed_source=transformed_source,
            transformed_target=transformed_target,
            output_feature_names=output_feature_names,
            metrics=metrics,
        )
