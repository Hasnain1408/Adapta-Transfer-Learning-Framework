from __future__ import annotations

from itertools import permutations

from domain.interfaces import DatasetLoader, TransferLearningMethod
from domain.models import (
    BatchTransferLearningRequest,
    BatchTransferLearningResult,
    EvaluationMetrics,
    PairTransferLearningResult,
    TransferLearningRequest,
    TransferLearningResult,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


#: Request field -> the private attribute a method must expose to receive it.
TUNABLE_PARAMETERS = {
    "kernel_type": "_kernel_type",
    "regularization": "_regularization",
    "latent_dim": "_latent_dim",
    "mu": "_mu",
}


class RunTransferLearningUseCase:
    def __init__(self, loader: DatasetLoader, methods: list[TransferLearningMethod]) -> None:
        self._loader = loader
        self._methods = {method.name: method for method in methods}

    @property
    def method_names(self) -> list[str]:
        """Registered method names, in the order they were configured."""
        return list(self._methods)

    def dataset_has_combined_domains(self, path: str) -> bool:
        """Whether *path* alone carries both a source and a target half.

        Delegates to the loader when it supports the check (only
        ``CSVDatasetLoader`` does today); other ``DatasetLoader``
        implementations simply report ``False``, meaning "use two files".
        """
        checker = getattr(self._loader, "has_combined_domains", None)
        return bool(checker(path)) if checker is not None else False

    def supported_parameters(self, method_name: str) -> frozenset[str]:
        """Which tunable parameters the named method actually consumes.

        Mirrors the ``hasattr`` rule ``execute`` uses to inject them, so the UI
        can disable controls that would otherwise be silently ignored.
        """
        method = self._methods.get(method_name)
        if method is None:
            return frozenset()
        return frozenset(
            field for field, attribute in TUNABLE_PARAMETERS.items() if hasattr(method, attribute)
        )

    def execute(self, request: TransferLearningRequest) -> TransferLearningResult:
        if request.method_name not in self._methods:
            supported = ", ".join(sorted(self._methods.keys()))
            raise ValueError(f"Unsupported method: {request.method_name}. Supported: {supported}")

        source_dataset = self._loader.load(request.source_path, is_source=True)
        target_dataset = self._loader.load(request.target_path, is_source=False)

        if source_dataset.features.size == 0 or target_dataset.features.size == 0:
            raise ValueError("Source and target datasets must contain at least one feature row.")

        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
        import scipy.linalg
        
        def apply_normalization(X, norm_type):
            if norm_type == "Min-Max Normalization":
                return MinMaxScaler().fit_transform(X)
            elif norm_type == "Z-score Standardization":
                return StandardScaler().fit_transform(X)
            elif norm_type == "Robust Standardization":
                return RobustScaler().fit_transform(X)
            elif norm_type == "L1 normalization":
                return Normalizer(norm='l1').fit_transform(X)
            elif norm_type in ["L2 (Unit Vector) Normalization", "Row-wise unit norm"]:
                return Normalizer(norm='l2').fit_transform(X)
            elif norm_type == "Mean Centering":
                return StandardScaler(with_std=False).fit_transform(X)
            elif norm_type == "Column-wise unit norm":
                return Normalizer(norm='l2').fit_transform(X.T).T
            elif norm_type == "Frobenius normalization":
                norm = np.linalg.norm(X, ord='fro')
                return X / norm if norm > 0 else X
            elif norm_type == "Spectral normalization":
                norm = np.linalg.norm(X, ord=2)
                return X / norm if norm > 0 else X
            return X

        def apply_standardization(X, std_type):
            if std_type == "Z-score Standardization":
                return StandardScaler().fit_transform(X)
            elif std_type == "Robust Standardization":
                return RobustScaler().fit_transform(X)
            elif std_type == "Variance Scaling":
                return StandardScaler(with_mean=False).fit_transform(X)
            elif std_type == "Column mean centering":
                return StandardScaler(with_std=False).fit_transform(X)
            elif std_type == "Row mean centering":
                return X - X.mean(axis=1, keepdims=True)
            elif std_type == "Mahalanobis Standardization":
                cov = np.cov(X, rowvar=False)
                U, S, V = np.linalg.svd(cov)
                S_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-10))
                W = U @ S_inv_sqrt @ U.T
                mean = np.mean(X, axis=0)
                return (X - mean) @ W
            return X

        source_features = apply_normalization(source_dataset.features.copy(), request.normalization)
        target_features = apply_normalization(target_dataset.features.copy(), request.normalization)
        
        source_features = apply_standardization(source_features, request.standardization)
        target_features = apply_standardization(target_features, request.standardization)

        method = self._methods[request.method_name]
        
        for field, attribute in TUNABLE_PARAMETERS.items():
            if hasattr(method, attribute):
                setattr(method, attribute, getattr(request, field))


        transformed_source, transformed_target = method.run(
            source_features,
            target_features,
            source_labels=source_dataset.labels,
        )

        output_feature_names = [
            f"{request.method_name.lower()}_feature_{index + 1}"
            for index in range(transformed_target.shape[1])
        ]

        metrics = None
        target_predictions = None
        target_actuals = None
        prediction_correctness = None

        if source_dataset.labels is not None and target_dataset.labels is not None:
            clf = self._build_classifier(request.classifier)
            clf.fit(transformed_source, source_dataset.labels)
            predictions = clf.predict(transformed_target)

            target_predictions = predictions
            target_actuals = target_dataset.labels

            total_samples = len(target_actuals)
            correct_predictions = int(np.sum(target_predictions == target_actuals))
            wrong_predictions = total_samples - correct_predictions

            prediction_correctness = [
                "Right" if p == a else "Wrong"
                for p, a in zip(target_predictions, target_actuals)
            ]

            # print(f"Total samples size: {total_samples}")
            # print(f"Number of correct predictions: {correct_predictions}")
            # print(f"Number of wrong predictions: {wrong_predictions}\n")

            # print("Target Label Predictions:")
            # for i, pred in enumerate(target_predictions):
            #     sample_name = f"Sample {i+1}"
            #     print(f"{sample_name} -> {pred}")

            # print("\nActual Target Labels:")
            # for i, actual in enumerate(target_actuals):
            #     sample_name = f"Sample {i+1}"
            #     print(f"{sample_name} -> {actual}")

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
                total_samples=total_samples,
                correct_predictions=correct_predictions,
                wrong_predictions=wrong_predictions,
            )

        return TransferLearningResult(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            transformed_source=transformed_source,
            transformed_target=transformed_target,
            output_feature_names=output_feature_names,
            metrics=metrics,
            target_predictions=target_predictions,
            target_actuals=target_actuals,
            prediction_correctness=prediction_correctness,
        )

    def execute_batch(self, request: BatchTransferLearningRequest) -> BatchTransferLearningResult:
        """Run the same configuration over every ordered (source, target) pair.

        A single bad pair (unreadable file, incompatible shapes, ...) is
        recorded as an error on that pair rather than aborting the whole
        batch, so one broken file doesn't cost every other combination.
        """
        if len(request.file_paths) < 2:
            raise ValueError("Batch mode requires at least two dataset files.")

        pairs: list[PairTransferLearningResult] = []
        for source_path, target_path in permutations(request.file_paths, 2):
            pair_request = TransferLearningRequest(
                method_name=request.method_name,
                source_path=source_path,
                target_path=target_path,
                normalization=request.normalization,
                standardization=request.standardization,
                kernel_type=request.kernel_type,
                regularization=request.regularization,
                latent_dim=request.latent_dim,
                mu=request.mu,
                classifier=request.classifier,
            )
            try:
                result = self.execute(pair_request)
                pairs.append(
                    PairTransferLearningResult(source_path=source_path, target_path=target_path, result=result)
                )
            except Exception as exc:  # noqa: BLE001 - surfaced per-pair, not raised
                pairs.append(
                    PairTransferLearningResult(source_path=source_path, target_path=target_path, error=str(exc))
                )

        metric_sets = [pair.result.metrics for pair in pairs if pair.result is not None and pair.result.metrics is not None]
        average_metrics = None
        if metric_sets:
            count = len(metric_sets)
            average_metrics = EvaluationMetrics(
                accuracy=sum(m.accuracy for m in metric_sets) / count,
                precision=sum(m.precision for m in metric_sets) / count,
                recall=sum(m.recall for m in metric_sets) / count,
                f1_score=sum(m.f1_score for m in metric_sets) / count,
                total_samples=sum(m.total_samples for m in metric_sets),
                correct_predictions=sum(m.correct_predictions for m in metric_sets),
                wrong_predictions=sum(m.wrong_predictions for m in metric_sets),
            )

        return BatchTransferLearningResult(pairs=pairs, average_metrics=average_metrics)

    # ------------------------------------------------------------------
    # Classifier factory
    # ------------------------------------------------------------------
    @staticmethod
    def _build_classifier(name: str):
        """Return a fitted-ready sklearn classifier for the given *name*."""
        name = name.strip()
        if name == "SVM":
            return SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)
        if name == "Random Forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        if name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        if name == "Decision Tree":
            return DecisionTreeClassifier(random_state=42)
        # Default / "k-NN"
        return KNeighborsClassifier(n_neighbors=5)
