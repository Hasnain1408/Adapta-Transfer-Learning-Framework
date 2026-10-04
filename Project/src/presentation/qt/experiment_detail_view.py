"""Full-detail read-only view of one recorded experiment.

Opened as a closable tab in MainWindow's results tab strip when a row in
the History tab is clicked. The history table only shows a slice of one row
per experiment; this joins together everything the experiment, evaluation,
user, and (source/target) dataset rows know about that run.
"""

from __future__ import annotations

from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from domain.entities import DatasetRecord, EvaluationRecord, ExperimentHistoryEntry, ExperimentRecord
from presentation.qt.metric_card import MetricCard


def _format_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return str(value or "—")


def _dash(value: object) -> str:
    return "—" if value is None or value == "" else str(value)


class ExperimentDetailView(QWidget):
    """Read-only page combining the experiment, evaluation, user, and dataset rows."""

    export_requested = pyqtSignal()

    def __init__(
        self,
        entry: ExperimentHistoryEntry,
        source_dataset: DatasetRecord | None,
        target_dataset: DatasetRecord | None,
        palette: dict[str, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette
        experiment = entry.experiment

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        layout.addLayout(self._build_header(entry))
        if entry.evaluation is not None:
            layout.addLayout(self._build_metrics_row(entry.evaluation))
        layout.addWidget(self._build_overview_group(entry))
        layout.addWidget(self._build_configuration_group(experiment))

        datasets_row = QHBoxLayout()
        datasets_row.setSpacing(14)
        datasets_row.addWidget(
            self._build_dataset_group(
                "Source Dataset", experiment.source_domain, source_dataset
            )
        )
        datasets_row.addWidget(
            self._build_dataset_group(
                "Target Dataset", experiment.target_domain, target_dataset
            )
        )
        layout.addLayout(datasets_row)

        layout.addStretch(1)
        scroll.setWidget(content)

    # ------------------------------------------------------------------
    def _build_header(self, entry: ExperimentHistoryEntry) -> QHBoxLayout:
        experiment = entry.experiment
        row = QHBoxLayout()
        row.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel(f"Experiment #{experiment.id} — {experiment.tl_algorithm}")
        title.setObjectName("appTitle")
        subtitle = QLabel(
            f"{experiment.source_domain} → {experiment.target_domain}  ·  "
            f"{_format_timestamp(experiment.date)}"
        )
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        row.addLayout(title_box)
        row.addStretch(1)

        badge = QLabel(experiment.status.label)
        badge.setObjectName("statusBadge")
        badge.setProperty("status", experiment.status.value)
        row.addWidget(badge, 0, Qt.AlignTop)

        export_button = QPushButton("Export Report…")
        export_button.setCursor(Qt.PointingHandCursor)
        export_button.clicked.connect(self.export_requested.emit)
        row.addWidget(export_button, 0, Qt.AlignTop)
        return row

    def _build_metrics_row(self, evaluation: EvaluationRecord) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        specs = [
            ("Accuracy", evaluation.accuracy, "teal"),
            ("Precision", evaluation.precision, "purple"),
            ("Recall", evaluation.recall, "amber"),
            ("F1 Score", evaluation.f1_score, "pink"),
        ]
        for title, value, accent in specs:
            card = MetricCard(title, accent=accent)
            card.set_value(f"{value:.4f}")
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(card)
        return row

    def _build_overview_group(self, entry: ExperimentHistoryEntry) -> QGroupBox:
        experiment = entry.experiment
        group = QGroupBox("Overview")
        form = QFormLayout(group)
        form.setSpacing(8)

        user = entry.user
        form.addRow(
            self._label("User"),
            self._value(f"{user.name}  ({user.email})" if user else "— (account deleted)"),
        )
        if user is not None:
            form.addRow(self._label("Role"), self._value(user.role.label))
        if entry.evaluation is not None:
            evaluation = entry.evaluation
            form.addRow(
                self._label("Samples Evaluated"),
                self._value(
                    f"{evaluation.total_samples} total  ·  "
                    f"{evaluation.correct_predictions} correct  ·  "
                    f"{evaluation.wrong_predictions} incorrect"
                ),
            )
        else:
            form.addRow(
                self._label("Samples Evaluated"),
                self._value("No metrics recorded (unlabelled data, or the run failed)."),
            )
        if experiment.error_message:
            error_value = self._value(experiment.error_message)
            error_value.setStyleSheet(f"color: {self._palette['danger']};")
            form.addRow(self._label("Error"), error_value)
        return group

    def _build_configuration_group(self, experiment: ExperimentRecord) -> QGroupBox:
        group = QGroupBox("Run Configuration")
        form = QFormLayout(group)
        form.setSpacing(8)
        form.addRow(self._label("Algorithm"), self._value(experiment.tl_algorithm))
        form.addRow(self._label("Classifier"), self._value(_dash(experiment.classifier)))
        form.addRow(self._label("Normalization"), self._value(experiment.normalization_type))
        form.addRow(self._label("Standardization"), self._value(experiment.standardization_type))
        form.addRow(self._label("Kernel Type"), self._value(_dash(experiment.kernel_type)))
        form.addRow(
            self._label("Regularization (λ)"),
            self._value(
                "—"
                if experiment.regularization_value is None
                else f"{experiment.regularization_value:g}"
            ),
        )
        form.addRow(self._label("Latent Dimensions"), self._value(_dash(experiment.latent_feature)))
        return group

    def _build_dataset_group(
        self, title: str, domain_name: str, dataset: DatasetRecord | None
    ) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setSpacing(8)
        form.addRow(self._label("Domain"), self._value(domain_name or "—"))

        if dataset is None:
            form.addRow(self._value("No longer in the dataset registry."))
            return group

        form.addRow(self._label("File"), self._value(dataset.name))
        form.addRow(self._label("Format"), self._value(dataset.file_format.value))
        form.addRow(self._label("Samples"), self._value(str(dataset.num_samples)))
        form.addRow(self._label("Features"), self._value(str(dataset.num_features)))
        form.addRow(self._label("Label Column"), self._value(_dash(dataset.label_name)))

        classes_value = self._value(str(dataset.num_classes))
        if dataset.class_names:
            classes_value.setToolTip(f"Classes: {', '.join(dataset.class_names)}")
        form.addRow(self._label("Classes"), classes_value)

        dtype_value = self._value(dataset.dtype_summary or "—")
        form.addRow(self._label("Data Types"), dtype_value)
        form.addRow(self._label("Missing Values"), self._value(str(dataset.missing_value_count)))
        form.addRow(self._label("Registered"), self._value(_format_timestamp(dataset.created_at)))
        return group

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _value(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        return label


__all__ = ["ExperimentDetailView"]
