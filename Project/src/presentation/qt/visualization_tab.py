"""Domain-adaptation visualization: projected feature space + analysis.

Answers the question the metric cards cannot: *why* did this run score the
way it did? The scatter plot shows where the two domains sit relative to
each other before and after the method ran, and the side panel puts three
independent readings next to it —

- **Domain alignment** (``result.alignment``): MMD before vs. after, which
  measures the distributions directly.
- **Classification performance** (``result.negative_transfer``): the same
  classifier with and without adaptation, which measures the actual task.
- **Negative transfer**: raised from that second comparison alone. A method
  can close the domain gap (good MMD) and still hurt accuracy, so the
  warning deliberately does not consult MMD or the plot.

Everything is projected from the run's real feature matrices — the
preprocessed ones for "Before", ``transformed_*`` for "After". Nothing here
generates or perturbs data.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QCursor

import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from domain.models import TransferLearningResult
from domain.overlap import OVERLAP_DISCLAIMER
from presentation.qt.duration import format_duration
from presentation.qt.report import REPORT_IMAGE_WIDTH
from presentation.qt.style import DARK_PALETTE

STAGE_BEFORE = "Before Adaptation"
STAGE_AFTER = "After Adaptation"
STAGE_BOTH = "Compare Both"

PROJECTION_PCA = "PCA"
PROJECTION_TSNE = "t-SNE"

COLOR_BY_DOMAIN = "Domain"
COLOR_BY_CLASS = "Class"
COLOR_BY_BOTH = "Domain + Class"

#: Every projection is seeded with this so the same run always draws the
#: same picture — a scatter that reshuffles on each redraw is not evidence.
RANDOM_SEED = 42

#: Scatter plots stop being readable long before they stop being drawable,
#: and t-SNE is O(n log n) with a large constant. Past these caps a
#: fixed-seed subsample is drawn; the indices depend only on (n, cap, seed),
#: so Before and After keep showing the *same* samples.
MAX_PLOT_SAMPLES = 3000
MAX_TSNE_SAMPLES = 1500

_TSNE_DEFAULT_PERPLEXITY = 30
_TSNE_DEFAULT_ITERATIONS = 1000


def _subsample_indices(count: int, cap: int) -> np.ndarray:
    """Deterministic row indices, so Before and After plot the same samples."""
    if count <= cap:
        return np.arange(count)
    rng = np.random.default_rng(RANDOM_SEED)
    return np.sort(rng.choice(count, size=cap, replace=False))


def _as_2d(matrix: np.ndarray) -> np.ndarray:
    """Widen a 1-column feature matrix so a 2-component projection is defined."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[1] == 1:
        return np.column_stack([matrix[:, 0], np.zeros(matrix.shape[0])])
    return matrix


class VisualizationTab(QWidget):
    """Interactive PCA/t-SNE view of one run, with its alignment analysis."""

    def __init__(self, palette: dict[str, str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._result: Optional[TransferLearningResult] = None
        self._method_name = "—"
        self._source_name = "—"
        self._target_name = "—"
        #: (stage, projection, perplexity, iterations) -> (source_2d, target_2d)
        self._projection_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        #: Per-axes hover payload: axes -> (coords Nx2, labels list[str]).
        self._hover_data: dict[object, tuple[np.ndarray, list[str]]] = {}
        self._annotation = None
        self._build_ui()
        self._apply_rcparams()
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        self._context_label = QLabel()
        self._context_label.setObjectName("fieldLabel")
        self._context_label.setWordWrap(True)
        outer.addWidget(self._context_label)

        outer.addLayout(self._build_control_bar())
        outer.addWidget(self._build_tsne_bar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_plot_panel())
        splitter.addWidget(self._build_analysis_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 340])
        outer.addWidget(splitter, 1)

        self._status_label = QLabel()
        self._status_label.setObjectName("captionLabel")
        outer.addWidget(self._status_label)

    def _build_control_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self._projection_combo = QComboBox()
        self._projection_combo.addItems([PROJECTION_PCA, PROJECTION_TSNE])
        self._projection_combo.setToolTip(
            "PCA gives a global, linear view of the feature space; t-SNE "
            "emphasises local neighbourhood structure."
        )
        self._projection_combo.currentTextChanged.connect(self._on_projection_changed)

        self._stage_combo = QComboBox()
        self._stage_combo.addItems([STAGE_BEFORE, STAGE_AFTER, STAGE_BOTH])
        self._stage_combo.setCurrentText(STAGE_BOTH)
        self._stage_combo.setToolTip(
            "Before = preprocessed features fed to the method; "
            "After = the method's output."
        )
        self._stage_combo.currentTextChanged.connect(self._render)

        self._color_combo = QComboBox()
        self._color_combo.addItems([COLOR_BY_DOMAIN, COLOR_BY_CLASS, COLOR_BY_BOTH])
        self._color_combo.setToolTip("What the point colours encode")
        self._color_combo.currentTextChanged.connect(self._render)

        self._show_classes_check = QCheckBox("Show Classes")
        self._show_classes_check.setChecked(True)
        self._show_classes_check.setToolTip(
            "Off falls back to plain domain colouring, ignoring class labels"
        )
        self._show_classes_check.toggled.connect(self._on_show_classes_toggled)

        self._reset_button = QPushButton("Reset Visualization")
        self._reset_button.setCursor(Qt.PointingHandCursor)
        self._reset_button.setToolTip("Restore every control above to its default")
        self._reset_button.clicked.connect(self.reset_visualization)

        self._reset_view_button = QPushButton("Reset View")
        self._reset_view_button.setCursor(Qt.PointingHandCursor)
        self._reset_view_button.setToolTip("Undo zoom and pan, keeping the current projection")
        self._reset_view_button.clicked.connect(self._reset_view)

        for label_text, widget in (
            ("Projection:", self._projection_combo),
            ("Stage:", self._stage_combo),
            ("Color By:", self._color_combo),
        ):
            label = QLabel(label_text)
            label.setObjectName("fieldLabel")
            bar.addWidget(label)
            bar.addWidget(widget)
        bar.addWidget(self._show_classes_check)
        bar.addStretch(1)
        bar.addWidget(self._reset_view_button)
        bar.addWidget(self._reset_button)
        return bar

    def _build_tsne_bar(self) -> QWidget:
        self._tsne_bar = QWidget()
        bar = QHBoxLayout(self._tsne_bar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)

        self._perplexity_input = QSpinBox()
        self._perplexity_input.setRange(2, 200)
        self._perplexity_input.setValue(_TSNE_DEFAULT_PERPLEXITY)
        self._perplexity_input.setToolTip(
            "Roughly the number of near neighbours each point is fitted to. "
            "Clamped below (n_samples - 1) / 3 when the run is small."
        )

        self._iterations_input = QSpinBox()
        self._iterations_input.setRange(250, 5000)
        self._iterations_input.setSingleStep(250)
        self._iterations_input.setValue(_TSNE_DEFAULT_ITERATIONS)
        self._iterations_input.setToolTip("Gradient-descent iterations")

        self._recompute_button = QPushButton("Recompute t-SNE")
        self._recompute_button.setCursor(Qt.PointingHandCursor)
        self._recompute_button.setToolTip(
            "t-SNE is slow, so it is only re-run on request after you change "
            "perplexity or iterations."
        )
        self._recompute_button.clicked.connect(self._recompute_tsne)

        for label_text, widget in (
            ("Perplexity:", self._perplexity_input),
            ("Iterations:", self._iterations_input),
        ):
            label = QLabel(label_text)
            label.setObjectName("fieldLabel")
            bar.addWidget(label)
            bar.addWidget(widget)
        bar.addWidget(self._recompute_button)
        bar.addStretch(1)
        self._tsne_bar.setVisible(False)
        return self._tsne_bar

    def _build_plot_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._figure = Figure(figsize=(8, 6), dpi=100)
        # Constrained layout re-solves on every draw, unlike a one-shot
        # tight_layout() that is computed before Qt has given the canvas its
        # real size — which left the title and the x-axis label clipped.
        try:
            self._figure.set_layout_engine("constrained")
        except AttributeError:  # matplotlib < 3.6
            self._figure.set_constrained_layout(True)
        self._canvas = FigureCanvas(self._figure)
        # Below roughly this height Matplotlib runs out of room for the title
        # and the x-axis label and silently clips them, so the canvas gets a
        # floor rather than shrinking to whatever the splitter allows.
        self._canvas.setMinimumHeight(320)
        self._toolbar = NavigationToolbar(self._canvas, self)
        # Zoom/pan come from the toolbar; hover is ours.
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)

        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._toolbar)
        return panel

    def _build_analysis_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 0, 4, 0)
        layout.setSpacing(12)

        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setObjectName("negativeTransferWarning")
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)

        self._alignment_form = QFormLayout()
        layout.addWidget(self._wrap_group("Domain Alignment (MMD)", self._alignment_form))

        # Its own group directly under MMD: both describe the domain gap, but
        # overlap is the one that answers what the scatter beside it *looks*
        # like, so it reads best next to the plot rather than buried in the
        # MMD table.
        self._overlap_form = QFormLayout()
        layout.addWidget(
            self._wrap_group(
                "Domain Overlap",
                self._overlap_form,
                note=OVERLAP_DISCLAIMER,
            )
        )

        self._performance_form = QFormLayout()
        layout.addWidget(self._wrap_group("Classification Performance", self._performance_form))

        self._metadata_form = QFormLayout()
        layout.addWidget(self._wrap_group("Visualization Metadata", self._metadata_form))

        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _wrap_group(title: str, form: QFormLayout, note: str = "") -> QGroupBox:
        group = QGroupBox(title)
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignLeft)
        if not note:
            group.setLayout(form)
            return group

        # A note turns the group into a column: the form, then the caption
        # under it. Set on the group rather than left to the caller so the
        # disclaimer can never be detached from the number it qualifies.
        column = QVBoxLayout(group)
        column.setSpacing(8)
        column.addLayout(form)
        caption = QLabel(note)
        caption.setObjectName("captionLabel")
        caption.setWordWrap(True)
        column.addWidget(caption)
        return group

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------
    def _apply_rcparams(self) -> None:
        palette = self._palette
        fig_bg = palette["surface"]
        matplotlib.rcParams.update(
            {
                "figure.facecolor": fig_bg,
                "savefig.facecolor": fig_bg,
                "axes.facecolor": palette["surface_alt"],
                "axes.edgecolor": palette["border"],
                "axes.labelcolor": palette["text_secondary"],
                "text.color": palette["text_secondary"],
                "xtick.color": palette["text_secondary"],
                "ytick.color": palette["text_secondary"],
                "grid.color": palette["border"],
                "legend.facecolor": palette["surface_alt"],
                "legend.edgecolor": palette["border"],
                "legend.labelcolor": palette["text_secondary"],
            }
        )
        self._figure.set_facecolor(fig_bg)
        self._canvas.setStyleSheet(f"background-color: {fig_bg}; border-radius: 10px;")
        self._toolbar.setStyleSheet(
            f"background-color: {palette['surface_alt']}; border: 1px solid {palette['border']};"
            " border-radius: 7px; padding: 2px;"
        )
        self._warning_label.setStyleSheet(
            f"background-color: {palette['surface_alt']}; color: {palette['danger']};"
            f" border: 1px solid {palette['danger']}; border-radius: 8px; padding: 9px;"
            " font-weight: 600;"
        )

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        self._apply_rcparams()
        if self._result is None:
            self._show_placeholder()
        else:
            self._render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_result(
        self,
        result: TransferLearningResult,
        method_name: str = "—",
        source_name: str = "—",
        target_name: str = "—",
    ) -> None:
        """Point the tab at a finished run and draw it."""
        self._result = result
        self._method_name = method_name
        self._source_name = source_name
        self._target_name = target_name
        self._projection_cache.clear()

        self._context_label.setText(
            f"Method: {method_name}    ·    Source domain: {source_name}"
            f"    →    Target domain: {target_name}"
        )
        self._sync_class_controls()
        self._populate_analysis()
        self._render()

    def clear(self) -> None:
        self._result = None
        self._projection_cache.clear()
        self._context_label.setText("")
        self._show_placeholder()

    def reset_visualization(self) -> None:
        """Every control back to defaults, then redraw."""
        for widget, value in (
            (self._projection_combo, PROJECTION_PCA),
            (self._stage_combo, STAGE_BOTH),
            (self._color_combo, COLOR_BY_DOMAIN),
        ):
            widget.blockSignals(True)
            widget.setCurrentText(value)
            widget.blockSignals(False)
        self._show_classes_check.blockSignals(True)
        self._show_classes_check.setChecked(True)
        self._show_classes_check.blockSignals(False)
        self._perplexity_input.setValue(_TSNE_DEFAULT_PERPLEXITY)
        self._iterations_input.setValue(_TSNE_DEFAULT_ITERATIONS)
        self._tsne_bar.setVisible(False)
        self._sync_class_controls()
        self._render()

    # ------------------------------------------------------------------
    # Control handlers
    # ------------------------------------------------------------------
    def _on_projection_changed(self, text: str) -> None:
        self._tsne_bar.setVisible(text == PROJECTION_TSNE)
        self._render()

    def _on_show_classes_toggled(self, checked: bool) -> None:
        self._sync_class_controls()
        self._render()

    def _sync_class_controls(self) -> None:
        """Grey out class colouring when there are no labels to colour by."""
        has_labels = self._has_class_labels()
        self._show_classes_check.setEnabled(has_labels)
        self._color_combo.setEnabled(has_labels and self._show_classes_check.isChecked())
        if not has_labels:
            self._show_classes_check.setToolTip("This run has no labels to group by")
            self._color_combo.blockSignals(True)
            self._color_combo.setCurrentText(COLOR_BY_DOMAIN)
            self._color_combo.blockSignals(False)

    def _recompute_tsne(self) -> None:
        # Only the t-SNE entries are stale; PCA does not depend on these.
        self._projection_cache = {
            key: value for key, value in self._projection_cache.items()
            if key[1] != PROJECTION_TSNE
        }
        self._render()

    def _reset_view(self) -> None:
        for axes in self._figure.axes:
            axes.relim()
            axes.autoscale()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _has_class_labels(self) -> bool:
        result = self._result
        return result is not None and (
            result.source_dataset.labels is not None
            or result.target_dataset.labels is not None
        )

    def _stage_matrices(self, stage: str) -> tuple[np.ndarray, np.ndarray] | None:
        """The (source, target) feature matrices backing one stage."""
        result = self._result
        if result is None:
            return None
        if stage == STAGE_BEFORE:
            if result.preprocessed_source is None or result.preprocessed_target is None:
                return None
            return result.preprocessed_source, result.preprocessed_target
        return result.transformed_source, result.transformed_target

    def _plot_indices(self, source_rows: int, target_rows: int) -> tuple[np.ndarray, np.ndarray]:
        cap = MAX_TSNE_SAMPLES if self._is_tsne() else MAX_PLOT_SAMPLES
        return _subsample_indices(source_rows, cap), _subsample_indices(target_rows, cap)

    def _is_tsne(self) -> bool:
        return self._projection_combo.currentText() == PROJECTION_TSNE

    def _project(self, stage: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Project one stage's features to 2D, memoised per configuration."""
        projection = self._projection_combo.currentText()
        key = (stage, projection, self._perplexity_input.value(), self._iterations_input.value())
        if key in self._projection_cache:
            return self._projection_cache[key]

        matrices = self._stage_matrices(stage)
        if matrices is None:
            return None
        source, target = (_as_2d(np.asarray(m, dtype=np.float64)) for m in matrices)
        src_idx, tgt_idx = self._plot_indices(source.shape[0], target.shape[0])
        source, target = source[src_idx], target[tgt_idx]

        # Fitted on both domains at once: separate fits would put the two
        # domains in different coordinate systems, and the whole point is to
        # see where they sit relative to each other.
        combined = np.vstack([source, target])
        if projection == PROJECTION_TSNE:
            coords = self._run_tsne(combined)
        else:
            coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(combined)

        split = source.shape[0]
        projected = (coords[:split], coords[split:])
        self._projection_cache[key] = projected
        return projected

    def _run_tsne(self, combined: np.ndarray) -> np.ndarray:
        # sklearn requires perplexity < n_samples, and the usual guidance is
        # to stay under (n - 1) / 3; a small run would otherwise raise.
        n_samples = combined.shape[0]
        perplexity = min(self._perplexity_input.value(), max(2.0, (n_samples - 1) / 3.0))
        kwargs = dict(
            n_components=2,
            perplexity=perplexity,
            random_state=RANDOM_SEED,
            init="pca",
            learning_rate="auto",
        )
        iterations = self._iterations_input.value()
        try:
            return TSNE(max_iter=iterations, **kwargs).fit_transform(combined)
        except TypeError:
            # sklearn < 1.5 spells it n_iter.
            return TSNE(n_iter=iterations, **kwargs).fit_transform(combined)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _show_placeholder(self, message: str = "Run an experiment to visualise the feature space") -> None:
        self._figure.clear()
        self._hover_data.clear()
        axes = self._figure.add_subplot(111)
        axes.axis("off")
        axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=12,
            color=self._palette.get("text_muted", "#8a97a5"),
            transform=axes.transAxes,
        )
        self._canvas.draw()

    def _render(self, *_args) -> None:
        if self._result is None:
            self._show_placeholder()
            return

        stage = self._stage_combo.currentText()
        stages = [STAGE_BEFORE, STAGE_AFTER] if stage == STAGE_BOTH else [stage]

        # t-SNE can take seconds on a large run; the cursor is the only hint
        # the user gets that the app is working rather than wedged.
        busy = self._is_tsne()
        if busy:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            self._status_label.setText("Computing t-SNE projection…")
            QApplication.processEvents()
        try:
            self._figure.clear()
            self._hover_data.clear()
            self._annotation = None

            drawn = 0
            for index, one_stage in enumerate(stages):
                projected = self._project(one_stage)
                if projected is None:
                    continue
                axes = self._figure.add_subplot(1, len(stages), index + 1)
                self._plot_stage(axes, one_stage, *projected)
                drawn += 1

            if drawn == 0:
                self._show_placeholder(
                    "This run predates the Before/After views — re-run the "
                    "experiment to record its preprocessed features."
                )
                return

            self._canvas.draw()
        finally:
            if busy:
                QApplication.restoreOverrideCursor()
        self._update_status()

    def _plot_stage(
        self, axes, stage: str, source_2d: np.ndarray, target_2d: np.ndarray
    ) -> None:
        result = self._result
        projection = self._projection_combo.currentText()
        src_idx, tgt_idx = self._plot_indices(
            self._stage_matrices(stage)[0].shape[0], self._stage_matrices(stage)[1].shape[0]
        )
        source_labels = self._labels_for(result.source_dataset.labels, src_idx)
        target_labels = self._labels_for(result.target_dataset.labels, tgt_idx)

        color_mode = self._color_combo.currentText()
        use_classes = (
            self._show_classes_check.isChecked()
            and self._show_classes_check.isEnabled()
            and color_mode in (COLOR_BY_CLASS, COLOR_BY_BOTH)
            and (source_labels is not None or target_labels is not None)
        )

        if use_classes:
            self._plot_by_class(
                axes, source_2d, target_2d, source_labels, target_labels, color_mode
            )
        else:
            self._plot_by_domain(axes, source_2d, target_2d)

        axis_names = ("PC1", "PC2") if projection == PROJECTION_PCA else ("t-SNE 1", "t-SNE 2")
        axes.set_xlabel(axis_names[0], fontsize=9)
        axes.set_ylabel(axis_names[1], fontsize=9)
        # Deliberately one line: a two-line title is the first thing Qt
        # clips when the splitter squeezes the canvas, and it takes the
        # x-axis label down with it.
        axes.set_title(
            f"{stage} — {projection}  ·  {source_2d.shape[0]} source / "
            f"{target_2d.shape[0]} target",
            fontsize=9,
            fontweight="bold",
            pad=6,
        )
        axes.grid(True, linestyle="--", alpha=0.3)
        axes.legend(fontsize=7, loc="best", framealpha=0.85)

        self._register_hover(axes, source_2d, target_2d, source_labels, target_labels)

    @staticmethod
    def _labels_for(labels, indices: np.ndarray):
        if labels is None:
            return None
        return np.asarray(labels)[indices]

    def _plot_by_domain(self, axes, source_2d: np.ndarray, target_2d: np.ndarray) -> None:
        axes.scatter(
            source_2d[:, 0],
            source_2d[:, 1],
            c=self._palette.get("accent", "#1e6fd9"),
            alpha=0.6,
            s=34,
            edgecolors="none",
            label=f"Source ({source_2d.shape[0]})",
        )
        axes.scatter(
            target_2d[:, 0],
            target_2d[:, 1],
            c=self._palette.get("metric_amber", "#c77700"),
            alpha=0.6,
            s=34,
            marker="^",
            edgecolors="none",
            label=f"Target ({target_2d.shape[0]})",
        )

    def _plot_by_class(
        self, axes, source_2d, target_2d, source_labels, target_labels, color_mode: str
    ) -> None:
        """Colour by class, keeping the domains distinguishable by marker.

        Shape always carries the domain (circle = source, triangle = target)
        so that "are the class clusters preserved?" and "did the domains
        merge?" stay separately readable in the same plot.
        """
        present = [labels for labels in (source_labels, target_labels) if labels is not None]
        classes = np.unique(np.concatenate(present))
        colormap = matplotlib.colormaps["tab10" if len(classes) <= 10 else "tab20"]

        for position, class_value in enumerate(classes):
            color = colormap(position % colormap.N)
            for labels, coords, marker, domain in (
                (source_labels, source_2d, "o", "Source"),
                (target_labels, target_2d, "^", "Target"),
            ):
                if labels is None:
                    continue
                mask = labels == class_value
                if not np.any(mask):
                    continue
                # "Domain + Class" additionally rings the target points, so
                # the two domains stay apart even when printed in greyscale.
                edge = "none"
                if color_mode == COLOR_BY_BOTH and domain == "Target":
                    edge = self._palette.get("text", "#e6edf3")
                axes.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    color=[color],
                    marker=marker,
                    alpha=0.7,
                    s=34,
                    linewidths=0.5,
                    edgecolors=edge,
                    label=f"{domain} · class {class_value} ({int(np.sum(mask))})",
                )

    # ------------------------------------------------------------------
    # Hover
    # ------------------------------------------------------------------
    def _register_hover(self, axes, source_2d, target_2d, source_labels, target_labels) -> None:
        descriptions: list[str] = []
        for position in range(source_2d.shape[0]):
            label = "" if source_labels is None else f", class {source_labels[position]}"
            descriptions.append(f"Source #{position}{label}")
        for position in range(target_2d.shape[0]):
            label = "" if target_labels is None else f", class {target_labels[position]}"
            descriptions.append(f"Target #{position}{label}")
        self._hover_data[axes] = (np.vstack([source_2d, target_2d]), descriptions)

    def _on_hover(self, event) -> None:
        axes = event.inaxes
        payload = self._hover_data.get(axes)
        if payload is None or event.xdata is None:
            self._hide_annotation()
            return

        coords, descriptions = payload
        # Distance is measured in display space so the hit radius is a
        # constant number of pixels regardless of zoom level or axis scale.
        points = axes.transData.transform(coords)
        distances = np.hypot(points[:, 0] - event.x, points[:, 1] - event.y)
        nearest = int(np.argmin(distances))
        if distances[nearest] > 14:
            self._hide_annotation()
            return

        text = f"{descriptions[nearest]}\n({coords[nearest, 0]:.3f}, {coords[nearest, 1]:.3f})"
        if self._annotation is not None:
            self._annotation.remove()
        self._annotation = axes.annotate(
            text,
            xy=(coords[nearest, 0], coords[nearest, 1]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=8,
            color=self._palette.get("tooltip_text", "#ffffff"),
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=self._palette.get("tooltip_bg", "#1f2933"),
                edgecolor=self._palette.get("border", "#39424e"),
            ),
            zorder=20,
        )
        self._canvas.draw_idle()

    def _hide_annotation(self) -> None:
        if self._annotation is not None:
            self._annotation.remove()
            self._annotation = None
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Analysis panel
    # ------------------------------------------------------------------
    def _populate_analysis(self) -> None:
        result = self._result
        self._fill_form(self._alignment_form, self._alignment_rows())
        self._fill_form(self._overlap_form, self._overlap_rows())
        self._fill_form(self._performance_form, self._performance_rows())
        self._fill_form(self._metadata_form, self._metadata_rows())

        analysis = result.negative_transfer if result else None
        if analysis is not None and analysis.is_negative_transfer:
            self._warning_label.setText(
                "⚠  Potential Negative Transfer\n"
                f"Target accuracy fell {abs(analysis.accuracy_delta):.4f} "
                f"({analysis.baseline.accuracy:.4f} → {analysis.adapted.accuracy:.4f}) "
                "against the same classifier without adaptation. This method may be "
                "hurting this domain pair."
            )
            self._warning_label.setVisible(True)
        else:
            self._warning_label.setVisible(False)

    def _fill_form(self, form: QFormLayout, rows: list[tuple[str, str]]) -> None:
        while form.rowCount():
            form.removeRow(0)
        for label_text, value_text in rows:
            label = QLabel(label_text)
            label.setObjectName("fieldLabel")
            value = QLabel(value_text)
            value.setWordWrap(True)
            form.addRow(label, value)

    def _alignment_rows(self) -> list[tuple[str, str]]:
        alignment = self._result.alignment if self._result else None
        if alignment is None:
            return [("Status", "MMD unavailable for this run.")]
        return [
            ("MMD Before", f"{alignment.mmd_before:.6f}"),
            ("MMD After", f"{alignment.mmd_after:.6f}"),
            ("Change", f"{alignment.change_percent:+.2f}%"),
            ("Alignment Status", alignment.status),
        ]

    def _overlap_rows(self) -> list[tuple[str, str]]:
        """Domain Overlap before/after plus the change, in percentage points."""
        alignment = self._result.alignment if self._result else None
        if alignment is None or alignment.overlap_before is None:
            return [("Status", "Overlap unavailable for this run.")]

        change = alignment.overlap_change_points
        rows = [
            ("Overlap Before", f"{alignment.overlap_before:.1f}%"),
            ("Overlap After", f"{alignment.overlap_after:.1f}%"),
        ]
        if change is not None:
            # "pp" spelled out: a change from 60% to 81% is +21 points, and
            # writing it as "+21%" invites reading it as a 21% relative gain.
            direction = "more overlap" if change > 0 else "less overlap"
            rows.append(
                (
                    "Change",
                    f"{change:+.1f} points"
                    + (f"  ({direction})" if abs(change) >= 0.05 else "  (no change)"),
                )
            )
        return rows

    def _performance_rows(self) -> list[tuple[str, str]]:
        analysis = self._result.negative_transfer if self._result else None
        if analysis is None:
            return [("Status", "No target labels — performance cannot be compared.")]
        return [
            (
                "Target Accuracy",
                f"{analysis.baseline.accuracy:.4f} → {analysis.adapted.accuracy:.4f} "
                f"({analysis.accuracy_delta:+.4f})",
            ),
            (
                "Target F1",
                f"{analysis.baseline.f1_score:.4f} → {analysis.adapted.f1_score:.4f} "
                f"({analysis.f1_delta:+.4f})",
            ),
            (
                "Verdict",
                "Potential negative transfer"
                if analysis.is_negative_transfer
                else "Adaptation did not reduce target accuracy",
            ),
        ]

    def _metadata_rows(self) -> list[tuple[str, str]]:
        result = self._result
        if result is None:
            return []
        original_dims = (
            result.preprocessed_source.shape[1]
            if result.preprocessed_source is not None
            else result.source_dataset.features.shape[1]
        )
        return [
            ("Source Samples", str(result.transformed_source.shape[0])),
            ("Target Samples", str(result.transformed_target.shape[0])),
            ("Original Dimensions", str(original_dims)),
            ("Adapted Dimensions", str(result.transformed_target.shape[1])),
            ("Projected Dimensions", "2"),
            ("Method", self._method_name),
            ("Projection", self._projection_combo.currentText()),
            ("Stage", self._stage_combo.currentText()),
            ("Execution Time", format_duration(result.duration_seconds) or "—"),
            ("Adaptation Time", format_duration(result.adaptation_seconds) or "—"),
        ]

    def _update_status(self) -> None:
        result = self._result
        if result is None:
            self._status_label.setText("")
            return
        source_rows = result.transformed_source.shape[0]
        target_rows = result.transformed_target.shape[0]
        cap = MAX_TSNE_SAMPLES if self._is_tsne() else MAX_PLOT_SAMPLES
        parts = [
            f"{source_rows} source and {target_rows} target samples in this run",
            f"projected to 2D with {self._projection_combo.currentText()} (seed {RANDOM_SEED})",
        ]
        if source_rows > cap or target_rows > cap:
            parts.append(f"plotting a fixed {cap}-sample subset per domain for legibility")
        self._status_label.setText("  ·  ".join(parts) + ".")
        # Metadata echoes the live control values, so it has to follow them.
        self._fill_form(self._metadata_form, self._metadata_rows())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def render_to_base64_html(self) -> str:
        """The current scatter plot as an inline PNG for the PDF report.

        See ``REPORT_IMAGE_WIDTH`` for why this is a ``width`` attribute
        rather than a CSS style.
        """
        if self._result is None:
            return ""
        buffer = io.BytesIO()
        self._figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("ascii")
        return (
            f"<img src='data:image/png;base64,{encoded}' "
            f"width='{REPORT_IMAGE_WIDTH}' style='margin-top:10px;' />"
        )

    def analysis_rows(self) -> list[tuple[str, str]]:
        """Flat label/value pairs describing what the plot is showing.

        Consumed by ``report.py`` so the exported PDF carries the same
        numbers as the panel beside the chart.
        """
        if self._result is None:
            return []
        rows = [
            ("Method", self._method_name),
            ("Source domain", self._source_name),
            ("Target domain", self._target_name),
            ("Projection", self._projection_combo.currentText()),
            ("Stage", self._stage_combo.currentText()),
            ("Color by", self._color_combo.currentText()),
        ]
        rows.extend(self._alignment_rows())
        rows.extend(self._overlap_rows())
        rows.extend(self._performance_rows())
        return rows


__all__ = [
    "VisualizationTab",
    "OVERLAP_DISCLAIMER",
    "STAGE_BEFORE",
    "STAGE_AFTER",
    "STAGE_BOTH",
    "PROJECTION_PCA",
    "PROJECTION_TSNE",
]
