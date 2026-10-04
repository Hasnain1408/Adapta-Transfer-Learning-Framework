"""Drawing the six comparison views.

One _draw_* per entry in VIEWS; _render in the tab picks which
to call. Each takes the axes and the already-filtered rows, so none of
them re-reads the controls or the database.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from presentation.qt.comparison_model import (
    BASELINE_LABEL,
    METRICS,
    METRIC_OVERALL,
    MethodRow,
    rows_from_entries,
)
from presentation.qt.duration import format_duration


class ComparisonChartsMixin:
    """The per-view chart renderers used by ComparisonTab._render."""

    def _bar_colors(self, count: int) -> list:
        keys = ["metric_teal", "metric_purple", "metric_amber", "metric_pink", "accent"]
        return [self._palette.get(keys[i % len(keys)], "#1e6fd9") for i in range(count)]

    def _register_bars(self, axes, bars, labels: list[str]) -> None:
        for bar, label in zip(bars, labels):
            self._hover_bars.append((axes, bar, label))

    # -- individual views ----------------------------------------------
    def _draw_performance(self, axes, rows: list[MethodRow]) -> None:
        """Grouped bars: one cluster per method, one bar per metric."""
        names = [row.method for row in rows]
        include_baseline = self._baseline_check.isChecked()
        baseline_accuracy = self._baseline_value(rows, "accuracy")
        baseline_f1 = self._baseline_value(rows, "f1_score")
        if include_baseline and (baseline_accuracy is not None or baseline_f1 is not None):
            names = names + [BASELINE_LABEL]

        positions = np.arange(len(names))
        width = 0.8 / len(METRICS)
        colors = self._bar_colors(len(METRICS))

        for index, (label, attribute) in enumerate(METRICS):
            values, offsets, texts = [], [], []
            for slot, name in enumerate(names):
                if name == BASELINE_LABEL:
                    # Only accuracy and F1 have a recorded no-adaptation value.
                    value = {"accuracy": baseline_accuracy, "f1_score": baseline_f1}.get(attribute)
                else:
                    value = rows[slot].value(attribute)
                if value is None:
                    continue
                values.append(value)
                offsets.append(positions[slot] + index * width - 0.4 + width / 2)
                texts.append(f"{name} · {label} = {value:.4f}")
            if not values:
                continue
            bars = axes.bar(offsets, values, width=width, color=colors[index], label=label)
            self._register_bars(axes, bars, texts)

        axes.set_xticks(positions)
        axes.set_xticklabels(names, rotation=0, fontsize=8)
        axes.set_ylabel("Score", fontsize=9)
        axes.set_ylim(0, 1.05)
        axes.set_title(self._chart_title("Performance by method"), fontsize=10, fontweight="bold")
        axes.grid(axis="y", linestyle="--", alpha=0.3)
        axes.legend(fontsize=7, ncol=len(METRICS), loc="upper right", framealpha=0.85)

    def _draw_ranking(self, axes, rows: list[MethodRow]) -> None:
        """Horizontal bars, best at the top, for the selected metric."""
        attribute = self._metric_attribute()
        label = self._metric_label()
        overall = attribute == METRIC_OVERALL

        ranked = [(rank, row) for rank, row in self.ranked_rows() if rank is not None]
        if not ranked:
            self._placeholder(f"No stored {label} values for this pair.")
            return

        # Best at the top: barh draws upwards, so plot worst-first.
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        names = [row.method for _rank, row in ranked]
        values = [self._metric_value(row) or 0.0 for _rank, row in ranked]
        positions = np.arange(len(ranked))
        places = 2 if overall else 4

        # The winner is coloured apart so the answer is readable at a glance.
        best = min(rank for rank, _row in ranked)
        colors = [
            self._palette.get("success", "#14804a") if rank == best
            else self._palette.get("metric_teal", "#0f9b8e")
            for rank, _row in ranked
        ]
        bars = axes.barh(positions, values, color=colors)
        self._register_bars(
            axes,
            bars,
            [f"#{rank}  {row.method} · {label} = {self._metric_value(row):.{places}f}"
             for rank, row in ranked],
        )

        span = max(values) or 1.0
        for position, value, (rank, _row) in zip(positions, values, ranked):
            axes.text(
                value + span * 0.015, position,
                f"{value:.{places}f}   #{rank}",
                va="center", fontsize=8,
            )

        if self._baseline_check.isChecked() and not overall:
            # An average rank has no baseline to compare against — the
            # baseline never competed in the ranking.
            baseline = self._baseline_value(rows, attribute)
            if baseline is not None:
                axes.axvline(
                    baseline,
                    color=self._palette.get("danger", "#b3261e"),
                    linestyle="--",
                    linewidth=1.4,
                    label=f"{BASELINE_LABEL} {baseline:.4f}",
                )
                axes.legend(fontsize=7, loc="lower right", framealpha=0.85)

        axes.set_yticks(positions)
        axes.set_yticklabels(names, fontsize=8)
        axes.set_xlim(0, span * 1.28 if overall else 1.18)
        axes.set_xlabel(
            "Average rank across metrics (lower is better)" if overall else label, fontsize=9
        )
        axes.set_title(
            self._chart_title(
                "Methods ranked overall" if overall else f"Methods ranked by {label}"
            ),
            fontsize=10,
            fontweight="bold",
        )
        axes.grid(axis="x", linestyle="--", alpha=0.3)

    def _draw_alignment(self, axes, rows: list[MethodRow]) -> None:
        """MMD before vs. after, per method, with the change annotated."""
        scored = [row for row in rows if row.mmd_before is not None and row.mmd_after is not None]
        if not scored:
            self._placeholder(
                "No stored MMD values for this pair.\n"
                "Re-run these experiments to record domain alignment."
            )
            return

        positions = np.arange(len(scored))
        width = 0.38
        before = [row.mmd_before or 0.0 for row in scored]
        after = [row.mmd_after or 0.0 for row in scored]

        bars_before = axes.bar(
            positions - width / 2, before, width,
            color=self._palette.get("metric_amber", "#c77700"), label="MMD before",
        )
        bars_after = axes.bar(
            positions + width / 2, after, width,
            color=self._palette.get("metric_teal", "#0f9b8e"), label="MMD after",
        )
        self._register_bars(axes, bars_before,
            [f"{r.method} · MMD before = {r.mmd_before:.6f}" for r in scored])
        self._register_bars(axes, bars_after,
            [f"{r.method} · MMD after = {r.mmd_after:.6f}  ({r.alignment_status or '—'})"
             for r in scored])

        top = max(max(before), max(after)) or 1.0
        for position, row in zip(positions, scored):
            if row.mmd_change is None:
                continue
            improved = row.mmd_change >= 0
            axes.text(
                position,
                max(row.mmd_before or 0, row.mmd_after or 0) + top * 0.03,
                f"{row.mmd_change:+.1f}%",
                ha="center",
                fontsize=7,
                fontweight="bold",
                color=self._palette["success"] if improved else self._palette["danger"],
            )

        axes.set_xticks(positions)
        axes.set_xticklabels([row.method for row in scored], rotation=0, fontsize=8)
        axes.set_ylabel("MMD", fontsize=9)
        axes.set_ylim(0, top * 1.2)
        axes.set_title(
            self._chart_title("Domain alignment before vs. after adaptation"),
            fontsize=10, fontweight="bold",
        )
        axes.grid(axis="y", linestyle="--", alpha=0.3)
        axes.legend(fontsize=7, framealpha=0.85)

    def _draw_execution(self, axes, rows: list[MethodRow]) -> None:
        scored = [row for row in rows if row.execution_seconds is not None]
        if not scored:
            self._placeholder(
                "No stored execution times for this pair.\n"
                "Re-run these experiments to record timing."
            )
            return

        scored.sort(key=lambda row: row.execution_seconds or 0.0)
        positions = np.arange(len(scored))
        values = [row.execution_seconds or 0.0 for row in scored]
        bars = axes.bar(positions, values, color=self._palette.get("metric_purple", "#7b5bd6"))
        self._register_bars(
            axes, bars, [f"{r.method} · {format_duration(r.execution_seconds)}" for r in scored]
        )
        for position, value in zip(positions, values):
            axes.text(position, value, f" {format_duration(value)}", ha="center",
                      va="bottom", fontsize=7, rotation=90)

        axes.set_xticks(positions)
        axes.set_xticklabels([row.method for row in scored], rotation=0, fontsize=8)
        axes.set_ylabel("Seconds", fontsize=9)
        axes.set_ylim(0, max(values) * 1.28 if max(values) else 1.0)
        axes.set_title(self._chart_title("Execution time by method"), fontsize=10, fontweight="bold")
        axes.grid(axis="y", linestyle="--", alpha=0.3)

    def _draw_negative_transfer(self, axes, rows: list[MethodRow]) -> None:
        """Baseline vs. adapted accuracy, flagging the methods that lost."""
        scored = [row for row in rows if row.baseline_accuracy is not None]
        if not scored:
            self._placeholder(
                "No stored baseline scores for this pair.\n"
                "Re-run these experiments to record the no-adaptation baseline."
            )
            return

        positions = np.arange(len(scored))
        width = 0.38
        baseline = [row.baseline_accuracy or 0.0 for row in scored]
        adapted = [row.accuracy or 0.0 for row in scored]

        bars_base = axes.bar(
            positions - width / 2, baseline, width,
            color=self._palette.get("text_muted", "#8a97a5"), label="Without adaptation",
        )
        # Losing methods are coloured as the warning they are, so negative
        # transfer is visible without reading the delta labels.
        colors = [
            self._palette["danger"] if row.is_negative_transfer else self._palette["success"]
            for row in scored
        ]
        bars_adapted = axes.bar(
            positions + width / 2, adapted, width, color=colors, label="After adaptation"
        )
        self._register_bars(axes, bars_base,
            [f"{r.method} · baseline accuracy = {r.baseline_accuracy:.4f}" for r in scored])
        self._register_bars(axes, bars_adapted,
            [f"{r.method} · adapted = {r.accuracy:.4f} ({r.accuracy_delta:+.4f})"
             + ("  ⚠ negative transfer" if r.is_negative_transfer else "") for r in scored])

        for position, row in zip(positions, scored):
            if row.accuracy_delta is None:
                continue
            axes.text(
                position,
                max(row.baseline_accuracy or 0, row.accuracy or 0) + 0.02,
                f"{row.accuracy_delta:+.3f}",
                ha="center", fontsize=7, fontweight="bold",
                color=self._palette["danger"] if row.is_negative_transfer
                else self._palette["success"],
            )

        axes.set_xticks(positions)
        axes.set_xticklabels([row.method for row in scored], rotation=0, fontsize=8)
        axes.set_ylabel("Target accuracy", fontsize=9)
        axes.set_ylim(0, 1.12)
        axes.set_title(
            self._chart_title("Baseline vs. transfer accuracy"), fontsize=10, fontweight="bold"
        )
        axes.grid(axis="y", linestyle="--", alpha=0.3)
        axes.legend(fontsize=7, framealpha=0.85)

    def _draw_pairs(self, axes) -> None:
        """The selected metric for one method across every stored pair."""
        attribute = self._metric_attribute()
        label = self._metric_label()
        selected = self._selected_methods()
        if self._catalog is None:
            self._placeholder("No database connection.")
            return

        pair_labels: list[str] = []
        per_method: dict[str, list[Optional[float]]] = {name: [] for name in sorted(selected)}
        for source, target in self._pairs:
            try:
                entries = self._catalog.results_for_pair(source, target, self._user_id())
            except Exception:  # noqa: BLE001
                continue
            by_method = {row.method: row for row in rows_from_entries(entries)}
            if not any(name in by_method for name in per_method):
                continue
            pair_labels.append(f"{source}→{target}")
            for name in per_method:
                row = by_method.get(name)
                per_method[name].append(row.value(attribute) if row else None)

        if not pair_labels:
            self._placeholder("No other stored pairs use the selected methods.")
            return

        positions = np.arange(len(pair_labels))
        active = {name: values for name, values in per_method.items() if any(v is not None for v in values)}
        if not active:
            self._placeholder(f"No stored {label} values across pairs.")
            return
        width = 0.8 / len(active)
        colors = self._bar_colors(len(active))

        for index, (name, values) in enumerate(sorted(active.items())):
            offsets, heights, texts = [], [], []
            for slot, value in enumerate(values):
                if value is None:
                    continue
                offsets.append(positions[slot] + index * width - 0.4 + width / 2)
                heights.append(value)
                texts.append(f"{name} · {pair_labels[slot]} · {label} = {value:.4f}")
            if not heights:
                continue
            bars = axes.bar(offsets, heights, width=width, color=colors[index], label=name)
            self._register_bars(axes, bars, texts)

        axes.set_xticks(positions)
        axes.set_xticklabels(pair_labels, rotation=20, ha="right", fontsize=8)
        axes.set_ylabel(label, fontsize=9)
        axes.set_ylim(0, 1.05)
        axes.set_title(f"{label} across source–target pairs", fontsize=10, fontweight="bold")
        axes.grid(axis="y", linestyle="--", alpha=0.3)
        axes.legend(fontsize=7, ncol=min(len(active), 5), framealpha=0.85)

    def _chart_title(self, prefix: str) -> str:
        return f"{prefix}  ({self._source_combo.currentText()} → {self._target_combo.currentText()})"

    # ------------------------------------------------------------------
    # Table, caption, hover


__all__ = ["ComparisonChartsMixin"]
