"""Label/value rows for the analysis panel beside the scatter plot.

Pure formatting over a TransferLearningResult — the same rows feed the
on-screen panel and the exported PDF (via VisualizationTab.analysis_rows),
so the report always states the numbers the user saw.

The three readings are deliberately independent: MMD measures the
distributions, the performance rows measure the actual task, and the
negative-transfer verdict is raised from the performance comparison alone. A
method can close the domain gap and still hurt accuracy.
"""

from __future__ import annotations

from domain.models import TransferLearningResult
from presentation.qt.duration import format_duration


def alignment_rows(result: TransferLearningResult | None) -> list[tuple[str, str]]:
    alignment = result.alignment if result else None
    if alignment is None:
        return [("Status", "MMD unavailable for this run.")]
    return [
        ("MMD Before", f"{alignment.mmd_before:.6f}"),
        ("MMD After", f"{alignment.mmd_after:.6f}"),
        ("Change", f"{alignment.change_percent:+.2f}%"),
        ("Alignment Status", alignment.status),
    ]


def overlap_rows(result: TransferLearningResult | None) -> list[tuple[str, str]]:
    """Domain Overlap before/after plus the change, in percentage points."""
    alignment = result.alignment if result else None
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


def performance_rows(result: TransferLearningResult | None) -> list[tuple[str, str]]:
    analysis = result.negative_transfer if result else None
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


def metadata_rows(
    result: TransferLearningResult | None,
    method_name: str,
    projection: str,
    stage: str,
) -> list[tuple[str, str]]:
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
        ("Method", method_name),
        ("Projection", projection),
        ("Stage", stage),
        ("Execution Time", format_duration(result.duration_seconds) or "—"),
        ("Adaptation Time", format_duration(result.adaptation_seconds) or "—"),
    ]


def negative_transfer_warning(result: TransferLearningResult | None) -> str:
    """The banner text when adaptation hurt, or "" when it did not."""
    analysis = result.negative_transfer if result else None
    if analysis is None or not analysis.is_negative_transfer:
        return ""
    return (
        "⚠  Potential Negative Transfer\n"
        f"Target accuracy fell {abs(analysis.accuracy_delta):.4f} "
        f"({analysis.baseline.accuracy:.4f} → {analysis.adapted.accuracy:.4f}) "
        "against the same classifier without adaptation. This method may be "
        "hurting this domain pair."
    )


__all__ = [
    "alignment_rows",
    "metadata_rows",
    "negative_transfer_warning",
    "overlap_rows",
    "performance_rows",
]
