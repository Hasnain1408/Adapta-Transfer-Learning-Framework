"""Domain-alignment measurement: MMD and feature-space overlap, before/after adaptation."""

from __future__ import annotations

from domain.mmd import classify_alignment, compute_mmd, percentage_change
from domain.models import DomainAlignment
from domain.overlap import compute_overlap


def measure_domain_alignment(
    source_before, target_before, source_after, target_after
) -> DomainAlignment | None:
    """MMD and Domain Overlap across the gap, before and after the method.

    Returns ``None`` rather than raising if MMD cannot be computed (for
    example a method that emits source and target in different feature
    spaces) — the alignment panel is informational, and losing it must
    not cost the user the whole run.
    """
    try:
        mmd_before = compute_mmd(source_before, target_before)
        mmd_after = compute_mmd(source_after, target_after)
    except Exception:  # noqa: BLE001 - reported as "unavailable" in the UI
        return None
    change = percentage_change(mmd_before, mmd_after)

    # Overlap is measured separately and allowed to fail on its own: it
    # is the newer, more expensive of the two, and a run that has a
    # usable MMD should still show it rather than dropping the whole
    # alignment panel because the k-NN pass ran out of memory.
    try:
        overlap_before = compute_overlap(source_before, target_before)
        overlap_after = compute_overlap(source_after, target_after)
    except Exception:  # noqa: BLE001 - renders as "—", never as 0%
        overlap_before = None
        overlap_after = None

    return DomainAlignment(
        mmd_before=mmd_before,
        mmd_after=mmd_after,
        change_percent=change,
        status=classify_alignment(change),
        overlap_before=overlap_before,
        overlap_after=overlap_after,
    )


__all__ = ["measure_domain_alignment"]
