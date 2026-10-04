"""Domain alignment measurement for MMD divergence and feature space overlap."""

from __future__ import annotations

from domain.mmd import classify_alignment, compute_mmd, percentage_change
from domain.models import DomainAlignment
from domain.overlap import compute_overlap


def measure_domain_alignment(
    source_before, target_before, source_after, target_after
) -> DomainAlignment | None:
    """Measure MMD divergence and domain overlap before and after feature adaptation.

    Returns None if MMD calculation fails rather than aborting the experiment.
    """
    try:
        mmd_before = compute_mmd(source_before, target_before)
        mmd_after = compute_mmd(source_after, target_after)
    except Exception:
        return None
    change = percentage_change(mmd_before, mmd_after)

    try:
        overlap_before = compute_overlap(source_before, target_before)
        overlap_after = compute_overlap(source_after, target_after)
    except Exception:
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
