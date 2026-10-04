"""Read-only queries over the experiment catalogue for comparisons and statistics."""

from __future__ import annotations

from application.catalog_summaries import safe_count
from domain.entities import ExperimentHistoryEntry


class CatalogQueriesMixin:
    """Provides read-only queries for domain pairs, comparative benchmark sets, and table counts."""

    def domain_pairs(self, user_id: int | None = None) -> list[tuple[str, str]]:
        """Return distinct source-target domain name pairs that possess recorded evaluations."""
        return sorted(
            {
                (entry.experiment.source_domain, entry.experiment.target_domain)
                for entry in self.history(user_id)
                if entry.evaluation is not None
            }
        )

    def results_for_pair(
        self,
        source_domain: str,
        target_domain: str,
        user_id: int | None = None,
        latest_per_method: bool = True,
    ) -> list[ExperimentHistoryEntry]:
        """Fetch recorded benchmark results for a specific source and target domain pair.

        Args:
            source_domain: Source domain identifier name.
            target_domain: Target domain identifier name.
            user_id: Optional user ID filter.
            latest_per_method: If True, returns only the most recent run for each algorithm.

        Returns:
            list[ExperimentHistoryEntry]: Filtered history records sorted by date descending.
        """
        matching = [
            entry
            for entry in self.history(user_id)
            if entry.evaluation is not None
            and entry.experiment.source_domain == source_domain
            and entry.experiment.target_domain == target_domain
        ]
        matching.sort(key=lambda entry: (entry.experiment.date is not None, entry.experiment.date), reverse=True)
        if not latest_per_method:
            return matching

        seen: set[str] = set()
        latest: list[ExperimentHistoryEntry] = []
        for entry in matching:
            algorithm = entry.experiment.tl_algorithm
            if algorithm not in seen:
                seen.add(algorithm)
                latest.append(entry)
        return latest

    def statistics(self) -> dict[str, int]:
        """Return table row counts for application status display."""
        return {
            "users": safe_count(self._users),
            "datasets": safe_count(self._datasets),
            "experiments": safe_count(self._experiments),
            "evaluations": safe_count(self._evaluations),
        }


__all__ = ["CatalogQueriesMixin"]
