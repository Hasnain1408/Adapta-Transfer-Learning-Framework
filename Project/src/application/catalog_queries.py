"""Read-only queries over the catalogue, for the comparison module and status bar."""

from __future__ import annotations

from application.catalog_summaries import safe_count
from domain.entities import ExperimentHistoryEntry


class CatalogQueriesMixin:
    """Domain-pair lookups, per-pair result sets, and row counts."""

    def domain_pairs(self, user_id: int | None = None) -> list[tuple[str, str]]:
        """Distinct (source, target) domains that have at least one score.

        Pairs with no evaluation row are left out: there is nothing to plot
        for a run that failed or had no labels.
        """
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
        """Stored, scored runs for one source→target pair.

        With *latest_per_method* (the default) only the most recent run of
        each algorithm is returned — re-running a method to tweak it would
        otherwise put the same algorithm on a comparison chart several times
        with no way to tell the bars apart.
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
        """Row counts per table, for the status bar and the About box."""
        return {
            "users": safe_count(self._users),
            "datasets": safe_count(self._datasets),
            "experiments": safe_count(self._experiments),
            "evaluations": safe_count(self._evaluations),
        }


__all__ = ["CatalogQueriesMixin"]
