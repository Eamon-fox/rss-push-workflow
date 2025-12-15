"""Data source aggregation - RSS, PubMed, BioRxiv."""

from .models import Paper


class Aggregator:
    """Aggregate papers from various sources."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def fetch(self, source_name: str) -> list[Paper]:
        """Fetch papers from a named source."""
        # TODO: Implement source routing
        pass

    def fetch_rss(self, url: str) -> list[Paper]:
        """Fetch papers from RSS feed."""
        # TODO: Implement with feedparser
        pass

    def fetch_pubmed(self, query: str, max_results: int = 100) -> list[Paper]:
        """Fetch papers from PubMed."""
        # TODO: Implement with E-utilities API
        pass

    def fetch_biorxiv(self, category: str = "neuroscience") -> list[Paper]:
        """Fetch papers from BioRxiv."""
        # TODO: Implement with BioRxiv API
        pass
