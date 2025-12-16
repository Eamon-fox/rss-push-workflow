"""Nature/Nature Neuroscience parser."""

from .base import BaseParser
from ...models import NewsItem


class NatureParser(BaseParser):
    """Parser for Nature journals RSS."""

    _SKIP_PREFIXES = (
        "author correction:",
        "publisher correction:",
        "correction to:",
        "retraction note:",
    )

    def should_skip_entry(self, entry: dict) -> bool:
        title = (entry.get("title") or "").strip().lower()
        if not title:
            return False
        return any(title.startswith(prefix) for prefix in self._SKIP_PREFIXES)

    def parse_entry(self, entry: dict) -> NewsItem:
        # Nature: authors appear as [{"name": "A"}, {"name": "B"}]
        authors = []
        if "authors" in entry:
            authors = [a.get("name", "") for a in entry.get("authors", [])]

        # DOI
        doi = entry.get("prism_doi", "")

        return NewsItem(
            title=entry.get("title", ""),
            content=entry.get("summary", entry.get("description", "")),
            link=entry.get("link", ""),
            authors=authors,
            doi=doi,
            source_name=self.source_name,
            source_url=self.source_url,
        )

