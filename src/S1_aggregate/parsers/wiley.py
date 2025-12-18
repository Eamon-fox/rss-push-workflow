"""Wiley journals parser (FEBS Journal, FEBS Letters, etc.)."""

from .base import BaseParser
from ...models import NewsItem


class WileyParser(BaseParser):
    """Parser for Wiley journal RSS feeds."""

    _SKIP_PREFIXES = (
        "correction:",
        "corrigendum:",
        "erratum:",
        "retraction:",
    )

    def should_skip_entry(self, entry: dict) -> bool:
        title = (entry.get("title") or "").strip().lower()
        if not title:
            return False
        return any(title.startswith(prefix) for prefix in self._SKIP_PREFIXES)

    def parse_entry(self, entry: dict) -> NewsItem:
        # Wiley: authors are newline-separated in a single string
        # e.g. "Akane Yato, \nYuki Kato, \nFuyuko Hayashi"
        authors = []
        author_str = entry.get("author", "")
        if author_str:
            # Split by newline and/or comma, clean up
            parts = author_str.replace("\n", ",").split(",")
            authors = [a.strip() for a in parts if a.strip()]

        # DOI: prism_doi or dc_identifier
        doi = (entry.get("prism_doi") or entry.get("dc_identifier") or "").strip()

        return NewsItem(
            title=entry.get("title", ""),
            content=self._extract_content(entry),
            link=entry.get("link", ""),
            authors=authors,
            doi=doi,
            source_name=self.source_name,
            source_url=self.source_url,
        )
