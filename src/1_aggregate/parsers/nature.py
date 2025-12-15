"""Nature/Nature Neuroscience parser."""

from .base import BaseParser
from ...models import NewsItem


class NatureParser(BaseParser):
    """Parser for Nature journals RSS."""

    def parse_entry(self, entry: dict) -> NewsItem:
        # Nature: authors 是列表 [{"name": "A"}, {"name": "B"}]
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
