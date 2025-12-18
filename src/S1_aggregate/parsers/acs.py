"""ACS journals parser (Biochemistry, JACS, etc.)."""

import re
from .base import BaseParser
from ...models import NewsItem


class ACSParser(BaseParser):
    """Parser for ACS journal RSS feeds."""

    _SKIP_PREFIXES = (
        "correction to",
        "addition/correction",
        "erratum:",
        "retraction:",
    )

    def should_skip_entry(self, entry: dict) -> bool:
        title = (entry.get("title") or "").strip().lower()
        if not title:
            return False
        return any(title.startswith(prefix) for prefix in self._SKIP_PREFIXES)

    def parse_entry(self, entry: dict) -> NewsItem:
        # ACS: authors in single string like "John Doe, Jane Smith, and Bob Wilson"
        authors = []
        author_str = entry.get("author", "")
        if author_str:
            # Remove " and " before last author, then split by comma
            author_str = author_str.replace(", and ", ", ")
            authors = [a.strip() for a in author_str.split(",") if a.strip()]

        # DOI: in id field as URL like "http://dx.doi.org/10.1021/acs.biochem.5c00559"
        doi = ""
        entry_id = entry.get("id", "")
        if entry_id:
            # Extract DOI from dx.doi.org URL
            m = re.search(r"doi\.org/(10\.\d+/[^\s]+)", entry_id)
            if m:
                doi = m.group(1)

        return NewsItem(
            title=entry.get("title", ""),
            content=self._extract_content(entry),
            link=entry.get("link", ""),
            authors=authors,
            doi=doi,
            source_name=self.source_name,
            source_url=self.source_url,
        )
