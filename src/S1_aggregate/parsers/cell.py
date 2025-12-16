"""Cell Press parser."""

import re

from .base import BaseParser
from ...models import NewsItem


class CellParser(BaseParser):
    """Parser for Cell Press RSS."""

    _SKIP_PREFIXES = (
        "correction:",
        "publisher correction:",
        "author correction:",
        "retraction notice",
        "erratum",
    )

    def should_skip_entry(self, entry: dict) -> bool:
        title = (entry.get("title") or "").strip().lower()
        if not title:
            return False
        return any(title.startswith(prefix) for prefix in self._SKIP_PREFIXES)

    def parse_entry(self, entry: dict) -> NewsItem:
        # Cell: authors 是 [{"name": "A, B, C"}] 单个字符串，需拆分
        authors = []
        if "author" in entry:
            author_str = entry.get("author", "")
            authors = [a.strip() for a in author_str.split(",") if a.strip()]

        # DOI: Cell RSS 通常在 dc_identifier 直接给 DOI（如 10.1016/j.cell.2025.11.021）
        doi = (entry.get("prism_doi") or "").strip()
        if not doi:
            dc_id = (entry.get("dc_identifier") or "").strip()
            if dc_id.startswith("10."):
                doi = dc_id
            else:
                m = re.search(r"(10\\.\\d{4,9}/[-._;()/:A-Z0-9]+)", dc_id, re.I)
                if m:
                    doi = m.group(1)

        return NewsItem(
            title=entry.get("title", ""),
            content=entry.get("summary", entry.get("description", "")),
            link=entry.get("link", ""),
            authors=authors,
            doi=doi,
            source_name=self.source_name,
            source_url=self.source_url,
        )
