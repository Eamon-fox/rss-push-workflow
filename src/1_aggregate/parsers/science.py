"""Science parser."""

from .base import BaseParser
from ...models import NewsItem


class ScienceParser(BaseParser):
    """Parser for Science RSS."""

    def parse_entry(self, entry: dict) -> NewsItem:
        # Science: authors 格式同 Nature
        authors = []
        if "authors" in entry:
            authors = [a.get("name", "") for a in entry.get("authors", [])]

        # DOI: prism_doi 或从 dc_identifier 提取
        doi = entry.get("prism_doi", "")
        if not doi:
            dc_id = entry.get("dc_identifier", "")
            if dc_id.startswith("doi:"):
                doi = dc_id[4:]  # 去掉 "doi:" 前缀

        return NewsItem(
            title=entry.get("title", ""),
            content=entry.get("summary", entry.get("description", "")),
            link=entry.get("link", ""),
            authors=authors,
            doi=doi,
            source_name=self.source_name,
            source_url=self.source_url,
        )
