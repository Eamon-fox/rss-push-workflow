"""Science parser."""

from .base import BaseParser
from ...models import NewsItem


class ScienceParser(BaseParser):
    """Parser for Science RSS."""

    _SKIP_PREFIXES = (
        # 勘误/撤稿
        "correction",
        "retraction",
        "publisher correction",
        "editorial expression of concern",
        "erratum",
        # 新闻摘要/目录
        "in other journals",
        "in science journals",
        "this week in science",
        # 编辑内容 (非研究)
        "editors' choice",
        "editor's choice",
    )

    def should_skip_entry(self, entry: dict) -> bool:
        title = (entry.get("title") or "").strip().lower()
        if not title:
            return False
        normalized = title.replace(":", "")
        return any(normalized.startswith(prefix) for prefix in self._SKIP_PREFIXES)

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
            content=self._extract_content(entry),
            link=entry.get("link", ""),
            authors=authors,
            doi=doi,
            source_name=self.source_name,
            source_url=self.source_url,
        )
