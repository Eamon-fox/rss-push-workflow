"""bioRxiv/medRxiv preprint parser."""

import re
from .base import BaseParser
from ...models import NewsItem, SourceType


class BiorxivParser(BaseParser):
    """Parser for bioRxiv/medRxiv RSS feeds."""

    source_type = SourceType.PREPRINT

    def parse_entry(self, entry: dict) -> NewsItem:
        """
        Parse bioRxiv RSS entry.

        bioRxiv RSS格式:
        - title: 论文标题
        - link: https://www.biorxiv.org/content/10.1101/2025.01.01.123456v1
        - description/summary: 摘要
        - dc:creator: 作者 (逗号分隔)
        - dc:identifier: doi:10.1101/2025.01.01.123456
        """
        title = entry.get("title", "")
        link = entry.get("link", "")

        # 摘要
        content = entry.get("summary", entry.get("description", ""))

        # 作者 - dc:creator 通常是逗号分隔的字符串
        authors = []
        creator = entry.get("dc_creator", entry.get("author", ""))
        if creator:
            if isinstance(creator, str):
                # "Author1, Author2, Author3"
                authors = [a.strip() for a in creator.split(",") if a.strip()]
            elif isinstance(creator, list):
                authors = [str(a).strip() for a in creator if a]

        # DOI - 从dc:identifier或link提取
        doi = ""
        dc_id = entry.get("dc_identifier", "")
        if dc_id and dc_id.startswith("doi:"):
            doi = dc_id[4:]
        elif not doi and link:
            # 从link提取: https://www.biorxiv.org/content/10.1101/2025.01.01.123456v1
            match = re.search(r"(10\.\d{4,}/[^\s]+?)(?:v\d+)?$", link)
            if match:
                doi = match.group(1)

        return NewsItem(
            title=title,
            content=content,
            link=link,
            authors=authors,
            doi=doi,
            source_type=self.source_type,
            source_name=self.source_name,
            source_url=self.source_url,
        )


class MedrxivParser(BiorxivParser):
    """Parser for medRxiv RSS feeds (same format as bioRxiv)."""
    pass
