"""Base parser class."""

from abc import ABC, abstractmethod
from datetime import datetime
from email.utils import parsedate_to_datetime
from ...models import NewsItem


class BaseParser(ABC):
    """Base class for RSS parsers."""

    def __init__(self, source_name: str, source_url: str):
        self.source_name = source_name
        self.source_url = source_url

    @abstractmethod
    def parse_entry(self, entry: dict) -> NewsItem:
        """Parse a single RSS entry into NewsItem."""
        pass

    def parse_feed(self, entries: list) -> list[NewsItem]:
        """Parse all entries from feed."""
        items = []
        for entry in entries:
            try:
                item = self.parse_entry(entry)
                published_at = self._extract_published_at(entry)
                if published_at and getattr(item, "published_at", None) is None:
                    item = item.model_copy(update={"published_at": published_at})
                if item and item.title:
                    items.append(item)
            except Exception as e:
                print(f"  Parse error: {e}")
        return items

    def _extract_published_at(self, entry: dict) -> datetime | None:
        published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if published_parsed:
            try:
                return datetime(*published_parsed[:6])
            except Exception:
                pass

        published_text = entry.get("published") or entry.get("updated")
        if published_text:
            try:
                dt = parsedate_to_datetime(published_text)
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            except Exception:
                pass

        return None
