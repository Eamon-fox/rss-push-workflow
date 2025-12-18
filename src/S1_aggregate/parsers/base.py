"""Base parser class."""

from abc import ABC, abstractmethod
from datetime import datetime
from email.utils import parsedate_to_datetime
from ...models import NewsItem, SourceType


class BaseParser(ABC):
    """Base class for RSS parsers."""

    # 子类可覆盖
    source_type: SourceType = SourceType.JOURNAL

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
            if self.should_skip_entry(entry):
                continue
            try:
                item = self.parse_entry(entry)
                # 补充published_at
                published_at = self._extract_published_at(entry)
                if published_at and getattr(item, "published_at", None) is None:
                    item = item.model_copy(update={"published_at": published_at})
                # 补充image_url
                image_url = self._extract_image_url(entry)
                if image_url and not item.image_url:
                    item = item.model_copy(update={"image_url": image_url})
                if item and item.title:
                    items.append(item)
            except Exception as e:
                print(f"  Parse error: {e}")
        return items

    def should_skip_entry(self, entry: dict) -> bool:
        """
        Hook for subclasses to skip noisy entries (e.g. corrections/retractions).

        Returns:
            True to skip entry entirely, False to continue parsing.
        """
        return False

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

    def _extract_image_url(self, entry: dict) -> str:
        """
        从RSS entry中提取配图URL.

        常见位置:
        - media_content: [{"url": "...", "type": "image/..."}]
        - media_thumbnail: [{"url": "..."}]
        - enclosures: [{"href": "...", "type": "image/..."}]
        - links: [{"rel": "enclosure", "type": "image/...", "href": "..."}]
        """
        # 1. media:content
        media_content = entry.get("media_content", [])
        for media in media_content:
            url = media.get("url", "")
            media_type = media.get("type", "")
            if url and ("image" in media_type or self._looks_like_image(url)):
                return url

        # 2. media:thumbnail
        media_thumbnail = entry.get("media_thumbnail", [])
        for thumb in media_thumbnail:
            url = thumb.get("url", "")
            if url:
                return url

        # 3. enclosures
        enclosures = entry.get("enclosures", [])
        for enc in enclosures:
            url = enc.get("href", enc.get("url", ""))
            enc_type = enc.get("type", "")
            if url and ("image" in enc_type or self._looks_like_image(url)):
                return url

        # 4. links with rel=enclosure
        links = entry.get("links", [])
        for link in links:
            if link.get("rel") == "enclosure":
                url = link.get("href", "")
                link_type = link.get("type", "")
                if url and ("image" in link_type or self._looks_like_image(url)):
                    return url

        return ""

    def _looks_like_image(self, url: str) -> bool:
        """检查URL是否看起来像图片."""
        lower = url.lower()
        return any(ext in lower for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"])

    def _extract_content(self, entry: dict) -> str:
        """
        从RSS entry中提取最佳内容/摘要.

        优先级: content (if longer) > summary > description
        """
        summary = entry.get("summary", "") or entry.get("description", "") or ""

        # 检查content字段 (feedparser返回为list of dicts)
        content = ""
        if "content" in entry and entry["content"]:
            content_list = entry["content"]
            if isinstance(content_list, list) and content_list:
                content = content_list[0].get("value", "")

        # 返回更长的那个
        return content if len(content) > len(summary) else summary
