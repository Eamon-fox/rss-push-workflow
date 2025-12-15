"""RSS feed fetcher - universal for all RSS sources."""

import feedparser
from ..models import NewsItem


def fetch(source: dict) -> list[NewsItem]:
    """
    Fetch from RSS source.

    Args:
        source: {"name": "...", "url": "..."}

    Returns:
        List of NewsItem
    """
    url = source["url"]
    name = source["name"]

    feed = feedparser.parse(url)

    items = []
    for entry in feed.entries:
        items.append(NewsItem(
            title=entry.get("title", ""),
            content=entry.get("summary", entry.get("description", "")),
            link=entry.get("link", ""),
            source_name=name,
            source_url=url,
        ))

    return items
