"""RSS feed fetcher."""

import feedparser
from ..models import NewsItem


def fetch_rss(url: str, source_name: str) -> list[NewsItem]:
    """
    Fetch items from RSS feed.

    Args:
        url: RSS feed URL
        source_name: Display name for source

    Returns:
        List of NewsItem
    """
    feed = feedparser.parse(url)

    items = []
    for entry in feed.entries:
        items.append(NewsItem(
            title=entry.get("title", ""),
            content=entry.get("summary", entry.get("description", "")),
            link=entry.get("link", ""),
            source_name=source_name,
            source_url=url,
        ))

    return items


if __name__ == "__main__":
    # Test with Nature Neuroscience
    items = fetch_rss(
        "https://www.nature.com/neuro.rss",
        "Nature Neuroscience"
    )
    print(f"Found {len(items)} items")
    for item in items[:3]:
        print(f"- {item.title[:60]}...")
        print(f"  {item.link}")
        print()
