"""RSS feed fetcher."""

from ..models import NewsItem


def fetch_rss(url: str, source_name: str) -> list[NewsItem]:
    """
    Fetch items from a single RSS feed.

    Args:
        url: RSS feed URL
        source_name: Display name for the source

    Returns:
        List of NewsItem objects
    """
    # TODO: Implement with feedparser
    # import feedparser
    # feed = feedparser.parse(url)
    # items = []
    # for entry in feed.entries:
    #     items.append(NewsItem(
    #         title=entry.title,
    #         content=entry.get("summary", ""),
    #         link=entry.link,
    #         source_name=source_name,
    #         source_url=url,
    #     ))
    # return items
    pass
