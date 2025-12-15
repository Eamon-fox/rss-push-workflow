"""Step 1: Aggregate news from multiple sources."""

from datetime import datetime

from .models import NewsItem


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


def fetch_pubmed(query: str, max_results: int = 50) -> list[NewsItem]:
    """
    Fetch items from PubMed.

    Args:
        query: PubMed search query
        max_results: Maximum number of results

    Returns:
        List of NewsItem objects
    """
    # TODO: Implement with E-utilities API
    pass


def fetch_all(sources: list[dict]) -> list[NewsItem]:
    """
    Fetch from all configured sources.

    Args:
        sources: List of source configs, e.g.:
            [{"type": "rss", "url": "...", "name": "Nature"}]

    Returns:
        Combined list of NewsItem objects
    """
    all_items = []

    for source in sources:
        source_type = source.get("type", "rss")

        if source_type == "rss":
            items = fetch_rss(source["url"], source["name"])
        elif source_type == "pubmed":
            items = fetch_pubmed(source.get("query", ""))
        else:
            continue

        if items:
            all_items.extend(items)

    return all_items
