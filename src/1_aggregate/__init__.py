"""Step 1: Aggregate news from multiple sources."""

from .rss import fetch_rss

__all__ = ["fetch_rss", "fetch_all"]


def fetch_all(sources: list[dict]) -> list:
    """
    Fetch from all configured sources.

    Args:
        sources: List of source configs

    Returns:
        Combined list of NewsItem
    """
    from ..models import NewsItem

    all_items = []

    for source in sources:
        source_type = source.get("type", "rss")

        if source_type == "rss":
            items = fetch_rss(source["url"], source["name"])
            if items:
                all_items.extend(items)

    return all_items
