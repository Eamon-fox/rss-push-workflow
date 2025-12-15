"""Cross-period deduplication filter."""

from ..models import NewsItem
from . import seen


def filter_unseen(
    items: list[NewsItem],
    seen_records: dict[str, str],
    window_hours: int = 72
) -> list[NewsItem]:
    """
    Filter out items seen in previous runs.

    Args:
        items: Items from current run
        seen_records: Historical {hash: timestamp} records
        window_hours: Time window for dedup

    Returns:
        Items not seen in recent history
    """
    new_items = []

    for item in items:
        if not seen.is_seen(seen_records, item.content_hash, window_hours):
            seen.mark_seen(seen_records, item.content_hash)
            new_items.append(item)

    return new_items
