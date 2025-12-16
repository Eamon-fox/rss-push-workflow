"""Deduplication filter using fingerprints."""

import json
from datetime import datetime
from pathlib import Path

from ..models import NewsItem
from .fingerprint import get_fingerprint
from . import seen

DEDUP_DIR = Path("data/deduped")


def filter_unseen(
    items: list[NewsItem],
    seen_records: dict[str, str],
    save: bool = True,
) -> tuple[list[NewsItem], list[str]]:
    """
    Filter out items already seen.

    Args:
        items: Items from current run
        seen_records: Historical {fingerprint: timestamp} records
        save: Whether to save results to disk

    Returns:
        Tuple of (new_items, new_fingerprints)
        - new_items: Items not seen before
        - new_fingerprints: Their fingerprints (for later recording)
    """
    new_items = []
    new_fingerprints = []

    for item in items:
        fp = get_fingerprint(item)

        if not fp:
            # Cannot compute fingerprint, skip
            continue

        if not seen.is_seen(seen_records, fp):
            new_items.append(item)
            new_fingerprints.append(fp)

    if save and new_items:
        _save_deduped(new_items)

    return new_items, new_fingerprints


def filter_duplicates_in_batch(items: list[NewsItem]) -> list[NewsItem]:
    """
    Remove duplicates within current batch (same fingerprint).
    When duplicates found, keep the one with longer content.

    Returns:
        Deduplicated items (longer content preferred)
    """
    # Map fingerprint -> (index in unique_items, content_length)
    fp_to_idx: dict[str, int] = {}
    unique_items: list[NewsItem] = []

    for item in items:
        fp = get_fingerprint(item)

        if not fp:
            continue

        content_len = len(item.content or "")

        if fp not in fp_to_idx:
            # First occurrence
            fp_to_idx[fp] = len(unique_items)
            unique_items.append(item)
        else:
            # Duplicate found - keep the one with longer content
            existing_idx = fp_to_idx[fp]
            existing_len = len(unique_items[existing_idx].content or "")

            if content_len > existing_len:
                # Replace with longer version
                unique_items[existing_idx] = item

    return unique_items


def _save_deduped(items: list[NewsItem]) -> None:
    """Save deduped results to data/deduped/{date}/all.json"""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = DEDUP_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    filepath = dir_path / "all.json"
    data = [item.model_dump() for item in items]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved deduped to {filepath}")
