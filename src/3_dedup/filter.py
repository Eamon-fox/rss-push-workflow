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

    Returns:
        Deduplicated items (first occurrence kept)
    """
    seen_fps = set()
    unique_items = []

    for item in items:
        fp = get_fingerprint(item)

        if not fp:
            continue

        if fp not in seen_fps:
            seen_fps.add(fp)
            unique_items.append(item)

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
