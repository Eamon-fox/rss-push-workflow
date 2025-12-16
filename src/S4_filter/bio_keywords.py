"""Bio keyword filter - requires at least one biology-related term.

Kept for backward-compatibility; the hybrid filter lives in `hybrid.py`.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import NewsItem

FILTER_DIR = Path("data/filtered")

from .config import GENERAL_BIO_KEYWORDS

# Backward compatible name
BIO_KEYWORDS = GENERAL_BIO_KEYWORDS


def has_bio_keyword(title: str, content: str) -> tuple[bool, list[str]]:
    """
    Check if title or content contains any bio keyword.

    Returns:
        Tuple of (has_keyword, matched_keywords)
    """
    text = f"{title} {content}".lower()
    matched = []
    for kw in BIO_KEYWORDS:
        if kw in text:
            matched.append(kw)
    return len(matched) > 0, matched


def filter_bio(
    items: "list[NewsItem]",
    save: bool = True,
) -> tuple["list[NewsItem]", "list[NewsItem]"]:
    """
    Filter items by bio keyword presence.

    Args:
        items: Items to filter
        save: Whether to save results to disk

    Returns:
        Tuple of (passed_items, filtered_items)
    """
    passed = []
    filtered = []

    for item in items:
        has_kw, _ = has_bio_keyword(item.title, item.content)
        if has_kw:
            passed.append(item)
        else:
            filtered.append(item)

    if save and passed:
        _save_filtered(passed)

    return passed, filtered


def _save_filtered(items: "list[NewsItem]") -> None:
    """Save filtered results to data/filtered/{date}/all.json"""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = FILTER_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    filepath = dir_path / "all.json"
    data = [item.model_dump() for item in items]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved filtered to {filepath}")
