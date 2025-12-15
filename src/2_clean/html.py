"""HTML and text cleaning utilities."""

import re
import html
import json
from datetime import datetime
from pathlib import Path

from ..models import NewsItem

CLEAN_DIR = Path("data/cleaned")


def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def clean_whitespace(text: str) -> str:
    """Normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate(text: str, max_length: int = 2000) -> str:
    """Truncate text to max length."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def clean_text(text: str, max_length: int = 2000) -> str:
    """Full text cleaning pipeline."""
    text = clean_html(text)
    text = clean_whitespace(text)
    text = truncate(text, max_length)
    return text


def clean(item: NewsItem) -> NewsItem:
    """Clean a single news item."""
    return NewsItem(
        title=clean_whitespace(item.title),
        content=clean_text(item.content),
        link=item.link.strip(),
        authors=item.authors,
        doi=item.doi.strip() if item.doi else "",
        source_name=item.source_name,
        source_url=item.source_url,
        score=item.score,
        summary=item.summary,
        status=item.status,
        fetched_at=item.fetched_at,
        published_at=item.published_at,
    )


def batch_clean(items: list[NewsItem], save: bool = True) -> list[NewsItem]:
    """Clean multiple items."""
    cleaned = [clean(item) for item in items]
    if save and cleaned:
        _save_cleaned(cleaned)
    return cleaned


def _save_cleaned(items: list[NewsItem]) -> None:
    """Save cleaned results to data/cleaned/{date}/all.json"""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = CLEAN_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    filepath = dir_path / "all.json"
    data = [item.model_dump() for item in items]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved cleaned to {filepath}")
