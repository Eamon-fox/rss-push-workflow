"""HTML and text cleaning utilities."""

import re
import html

from ..models import NewsItem


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
        source_name=item.source_name,
        source_url=item.source_url,
        score=item.score,
        summary=item.summary,
        status=item.status,
        fetched_at=item.fetched_at,
    )


def batch_clean(items: list[NewsItem]) -> list[NewsItem]:
    """Clean multiple items."""
    return [clean(item) for item in items]
