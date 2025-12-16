"""RSS fetcher with parser support."""

import json
import httpx
import feedparser
from email.utils import parsedate_to_datetime
from datetime import datetime
from pathlib import Path

from ..models import NewsItem
from .parsers import get_parser

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

RAW_DIR = Path("data/raw")


def fetch(source: dict, save_raw: bool = True) -> list[NewsItem]:
    """
    Fetch and parse RSS source.

    Args:
        source: {"name": "...", "url": "..."}
        save_raw: Whether to save raw results to disk

    Returns:
        List of NewsItem
    """
    url = source["url"]
    name = source["name"]

    # Fetch
    try:
        resp = httpx.get(url, headers=HEADERS, follow_redirects=True, timeout=30)
        if resp.status_code != 200:
            print(f"  [{name}] HTTP {resp.status_code}")
            return []
        content = resp.text
    except Exception as e:
        print(f"  [{name}] Fetch error: {e}")
        return []

    # Parse
    feed = feedparser.parse(content)

    # Get parser for this source
    parser_cls = get_parser(name)
    if parser_cls:
        parser = parser_cls(name, url)
        items = parser.parse_feed(feed.entries)
    else:
        # Fallback: generic parsing
        items = _generic_parse(feed.entries, name, url)

    # Optional enrichment (e.g. DOI -> PubMed abstract)
    if source.get("enrich_via_pubmed") and items:
        try:
            from . import pubmed

            items = pubmed.enrich_items_by_doi(
                items,
                min_content_len=int(source.get("enrich_min_content_len", 80)),
            )
        except Exception as e:
            print(f"  [{name}] PubMed enrich error: {e}")

    # Save raw results
    if save_raw and items:
        _save_raw(name, items)

    return items


def _generic_parse(entries: list, source_name: str, source_url: str) -> list[NewsItem]:
    """Generic RSS parsing fallback."""
    items = []
    for entry in entries:
        published_at = _extract_published_at(entry)
        items.append(NewsItem(
            title=entry.get("title", ""),
            content=entry.get("summary", entry.get("description", "")),
            link=entry.get("link", ""),
            source_name=source_name,
            source_url=source_url,
            published_at=published_at,
        ))
    return items


def _extract_published_at(entry: dict) -> datetime | None:
    published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if published_parsed:
        try:
            return datetime(*published_parsed[:6])
        except Exception:
            pass

    published_text = entry.get("published") or entry.get("updated")
    if published_text:
        try:
            dt = parsedate_to_datetime(published_text)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except Exception:
            pass

    return None


def _save_raw(source_name: str, items: list[NewsItem]) -> None:
    """Save raw results to data/raw/{date}/{source}.json"""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = RAW_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    filename = source_name.lower().replace(" ", "_") + ".json"
    filepath = dir_path / filename

    data = [item.model_dump() for item in items]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
