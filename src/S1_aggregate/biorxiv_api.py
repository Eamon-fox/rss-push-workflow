"""bioRxiv/medRxiv API fetcher.

Official API documentation: https://api.biorxiv.org/

This fetcher uses the official bioRxiv API to retrieve preprints,
which is more reliable than RSS feeds as it:
- Supports date range queries (no RSS expiration issues)
- Returns all subjects (no subscription gaps)
- Provides complete metadata
"""

import json
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from ..models import NewsItem, SourceType

API_BASE = "https://api.biorxiv.org/details"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ScholarPipe/1.0)"
}
RAW_DIR = Path("data/raw")
CACHE_MAX_AGE_HOURS = 12  # 每天最多跑 2 次


def _get_cache_path(source_name: str) -> Path:
    """Get cache file path for source."""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = source_name.lower().replace(" ", "_").replace("/", "_") + ".json"
    return RAW_DIR / today / filename


def _check_cache(source_name: str, max_age_hours: float = CACHE_MAX_AGE_HOURS) -> list[NewsItem] | None:
    """Check if valid cache exists."""
    cache_path = _get_cache_path(source_name)
    if not cache_path.exists():
        return None

    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    if age > timedelta(hours=max_age_hours):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [NewsItem(**item) for item in data]
    except Exception:
        return None


def _save_raw(source_name: str, items: list[NewsItem]) -> None:
    """Save raw results to data/raw/{date}/{source}.json"""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = RAW_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    filename = source_name.lower().replace(" ", "_").replace("/", "_") + ".json"
    filepath = dir_path / filename

    data = [item.model_dump() for item in items]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _parse_authors(authors_str: str) -> list[str]:
    """Parse authors string into list.

    API returns: "Smith, J.; Jones, A.; Wang, B."
    """
    if not authors_str:
        return []
    # Split by semicolon and clean up
    authors = [a.strip() for a in authors_str.split(";") if a.strip()]
    return authors


def _parse_date(date_str: str) -> datetime | None:
    """Parse date string from API (YYYY-MM-DD format)."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def fetch(source: dict, save_raw: bool = True) -> list[NewsItem]:
    """
    Fetch preprints from bioRxiv/medRxiv API.

    Args:
        source: {
            "name": "bioRxiv API",
            "type": "biorxiv_api",
            "server": "biorxiv" or "medrxiv",
            "days_back": 30,
            "subjects": ["biochemistry", "cell_biology"]  # optional, empty = all
        }
        save_raw: Whether to save raw results to disk

    Returns:
        List of NewsItem
    """
    name = source.get("name", "bioRxiv API")
    server: Literal["biorxiv", "medrxiv"] = source.get("server", "biorxiv")
    days_back = source.get("days_back", 30)
    subjects = source.get("subjects", [])  # Empty = all subjects

    # Check cache first
    cached = _check_cache(name)
    if cached is not None:
        cache_path = _get_cache_path(name)
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_min = (datetime.now() - mtime).total_seconds() / 60
        print(f"  [{name}] Cache hit ({len(cached)} items, {age_min:.0f}min ago)")
        return cached

    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    items: list[NewsItem] = []
    cursor = 0
    page_size = 100  # API max per request

    print(f"  [{name}] Fetching {server} papers from {start_date} to {end_date}...")

    while True:
        # API endpoint: /details/{server}/{start}/{end}/{cursor}
        url = f"{API_BASE}/{server}/{start_date}/{end_date}/{cursor}"

        try:
            resp = httpx.get(url, headers=HEADERS, timeout=60)
            if resp.status_code != 200:
                print(f"  [{name}] HTTP {resp.status_code}")
                break

            data = resp.json()
        except Exception as e:
            print(f"  [{name}] API error: {e}")
            break

        messages = data.get("messages", [])
        collection = data.get("collection", [])

        if not collection:
            break

        # Get total count from messages
        total = 0
        for msg in messages:
            if "total" in msg:
                total = int(msg["total"])
                break

        # Process papers
        for paper in collection:
            category = paper.get("category", "")

            # Filter by subject if specified
            if subjects and category not in subjects:
                continue

            doi = paper.get("doi", "")
            # Construct proper DOI link
            if doi and not doi.startswith("http"):
                link = f"https://www.{server}.org/content/{doi}"
            else:
                link = doi

            item = NewsItem(
                title=paper.get("title", "").strip(),
                content=paper.get("abstract", "").strip(),
                link=link,
                authors=_parse_authors(paper.get("authors", "")),
                doi=doi,
                source_type=SourceType.PREPRINT,
                source_name=name,
                source_url=f"{API_BASE}/{server}",
                journal_name=f"{server.title()} [{category}]",
                published_at=_parse_date(paper.get("date")),
            )
            items.append(item)

        # Check if we've fetched all
        cursor += len(collection)
        if cursor >= total:
            break

        # Progress indicator for large fetches
        if cursor % 500 == 0:
            print(f"  [{name}] Progress: {cursor}/{total}")

    print(f"  [{name}] Fetched {len(items)} papers" +
          (f" (filtered from {cursor} total)" if subjects else ""))

    # Save raw results
    if save_raw and items:
        _save_raw(name, items)

    return items


def fetch_by_doi(doi: str, server: str = "biorxiv") -> NewsItem | None:
    """
    Fetch a single paper by DOI.

    Args:
        doi: Paper DOI (e.g., "10.1101/2025.12.03.690598")
        server: "biorxiv" or "medrxiv"

    Returns:
        NewsItem or None if not found
    """
    # Handle different DOI formats
    if doi.startswith("10.1101/"):
        doi_query = doi
    elif doi.startswith("10.64898/"):
        # New bioRxiv DOI format
        doi_query = doi
    else:
        return None

    url = f"{API_BASE}/{server}/{doi_query}"

    try:
        resp = httpx.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None

        data = resp.json()
        collection = data.get("collection", [])

        if not collection:
            return None

        paper = collection[0]
        return NewsItem(
            title=paper.get("title", "").strip(),
            content=paper.get("abstract", "").strip(),
            link=f"https://www.{server}.org/content/{paper.get('doi', '')}",
            authors=_parse_authors(paper.get("authors", "")),
            doi=paper.get("doi", ""),
            source_type=SourceType.PREPRINT,
            source_name=f"{server.title()} API",
            journal_name=f"{server.title()} [{paper.get('category', '')}]",
            published_at=_parse_date(paper.get("date")),
        )
    except Exception:
        return None
