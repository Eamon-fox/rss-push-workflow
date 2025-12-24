"""Europe PMC API fetcher.

Europe PMC is a comprehensive database of life science publications.
API documentation: https://europepmc.org/RestfulWebService

Key features:
- Includes PubMed, PMC, and preprint servers (bioRxiv, medRxiv, Research Square, etc.)
- Faster preprint indexing than PubMed
- Full-text search capability
- Rich query syntax
"""

import json
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

from ..models import NewsItem, SourceType

API_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ScholarPipe/1.0)"
}
RAW_DIR = Path("data/raw")
CACHE_MAX_AGE_HOURS = 12  # 每天最多跑 2 次

# Source type mapping
SOURCE_MAP = {
    "PPR": SourceType.PREPRINT,   # Preprint
    "MED": SourceType.PUBMED,     # PubMed/MEDLINE
    "PMC": SourceType.JOURNAL,    # PubMed Central
    "PAT": SourceType.JOURNAL,    # Patents
    "AGR": SourceType.JOURNAL,    # Agricola
    "CBA": SourceType.JOURNAL,    # Chinese Biological Abstracts
    "CTX": SourceType.JOURNAL,    # CiteXplore
    "ETH": SourceType.JOURNAL,    # EthOs
    "HIR": SourceType.JOURNAL,    # NHS Evidence
    "NBK": SourceType.JOURNAL,    # NCBI Bookshelf
}


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


def _parse_authors(result: dict) -> list[str]:
    """Parse authors from Europe PMC result."""
    author_string = result.get("authorString", "")
    if not author_string:
        return []

    # Author string format: "Author1 A, Author2 B, Author3 C"
    authors = [a.strip() for a in author_string.split(",") if a.strip()]
    return authors[:10]  # Limit to first 10


def _parse_date(result: dict) -> datetime | None:
    """Parse publication date from various fields."""
    # Try first publication date
    first_pub = result.get("firstPublicationDate")
    if first_pub:
        try:
            return datetime.strptime(first_pub, "%Y-%m-%d")
        except ValueError:
            pass

    # Try publication date
    pub_date = result.get("pubDate")
    if pub_date:
        try:
            return datetime.strptime(pub_date, "%Y-%m-%d")
        except ValueError:
            pass

    # Try year only
    pub_year = result.get("pubYear")
    if pub_year:
        try:
            return datetime(int(pub_year), 1, 1)
        except ValueError:
            pass

    return None


def _get_source_type(result: dict) -> SourceType:
    """Determine source type from Europe PMC result."""
    source = result.get("source", "")
    return SOURCE_MAP.get(source, SourceType.JOURNAL)


def _build_query(source: dict) -> str:
    """Build Europe PMC query string from source config."""
    parts = []

    # Base query (if provided)
    query = source.get("query", "")
    if query:
        parts.append(f"({query})")

    # Date range
    days_back = source.get("days_back", 30)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")
    parts.append(f"FIRST_PDATE:[{from_date} TO {to_date}]")

    # Source filter (e.g., preprints only)
    sources = source.get("sources", [])
    if sources:
        source_filter = " OR ".join([f"SRC:{s}" for s in sources])
        parts.append(f"({source_filter})")

    # Publication type filter
    pub_types = source.get("pub_types", [])
    if pub_types:
        type_filter = " OR ".join([f"PUB_TYPE:{t}" for t in pub_types])
        parts.append(f"({type_filter})")

    return " AND ".join(parts)


def fetch(source: dict, save_raw: bool = True) -> list[NewsItem]:
    """
    Fetch articles from Europe PMC API.

    Args:
        source: {
            "name": "Europe PMC Preprints",
            "type": "europepmc",
            "days_back": 30,
            "query": "RNA splicing",  # optional search terms
            "sources": ["PPR"],  # PPR=preprints, MED=PubMed, PMC=PMC
            "pub_types": ["preprint"],  # optional
            "page_size": 100,
            "max_results": 500
        }
        save_raw: Whether to save raw results to disk

    Returns:
        List of NewsItem
    """
    name = source.get("name", "Europe PMC")
    page_size = min(source.get("page_size", 100), 1000)  # API max is 1000
    max_results = source.get("max_results", 1000)

    # Check cache first
    cached = _check_cache(name)
    if cached is not None:
        cache_path = _get_cache_path(name)
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_min = (datetime.now() - mtime).total_seconds() / 60
        print(f"  [{name}] Cache hit ({len(cached)} items, {age_min:.0f}min ago)")
        return cached

    # Build query
    query = _build_query(source)

    items: list[NewsItem] = []
    cursor_mark = "*"

    print(f"  [{name}] Searching: {query[:80]}...")

    while len(items) < max_results:
        params = {
            "query": query,
            "resultType": "core",
            "pageSize": page_size,
            "cursorMark": cursor_mark,
            "format": "json",
            "sort": "P_PDATE_D desc",  # Most recent first (Europe PMC sort syntax)
        }

        # Use safe=':[]' to prevent encoding of special chars in query
        url = f"{API_BASE}?{urlencode(params, safe=':[]')}"

        try:
            resp = httpx.get(url, headers=HEADERS, timeout=60)
            if resp.status_code != 200:
                print(f"  [{name}] HTTP {resp.status_code}")
                break

            data = resp.json()
        except Exception as e:
            print(f"  [{name}] API error: {e}")
            break

        result_list = data.get("resultList", {}).get("result", [])
        next_cursor = data.get("nextCursorMark")

        if not result_list:
            break

        for result in result_list:
            # Get DOI
            doi = result.get("doi", "") or ""

            # Get link
            if doi:
                link = f"https://doi.org/{doi}"
            else:
                # Fallback to Europe PMC link
                pmid = result.get("pmid")
                pmcid = result.get("pmcid")
                ppr_id = result.get("id")

                if pmcid:
                    link = f"https://europepmc.org/article/PMC/{pmcid}"
                elif pmid:
                    link = f"https://europepmc.org/article/MED/{pmid}"
                elif ppr_id:
                    link = f"https://europepmc.org/article/PPR/{ppr_id}"
                else:
                    link = ""

            # Get journal name
            journal = result.get("journalTitle", "") or result.get("bookOrReportDetails", {}).get("publisher", "")

            # For preprints, add preprint server info
            source_type = _get_source_type(result)
            if source_type == SourceType.PREPRINT:
                preprint_source = result.get("commentCorrectionList", {})
                if not journal:
                    journal = "Preprint"

            item = NewsItem(
                title=result.get("title", "").strip(),
                content=result.get("abstractText", "").strip(),
                link=link,
                authors=_parse_authors(result),
                doi=doi,
                source_type=source_type,
                source_name=name,
                source_url="https://europepmc.org",
                journal_name=journal,
                published_at=_parse_date(result),
            )

            # Skip items without title
            if item.title:
                items.append(item)

        # Check for next page
        if not next_cursor or next_cursor == cursor_mark:
            break
        cursor_mark = next_cursor

        # Progress indicator
        hit_count = data.get("hitCount", 0)
        if len(items) % 500 == 0:
            print(f"  [{name}] Progress: {len(items)}/{min(hit_count, max_results)}")

    print(f"  [{name}] Fetched {len(items)} articles")

    # Save raw results
    if save_raw and items:
        _save_raw(name, items)

    return items


def search_preprints(query: str = "", days_back: int = 30, max_results: int = 500) -> list[NewsItem]:
    """
    Convenience function to search preprints only.

    Args:
        query: Optional search terms
        days_back: How many days back to search
        max_results: Maximum results to return

    Returns:
        List of NewsItem
    """
    source = {
        "name": f"Europe PMC Preprints" + (f": {query[:20]}" if query else ""),
        "days_back": days_back,
        "query": query,
        "sources": ["PPR"],  # Preprints only
        "max_results": max_results,
    }
    return fetch(source, save_raw=False)


def search_pubmed_recent(query: str = "", days_back: int = 7, max_results: int = 200) -> list[NewsItem]:
    """
    Convenience function to search recent PubMed articles.
    Can be used as a fallback/supplement to NCBI E-utilities.

    Args:
        query: Optional search terms
        days_back: How many days back to search
        max_results: Maximum results to return

    Returns:
        List of NewsItem
    """
    source = {
        "name": f"Europe PMC/PubMed" + (f": {query[:20]}" if query else ""),
        "days_back": days_back,
        "query": query,
        "sources": ["MED"],  # PubMed/MEDLINE only
        "max_results": max_results,
    }
    return fetch(source, save_raw=False)
