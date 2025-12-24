"""OpenAlex API fetcher.

OpenAlex is a free and open catalog of the world's scholarly works.
API documentation: https://docs.openalex.org/

Key features:
- 250M+ works, completely free
- Rich filtering options (concepts, institutions, authors, journals)
- No API key required (but polite pool with email recommended)
"""

import json
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

from ..models import NewsItem, SourceType

API_BASE = "https://api.openalex.org/works"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ScholarPipe/1.0; mailto:scholarpipe@example.com)"
}
RAW_DIR = Path("data/raw")
CACHE_MAX_AGE_HOURS = 12  # 每天最多跑 2 次

# Common concept IDs for biology/life sciences
# Find more at: https://api.openalex.org/concepts?search=
CONCEPT_IDS = {
    # Molecular/Cell Biology
    "molecular_biology": "C104317684",
    "cell_biology": "C95444343",
    "biochemistry": "C55493867",
    "genetics": "C54355233",
    "genomics": "C70721500",

    # RNA/DNA
    "rna": "C95444343",  # Actually maps to broader concept
    "gene_expression": "C178663694",

    # Neuroscience
    "neuroscience": "C134018914",
    "neurobiology": "C89423630",

    # Immunology
    "immunology": "C203014093",
    "immune_system": "C203014093",  # Alias for immunology

    # Cancer
    "cancer_research": "C126322002",
    "oncology": "C126322002",
    "cancer": "C126322002",  # Alias
    "tumor": "C126322002",   # Alias

    # Developmental
    "developmental_biology": "C9390403",
    "embryology": "C9390403",  # Maps to developmental_biology
    "stem_cell": "C29456083",  # Stem cell research

    # Microbiology
    "microbiology": "C89423630",
    "virology": "C98274493",
    "bacteriology": "C89423630",  # Maps to microbiology

    # Pharmacology
    "pharmacology": "C89423630",
    "drug_discovery": "C2779356",
    "toxicology": "C58123633",

    # Bioinformatics
    "bioinformatics": "C60644358",
    "computational_biology": "C6557445",
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


def _parse_authors(authorships: list) -> list[str]:
    """Parse authors from OpenAlex authorships structure."""
    authors = []
    for authorship in authorships[:10]:  # Limit to first 10 authors
        author = authorship.get("author", {})
        name = author.get("display_name", "")
        if name:
            authors.append(name)
    return authors


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse publication date."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        # Try year-only
        try:
            return datetime.strptime(date_str[:4], "%Y")
        except ValueError:
            return None


def _get_source_type(work: dict) -> SourceType:
    """Determine source type from OpenAlex work."""
    work_type = work.get("type", "")

    if work_type == "preprint":
        return SourceType.PREPRINT
    elif work_type in ("article", "review", "editorial"):
        return SourceType.JOURNAL
    else:
        return SourceType.JOURNAL  # Default


def _build_filter(source: dict) -> str:
    """Build OpenAlex filter string from source config."""
    filters = []

    # Date range
    days_back = source.get("days_back", 30)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    filters.append(f"from_publication_date:{from_date}")

    # Concepts (OR logic within concepts)
    concepts = source.get("concepts", [])
    if concepts:
        # Convert named concepts to IDs
        concept_ids = []
        for c in concepts:
            if c.startswith("C"):
                concept_ids.append(c)
            elif c.lower() in CONCEPT_IDS:
                concept_ids.append(CONCEPT_IDS[c.lower()])
            else:
                concept_ids.append(c)

        if concept_ids:
            filters.append(f"concepts.id:{'|'.join(concept_ids)}")

    # Institution filter (optional)
    institutions = source.get("institutions", [])
    if institutions:
        filters.append(f"authorships.institutions.id:{'|'.join(institutions)}")

    # Type filter (optional)
    work_types = source.get("types", [])
    if work_types:
        filters.append(f"type:{'|'.join(work_types)}")

    # Open access filter (optional)
    if source.get("open_access_only", False):
        filters.append("is_oa:true")

    # Must have abstract
    filters.append("has_abstract:true")

    return ",".join(filters)


def fetch(source: dict, save_raw: bool = True) -> list[NewsItem]:
    """
    Fetch works from OpenAlex API.

    Args:
        source: {
            "name": "OpenAlex Biology",
            "type": "openalex",
            "days_back": 30,
            "concepts": ["molecular_biology", "cell_biology"],  # or concept IDs
            "per_page": 200,
            "max_results": 500,  # optional limit
            "open_access_only": false,
            "types": ["article", "preprint"]  # optional
        }
        save_raw: Whether to save raw results to disk

    Returns:
        List of NewsItem
    """
    name = source.get("name", "OpenAlex")
    per_page = min(source.get("per_page", 200), 200)  # API max is 200
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
    filter_str = _build_filter(source)

    items: list[NewsItem] = []
    cursor = "*"  # OpenAlex uses cursor-based pagination

    print(f"  [{name}] Fetching with filter: {filter_str[:100]}...")

    while len(items) < max_results:
        params = {
            "filter": filter_str,
            "per_page": per_page,
            "cursor": cursor,
            "select": "id,doi,title,authorships,publication_date,type,primary_location,abstract_inverted_index",
        }

        url = f"{API_BASE}?{urlencode(params)}"

        try:
            resp = httpx.get(url, headers=HEADERS, timeout=60)
            if resp.status_code != 200:
                print(f"  [{name}] HTTP {resp.status_code}: {resp.text[:200]}")
                break

            data = resp.json()
        except Exception as e:
            print(f"  [{name}] API error: {e}")
            break

        results = data.get("results", [])
        meta = data.get("meta", {})

        if not results:
            break

        for work in results:
            # Reconstruct abstract from inverted index
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

            # Get DOI
            doi = work.get("doi", "") or ""
            if doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")

            # Get journal/source info
            primary_location = work.get("primary_location") or {}
            source_info = primary_location.get("source") or {}
            journal_name = source_info.get("display_name", "")

            # Get link
            link = work.get("doi") or work.get("id", "")

            item = NewsItem(
                title=work.get("title", "").strip() if work.get("title") else "",
                content=abstract,
                link=link,
                authors=_parse_authors(work.get("authorships", [])),
                doi=doi,
                source_type=_get_source_type(work),
                source_name=name,
                source_url="https://api.openalex.org",
                journal_name=journal_name,
                published_at=_parse_date(work.get("publication_date")),
            )

            # Skip items without title
            if item.title:
                items.append(item)

        # Get next cursor
        cursor = meta.get("next_cursor")
        if not cursor:
            break

        # Progress indicator
        total_count = meta.get("count", 0)
        if len(items) % 500 == 0:
            print(f"  [{name}] Progress: {len(items)}/{min(total_count, max_results)}")

    print(f"  [{name}] Fetched {len(items)} works")

    # Save raw results
    if save_raw and items:
        _save_raw(name, items)

    return items


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """
    Reconstruct abstract from OpenAlex inverted index format.

    OpenAlex stores abstracts as inverted index to save space:
    {"word1": [0, 5], "word2": [1, 3]} -> positions of each word
    """
    if not inverted_index:
        return ""

    try:
        # Find max position to size the array
        max_pos = 0
        for positions in inverted_index.values():
            if positions:
                max_pos = max(max_pos, max(positions))

        # Reconstruct
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word

        return " ".join(words)
    except Exception:
        return ""


def search(query: str, days_back: int = 30, max_results: int = 100) -> list[NewsItem]:
    """
    Search OpenAlex by text query.

    Args:
        query: Search text
        days_back: How many days back to search
        max_results: Maximum results to return

    Returns:
        List of NewsItem
    """
    source = {
        "name": f"OpenAlex Search: {query[:30]}",
        "days_back": days_back,
        "max_results": max_results,
    }

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "search": query,
        "filter": f"from_publication_date:{from_date},has_abstract:true",
        "per_page": min(max_results, 200),
        "select": "id,doi,title,authorships,publication_date,type,primary_location,abstract_inverted_index",
    }

    url = f"{API_BASE}?{urlencode(params)}"

    try:
        resp = httpx.get(url, headers=HEADERS, timeout=60)
        if resp.status_code != 200:
            return []

        data = resp.json()
        results = data.get("results", [])

        items = []
        for work in results:
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
            doi = work.get("doi", "") or ""
            if doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")

            primary_location = work.get("primary_location") or {}
            source_info = primary_location.get("source") or {}

            item = NewsItem(
                title=work.get("title", "").strip() if work.get("title") else "",
                content=abstract,
                link=work.get("doi") or work.get("id", ""),
                authors=_parse_authors(work.get("authorships", [])),
                doi=doi,
                source_type=_get_source_type(work),
                source_name="OpenAlex Search",
                source_url="https://api.openalex.org",
                journal_name=source_info.get("display_name", ""),
                published_at=_parse_date(work.get("publication_date")),
            )

            if item.title:
                items.append(item)

        return items
    except Exception:
        return []
