"""Step 1: Aggregate from multiple sources."""

import yaml
from pathlib import Path

from . import rss

__all__ = ["fetch_all", "load_sources"]

SOURCES_FILE = Path(__file__).parent / "sources.yaml"


def load_sources() -> list[dict]:
    """Load sources from yaml config."""
    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_all(sources: list[dict] | None = None) -> list:
    """
    Fetch from all sources.

    Args:
        sources: Optional source list. If None, loads from sources.yaml

    Returns:
        Combined list of NewsItem
    """
    if sources is None:
        sources = load_sources()

    all_items = []

    for source in sources:
        source_type = source.get("type", "rss")

        if source_type == "rss":
            items = rss.fetch(source)
            if items:
                print(f"  [{source['name']}] {len(items)} items")
                all_items.extend(items)
        # 后续扩展:
        # elif source_type == "pubmed":
        #     items = pubmed.fetch(source)

    return all_items


if __name__ == "__main__":
    items = fetch_all()
    print(f"\nTotal: {len(items)} items")
