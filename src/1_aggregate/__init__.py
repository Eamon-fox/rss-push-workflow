"""Step 1: Aggregate from multiple sources."""

import json
import yaml
from datetime import datetime
from pathlib import Path

from . import rss
from . import pubmed

__all__ = ["fetch_all", "load_sources"]

SOURCES_FILE = Path(__file__).parent / "sources.yaml"
RAW_DIR = Path("data/raw")


def load_sources() -> list[dict]:
    """Load sources from yaml config."""
    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_all(sources: list[dict] | None = None, save_raw: bool = True) -> list:
    """
    Fetch from all sources and save raw results.

    Args:
        sources: Optional source list. If None, loads from sources.yaml
        save_raw: Whether to save raw results to disk

    Returns:
        Combined list of NewsItem
    """
    if sources is None:
        sources = load_sources()

    all_items = []

    for source in sources:
        source_type = source.get("type", "rss")

        if source_type == "rss":
            items = rss.fetch(source, save_raw=save_raw)
            if items:
                print(f"  [{source['name']}] {len(items)} items")
                all_items.extend(items)
        elif source_type == "pubmed":
            items = pubmed.fetch(source, save_raw=save_raw)
            if items:
                print(f"  [{source['name']}] {len(items)} items")
                all_items.extend(items)

    # Save combined raw
    if save_raw and all_items:
        _save_combined(all_items)

    return all_items


def _save_combined(items: list) -> None:
    """Save combined results to data/raw/{date}/all.json"""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = RAW_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    filepath = dir_path / "all.json"
    data = [item.model_dump() for item in items]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved to {filepath}")


if __name__ == "__main__":
    items = fetch_all()
    print(f"\nTotal: {len(items)} items")
