"""Step 1: Aggregate from multiple sources."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

from . import pubmed
from . import rss

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

    if not sources:
        return []

    max_workers = _resolve_max_workers(len(sources))
    ordered_results: dict[int, list] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_fetch_single_source, source, save_raw): (idx, source)
            for idx, source in enumerate(sources)
        }

        for future in as_completed(future_map):
            idx, source = future_map[future]
            name = source.get("name", f"source_{idx}")
            try:
                items = future.result() or []
            except Exception as exc:  # pragma: no cover
                print(f"  [{name}] Fetch failed: {exc}")
                items = []
            ordered_results[idx] = items
            if items:
                print(f"  [{name}] {len(items)} items")

    all_items: list = []
    for idx in range(len(sources)):
        items = ordered_results.get(idx, [])
        if items:
            all_items.extend(items)

    # Save combined raw
    if save_raw and all_items:
        _save_combined(all_items)

    return all_items


def _resolve_max_workers(total_sources: int) -> int:
    env_value = os.environ.get("AGGREGATE_MAX_WORKERS")
    try:
        configured = int(env_value) if env_value else total_sources  # 默认全并发
    except ValueError:
        configured = total_sources
    configured = max(1, configured)
    return min(configured, total_sources)


def _fetch_single_source(source: dict, save_raw: bool) -> list:
    """Fetch a single source (RSS or PubMed)."""
    source_type = source.get("type", "rss")
    if source_type == "rss":
        return rss.fetch(source, save_raw=save_raw)
    if source_type == "pubmed":
        return pubmed.fetch(source, save_raw=save_raw)
    raise ValueError(f"Unknown source type: {source_type}")


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
