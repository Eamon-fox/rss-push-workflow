"""Run filter module standalone - reads from step 3 output."""

import json
from pathlib import Path
from datetime import datetime

from ..models import NewsItem
from . import filter_bio, BIO_KEYWORDS

DEDUP_DIR = Path("data/deduped")


def main():
    # Load from step 3 output
    today = datetime.now().strftime("%Y-%m-%d")
    input_file = DEDUP_DIR / today / "all.json"

    if not input_file.exists():
        print(f"No input file: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = [NewsItem(**d) for d in data]
    print(f"Loaded {len(items)} items from {input_file}")
    print(f"Bio keywords count: {len(BIO_KEYWORDS)}")

    # Filter
    passed, filtered = filter_bio(items)

    print(f"\nFilter results:")
    print(f"  Passed: {len(passed)} ({len(passed)/len(items)*100:.1f}%)")
    print(f"  Filtered: {len(filtered)} ({len(filtered)/len(items)*100:.1f}%)")

    # Show filtered items
    print(f"\n=== FILTERED (non-bio) ===")
    for item in filtered[:20]:
        print(f"  - [{item.source_name}] {item.title[:60]}...")

    if len(filtered) > 20:
        print(f"  ... and {len(filtered) - 20} more")


if __name__ == "__main__":
    main()
