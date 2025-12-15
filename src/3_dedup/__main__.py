"""Run dedup module standalone - reads from step 2 output."""

import json
from pathlib import Path
from datetime import datetime

from ..models import NewsItem
from . import load, filter_unseen, filter_duplicates_in_batch, get_fingerprint

RAW_DIR = Path("data/raw")


def main():
    # Load from step 1 output
    today = datetime.now().strftime("%Y-%m-%d")
    input_file = RAW_DIR / today / "all.json"

    if not input_file.exists():
        print(f"No input file: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = [NewsItem(**d) for d in data]
    print(f"Loaded {len(items)} items")

    # Show fingerprint examples
    print(f"\nFingerprint examples:")
    for item in items[:5]:
        fp = get_fingerprint(item)
        fp_type = "DOI" if item.doi else "Title hash"
        print(f"  [{fp_type}] {fp[:40]}... <- {item.title[:40]}...")

    # Step 1: Remove duplicates within batch
    unique = filter_duplicates_in_batch(items)
    print(f"\nAfter in-batch dedup: {len(unique)} items (removed {len(items) - len(unique)})")

    # Step 2: Filter against seen.json
    seen_records = load()
    print(f"Seen records loaded: {len(seen_records)} fingerprints")

    new_items, new_fps = filter_unseen(unique, seen_records)
    print(f"After cross-period dedup: {len(new_items)} new items")

    # Show what would be recorded
    print(f"\nNew fingerprints to record: {len(new_fps)}")
    if new_fps:
        print(f"  First 3: {new_fps[:3]}")


if __name__ == "__main__":
    main()
