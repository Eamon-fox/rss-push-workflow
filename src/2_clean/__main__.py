"""Run clean module standalone - reads from step 1 output."""

import json
from pathlib import Path
from datetime import datetime

from ..models import NewsItem
from . import batch_clean

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

    # Clean
    cleaned = batch_clean(items)
    print(f"Cleaned {len(cleaned)} items")

    # Show sample
    if cleaned:
        print(f"\nSample (before → after):")
        print(f"  content: {items[0].content[:80]}...")
        print(f"       →   {cleaned[0].content[:80]}...")


if __name__ == "__main__":
    main()
