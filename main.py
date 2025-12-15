"""ScholarPipe - Academic news aggregation and AI digest."""

import os

from src import aggregate
from src import clean
from src import dedup
from src import seen
from src import llm_process
from src import deliver


def run():
    """Run the complete ScholarPipe pipeline."""

    # Config (TODO: load from settings.yaml)
    sources = [
        {"type": "rss", "url": "https://www.nature.com/neuro.rss", "name": "Nature Neuroscience"},
    ]
    api_key = os.getenv("DEEPSEEK_API_KEY")
    score_threshold = 6.0

    # Load history for cross-period dedup
    seen_records = seen.load()

    # 1. Aggregate
    print("[1/5] Aggregating from sources...")
    items = aggregate.fetch_all(sources)
    total = len(items)
    print(f"      Found {total} items")

    # 2. Clean
    print("[2/5] Cleaning content...")
    items = clean.batch_clean(items)

    # 3. Cross-period dedup (against history)
    print("[3/5] Cross-period dedup...")
    items = dedup.filter_unseen(items, seen_records)
    after_dedup = len(items)
    print(f"      {after_dedup} new items (filtered {total - after_dedup} seen)")

    # 4. LLM Process (intra-period dedup + score + summarize)
    print("[4/5] LLM processing...")
    results = llm_process.process_batch(items, api_key, score_threshold)
    print(f"      {len(results)} items recommended")

    # 5. Deliver
    print("[5/5] Generating output...")
    stats = {
        "total": total,
        "after_dedup": after_dedup,
        "recommended": len(results)
    }
    deliver.to_console(results, stats)
    deliver.to_json(results, "output/daily.json")

    # Save updated seen records
    seen_records = seen.cleanup(seen_records)
    seen.save(seen_records)

    print(f"\nDone! Saved to output/daily.json")


if __name__ == "__main__":
    run()
