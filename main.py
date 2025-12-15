"""ScholarPipe - Academic news aggregation and AI digest."""

import importlib

# Import numbered packages
aggregate = importlib.import_module("src.1_aggregate")
clean = importlib.import_module("src.2_clean")
dedup = importlib.import_module("src.3_dedup")
llm = importlib.import_module("src.4_llm")
deliver = importlib.import_module("src.5_deliver")


def run():
    """Run the complete pipeline."""

    # Config
    sources = [
        {"type": "rss", "url": "https://www.nature.com/neuro.rss", "name": "Nature Neuroscience"},
    ]
    score_threshold = 6.0

    # Load dedup history
    seen_records = dedup.load()

    # 1. Aggregate
    print("[1/5] Aggregating...")
    items = aggregate.fetch_all(sources)
    total = len(items) if items else 0
    print(f"      {total} items")

    if not items:
        print("No items found.")
        return

    # 2. Clean
    print("[2/5] Cleaning...")
    items = clean.batch_clean(items)

    # 3. Cross-period dedup
    print("[3/5] Dedup (cross-period)...")
    items = dedup.filter_unseen(items, seen_records)
    after_dedup = len(items)
    print(f"      {after_dedup} new")

    if not items:
        print("No new items.")
        dedup.save(seen_records)
        return

    # 4. LLM process
    print("[4/5] LLM processing...")
    results = llm.process_batch(items, score_threshold)
    print(f"      {len(results)} recommended")

    # 5. Deliver
    print("[5/5] Output...")
    stats = {"total": total, "after_dedup": after_dedup, "recommended": len(results)}
    deliver.to_console(results, stats)
    deliver.to_json(results, "output/daily.json")

    # Save dedup history
    seen_records = dedup.cleanup(seen_records)
    dedup.save(seen_records)

    print("\nDone!")


if __name__ == "__main__":
    run()
