"""ScholarPipe - Academic news aggregation and AI digest."""

import importlib

# Import numbered packages
aggregate = importlib.import_module("src.1_aggregate")
clean = importlib.import_module("src.2_clean")
dedup = importlib.import_module("src.3_dedup")
filt = importlib.import_module("src.4_filter")
llm = importlib.import_module("src.5_llm")
deliver = importlib.import_module("src.6_deliver")


def run():
    """Run the complete pipeline."""

    # Config
    score_threshold = 6.0

    # Load dedup history
    seen_records = dedup.load()

    # 1. Aggregate (uses sources.yaml)
    print("[1/6] Aggregating...")
    items = aggregate.fetch_all()
    total = len(items) if items else 0
    print(f"      {total} items")

    if not items:
        print("No items found.")
        return

    # 2. Clean
    print("[2/6] Cleaning...")
    items = clean.batch_clean(items)

    # 3. Dedup (in-batch + cross-period)
    print("[3/6] Dedup (in-batch + cross-period)...")
    items = dedup.filter_duplicates_in_batch(items)
    items, new_fps = dedup.filter_unseen(items, seen_records)
    after_dedup = len(items)
    print(f"      {after_dedup} new")

    if not items:
        print("No new items.")
        dedup.save(seen_records)
        return

    # 4. Hybrid filter (cheap rules before LLM)
    print("[4/6] Hybrid filtering...")
    items, _dropped, stats = filt.filter_hybrid(
        items,
        seen_records=seen_records,
        record_layer2_dropped_to_seen=True,
    )
    after_filter = len(items)
    print(
        f"      kept {after_filter} | "
        f"L1 dropped {stats.layer1_dropped} | "
        f"VIP kept {stats.layer2_vip_kept} | "
        f"semantic kept {stats.layer2_semantic_kept}"
    )

    if not items:
        print("No items after filtering.")
        dedup.mark_batch(seen_records, new_fps)
        seen_records = dedup.cleanup(seen_records)
        dedup.save(seen_records)
        return

    # 5. LLM process
    print("[5/6] LLM processing...")
    results = llm.process_batch(items, score_threshold)
    print(f"      {len(results)} recommended")

    # 6. Deliver
    print("[6/6] Output...")
    stats = {"total": total, "after_dedup": after_dedup, "recommended": len(results)}
    deliver.to_console(results, stats)
    deliver.to_json(results, "output/daily.json")

    # Save dedup history
    dedup.mark_batch(seen_records, new_fps)
    seen_records = dedup.cleanup(seen_records)
    dedup.save(seen_records)

    print("\nDone!")


if __name__ == "__main__":
    run()
