"""ScholarPipe - Academic news aggregation and AI digest."""

import argparse
import importlib
import logging
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────

def setup_logging():
    """Configure logging to file and console."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{today}.log"

    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler]
    )

    return log_file


# Import pipeline steps
aggregate = importlib.import_module("src.S1_aggregate")
clean = importlib.import_module("src.S2_clean")
dedup = importlib.import_module("src.S3_dedup")
filt = importlib.import_module("src.S4_filter")
enrich = importlib.import_module("src.S5_enrich")
llm = importlib.import_module("src.S6_llm")
deliver = importlib.import_module("src.S7_deliver")


# ─────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────

def print_header():
    """Print pipeline header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print()
    print("=" * 62)
    print("              ScholarPipe - Academic Digest                 ")
    print(f"                     {now}                       ")
    print("=" * 62)
    print()


def print_step(step: int, total: int, name: str):
    """Print step header."""
    print(f"[Step {step}/{total}] {name}")


def print_detail(key: str, value, indent: int = 1):
    """Print a detail line."""
    prefix = "|  " * indent
    print(f"{prefix}- {key}: {value}")


def print_table(headers: list[str], rows: list[list], indent: int = 1):
    """Print a simple table."""
    prefix = "|  " * indent
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"{prefix}{header_line}")
    print(f"{prefix}{'-' * len(header_line)}")

    # Print rows
    for row in rows:
        row_line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(f"{prefix}{row_line}")


def print_step_end(input_count: int, output_count: int):
    """Print step summary with funnel visualization."""
    dropped = input_count - output_count
    pct = (output_count / input_count * 100) if input_count > 0 else 0
    print(f"|")
    print(f"|  {input_count} -> {output_count} ({pct:.0f}% kept, {dropped} dropped)")
    print("-" * 50)
    print()


def sort_results(items):
    """Sort final results by semantic score (descending)."""
    if not items:
        return []

    has_semantic = any(getattr(it, "semantic_score", None) is not None for it in items)
    if not has_semantic:
        return items

    return sorted(
        items,
        key=lambda it: it.semantic_score if it.semantic_score is not None else float("-inf"),
        reverse=True,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ScholarPipe - Academic news aggregation and AI digest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with default settings
  python main.py --output report.json      # Save JSON to custom path
  python main.py --no-llm                  # Skip LLM summarization
        """
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output/daily.json",
        help="Output JSON file path (default: output/daily.json)"
    )

    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Output HTML file path (default: same as --output with .html extension)"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM processing (for testing pipeline)"
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip step 1 (aggregate), load from data/raw/{date}/all.json instead"
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to load raw data from (default: today). Format: YYYY-MM-DD"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top articles to include in daily report (default: 20)"
    )

    return parser.parse_args()


def run(args=None):
    """Run the complete pipeline."""
    if args is None:
        args = parse_args()

    # Initialize logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("ScholarPipe started")

    print_header()

    # Load dedup history
    seen_records = dedup.load()

    # ─────────────────────────────────────────────────────────
    # Step 1: Aggregate
    # ─────────────────────────────────────────────────────────
    print_step(1, 7, "Aggregate (fetch RSS/PubMed)")

    if args.skip_fetch:
        # Load from existing raw data
        import json
        from src.models import NewsItem

        date_str = args.date or datetime.now().strftime("%Y-%m-%d")
        raw_file = Path(f"data/raw/{date_str}/all.json")

        if not raw_file.exists():
            print_detail("Status", f"FAILED - {raw_file} not found")
            print_step_end(0, 0)
            print("No raw data found. Run without --skip-fetch first.")
            return

        print_detail("Status", f"SKIPPED (loading from {raw_file})")
        with open(raw_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        items = [NewsItem(**item) for item in raw_data]
    else:
        items = aggregate.fetch_all()

    total = len(items) if items else 0

    # Count by source
    source_counts: dict[str, int] = {}
    for item in (items or []):
        src = item.source_name or "Unknown"
        source_counts[src] = source_counts.get(src, 0) + 1

    if source_counts:
        rows = [[src, cnt] for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1])]
        print_table(["Source", "Count"], rows)

    print_step_end(total, total)

    if not items:
        print("No items found.")
        return

    # ─────────────────────────────────────────────────────────
    # Step 2: Clean
    # ─────────────────────────────────────────────────────────
    print_step(2, 7, "Clean (HTML sanitize)")
    before_clean = len(items)
    items = clean.batch_clean(items)
    print_step_end(before_clean, len(items))

    # ─────────────────────────────────────────────────────────
    # Step 3: Dedup
    # ─────────────────────────────────────────────────────────
    print_step(3, 7, "Dedup (fingerprint check)")
    before_dedup = len(items)
    items = dedup.filter_duplicates_in_batch(items)
    after_batch_dedup = len(items)
    items, new_fps = dedup.filter_unseen(items, seen_records)
    after_dedup = len(items)

    print_detail("In-batch duplicates removed", before_dedup - after_batch_dedup)
    print_detail("Previously seen removed", after_batch_dedup - after_dedup)
    print_detail("New items", after_dedup)
    print_step_end(before_dedup, after_dedup)

    if not items:
        print("No new items.")
        dedup.save(seen_records)
        return

    # ─────────────────────────────────────────────────────────
    # Step 4: Enrich (补充完整摘要，在语义过滤前执行)
    # ─────────────────────────────────────────────────────────
    print_step(4, 7, "Enrich (fetch full abstracts)")
    before_enrich = len(items)
    items, enrich_stats = enrich.enrich_batch(items, min_content_len=150)
    print_detail("Need enrichment", enrich_stats.need_enrich)
    print_detail("Successfully enriched", enrich_stats.enriched)
    print_step_end(before_enrich, len(items))

    # ─────────────────────────────────────────────────────────
    # Step 5: Hybrid Filter (用完整摘要做语义过滤)
    # ─────────────────────────────────────────────────────────
    print_step(5, 7, "Hybrid Filter (keyword + semantic)")
    before_filter = len(items)
    items, dropped_items, stats = filt.filter_hybrid(
        items,
        seen_records=seen_records,
        record_layer2_dropped_to_seen=True,
    )
    after_filter = len(items)

    print_detail("Layer 1 (bio keywords)", f"passed {stats.total - stats.layer1_dropped}, dropped {stats.layer1_dropped}")
    print_detail("Layer 2 (tiered scoring)", f"kept {stats.layer2_kept}, dropped {stats.layer2_dropped}")
    if stats.avg_final_score > 0:
        print_detail("Final scores", f"avg={stats.avg_final_score:.3f}, min={stats.min_final_score:.3f}, max={stats.max_final_score:.3f}")

    print_step_end(before_filter, after_filter)

    if not items:
        print("No items after filtering.")
        dedup.mark_batch(seen_records, new_fps)
        seen_records = dedup.cleanup(seen_records)
        dedup.save(seen_records)
        return

    # ─────────────────────────────────────────────────────────
    # Truncate to top N before LLM (save resources)
    # ─────────────────────────────────────────────────────────
    items = sort_results(items)  # Sort by semantic score first
    if len(items) > args.top:
        print(f"[Truncate] Keeping top {args.top} of {len(items)} items for daily report")
        items = items[:args.top]

    # ─────────────────────────────────────────────────────────
    # Step 6: LLM Summarize (生成中文摘要)
    # ─────────────────────────────────────────────────────────
    print_step(6, 7, "LLM Summarize (generate Chinese summaries)")
    before_llm = len(items)

    if args.no_llm:
        print_detail("Status", "SKIPPED (--no-llm flag)")
        llm_stats = None
    else:
        # Print LLM config
        from src.infra.llm import _llm_config, get_concurrency
        llm_cfg = _llm_config()
        provider = llm_cfg["provider"]
        if provider == "ollama":
            model_info = f"{provider}/{llm_cfg['ollama_model']}"
        elif provider == "siliconflow":
            model_info = f"{provider}/{llm_cfg['siliconflow_model']}"
        elif provider == "mimo":
            model_info = f"{provider}/{llm_cfg.get('mimo_model', 'mimo-v2-flash')}"
        else:
            model_info = f"{provider}/{llm_cfg['zhipu_model']}"
        print_detail("Model", f"{model_info} (concurrency={get_concurrency()})")

        items, llm_stats = llm.summarize_batch(items)
        print_detail("Summarized", f"{llm_stats.success}/{llm_stats.total}")
        if llm_stats.cached > 0:
            print_detail("From cache", llm_stats.cached)
        if llm_stats.failed > 0:
            print_detail("Failed", llm_stats.failed)

    print_step_end(before_llm, len(items))

    # Sort by semantic score (already computed in Step 5)
    results = sort_results(items)

    # ─────────────────────────────────────────────────────────
    # Step 7: Deliver
    # ─────────────────────────────────────────────────────────
    print_step(7, 7, "Deliver (output results)")

    # Save outputs
    deliver.to_json(results, args.output)
    print_detail("JSON saved", args.output)

    markdown_path = Path(args.output).with_suffix(".md")
    md_content = deliver.to_markdown(results)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print_detail("Markdown saved", str(markdown_path))

    # HTML output
    html_path = args.html or str(Path(args.output).with_suffix(".html"))
    stats = {
        "total": total,
        "after_dedup": after_dedup,
        "after_filter": after_filter,
    }
    deliver.to_html_file(results, html_path, stats)
    print_detail("HTML saved", html_path)

    # Save dedup history
    dedup.mark_batch(seen_records, new_fps)
    seen_records = dedup.cleanup(seen_records)
    dedup.save(seen_records)
    print_detail("Dedup history", "updated")

    print_step_end(len(results), len(results))

    # Final Summary
    print("=" * 62)
    print("                      Pipeline Summary                      ")
    print("=" * 62)
    print(f"  Fetched:      {total:>5}  items")
    print(f"  After dedup:  {after_dedup:>5}  items")
    print(f"  Enriched:     {enrich_stats.enriched:>5}  items")
    print(f"  After filter: {after_filter:>5}  items")
    print(f"  Summarized:   {llm_stats.success if llm_stats else 0:>5}  items")
    print(f"  Final output: {len(results):>5}  items")
    print("=" * 62)
    print()

    # Show top items preview
    if results:
        print("Top items:")
        for i, item in enumerate(results[:5], 1):
            vip_tag = ""
            if item.is_vip and item.vip_keywords:
                vip_tag = f" [VIP: {', '.join(item.vip_keywords)}]"
            elif item.is_vip:
                vip_tag = " [VIP]"
            score_tag = f" (sem={item.semantic_score:.2f})" if item.semantic_score else ""
            title = item.title[:50] + "..." if len(item.title) > 50 else item.title
            print(f"  {i}. {title}{vip_tag}{score_tag}")
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more")
        print()

    logger.info("ScholarPipe completed successfully")
    logger.info("=" * 60)
    print(f"[OK] Done! (log: {log_file})")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
