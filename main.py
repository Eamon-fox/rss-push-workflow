"""ScholarPipe - Academic news aggregation and AI digest."""

import argparse
import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────────────────────

PROGRESS_FILE = Path("data/pipeline_progress.json")

STEPS = [
    {"step": 1, "name": "Aggregate", "desc": "抓取 RSS/PubMed"},
    {"step": 2, "name": "Clean", "desc": "HTML 清洗"},
    {"step": 3, "name": "Dedup", "desc": "去重"},
    {"step": 4, "name": "Enrich", "desc": "补充摘要"},
    {"step": 5, "name": "Filter", "desc": "语义过滤"},
    {"step": 6, "name": "LLM", "desc": "生成摘要"},
    {"step": 7, "name": "Deliver", "desc": "输出结果"},
]


def update_progress(
    step: int,
    status: str = "running",
    input_count: int = 0,
    output_count: int = 0,
    detail: str = "",
):
    """Update progress file for API to read."""
    step_info = STEPS[step - 1] if 1 <= step <= len(STEPS) else {}
    progress = {
        "current_step": step,
        "total_steps": len(STEPS),
        "step_name": step_info.get("name", ""),
        "step_desc": step_info.get("desc", ""),
        "status": status,  # running, completed, failed
        "updated_at": datetime.now().isoformat(),
        "input_count": input_count,
        "output_count": output_count,
        "detail": detail,
    }
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)


def clear_progress():
    """Clear progress file when pipeline completes."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


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

    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User ID for per-user seen records isolation (default: 'default')"
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

    # Migrate legacy seen.json to per-user format (if exists)
    dedup.migrate_legacy_seen()

    # Load dedup history for this user
    seen_records = dedup.load(user_id=args.user)

    # ─────────────────────────────────────────────────────────
    # Step 1: Aggregate
    # ─────────────────────────────────────────────────────────
    update_progress(1, "running")
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
    update_progress(1, "completed", output_count=total)

    if not items:
        print("No items found.")
        clear_progress()
        return

    # ─────────────────────────────────────────────────────────
    # Step 2: Clean
    # ─────────────────────────────────────────────────────────
    update_progress(2, "running", input_count=total)
    print_step(2, 7, "Clean (HTML sanitize)")
    before_clean = len(items)
    items = clean.batch_clean(items)
    print_step_end(before_clean, len(items))
    update_progress(2, "completed", input_count=before_clean, output_count=len(items))

    # ─────────────────────────────────────────────────────────
    # Step 3: Dedup
    # ─────────────────────────────────────────────────────────
    update_progress(3, "running", input_count=len(items))
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
    update_progress(3, "completed", input_count=before_dedup, output_count=after_dedup)

    if not items:
        print("No new items.")
        dedup.save(seen_records, user_id=args.user)
        clear_progress()
        return

    # ─────────────────────────────────────────────────────────
    # Step 4: Enrich (补充完整摘要，在语义过滤前执行)
    # ─────────────────────────────────────────────────────────
    update_progress(4, "running", input_count=len(items))
    print_step(4, 7, "Enrich (fetch full abstracts)")
    before_enrich = len(items)
    items, enrich_stats = enrich.enrich_batch(items, min_content_len=150)
    print_detail("Need enrichment", enrich_stats.need_enrich)
    print_detail("Successfully enriched", enrich_stats.enriched)
    print_step_end(before_enrich, len(items))
    update_progress(4, "completed", input_count=before_enrich, output_count=len(items))

    # ─────────────────────────────────────────────────────────
    # Step 5: Hybrid Filter (用完整摘要做语义过滤)
    # ─────────────────────────────────────────────────────────
    update_progress(5, "running", input_count=len(items))
    print_step(5, 7, "Hybrid Filter (keyword + semantic)")
    before_filter = len(items)

    # Step 5a: Layer 1 - 关键词过滤
    layer1_passed, layer1_dropped = filt.filter_layer1(items)
    print_detail("Layer 1 (bio keywords)", f"passed {len(layer1_passed)}, dropped {len(layer1_dropped)}")

    if not layer1_passed:
        print("No items after Layer 1 filtering.")
        seen_records = dedup.cleanup(seen_records)
        dedup.save(seen_records, user_id=args.user)
        clear_progress()
        return

    # Step 5b: 计算语义分数 (embedding 会被缓存)
    scored_items = filt.score_with_anchors(layer1_passed)

    # Step 5c: 保存候选池 (供个性化使用)
    from src.archive import archive_candidate_pool
    candidates_for_pool = [item for item, score in scored_items]
    archive_candidate_pool(candidates_for_pool)
    print_detail("Candidate pool saved", len(candidates_for_pool))

    # Step 5d: 按阈值过滤 (用于默认日报)
    from src.S4_filter.config import SCORING_CONFIG
    threshold = SCORING_CONFIG.final_threshold
    items = [item for item, score in scored_items if score >= threshold]
    dropped_count = len(scored_items) - len(items)

    after_filter = len(items)
    print_detail("Layer 2 (tiered scoring)", f"kept {after_filter}, dropped {dropped_count} (threshold={threshold})")

    if items:
        scores = [item.semantic_score for item in items if item.semantic_score]
        if scores:
            print_detail("Final scores", f"avg={sum(scores)/len(scores):.3f}, min={min(scores):.3f}, max={max(scores):.3f}")

    print_step_end(before_filter, after_filter)
    update_progress(5, "completed", input_count=before_filter, output_count=after_filter)

    if not items:
        print("No items after filtering.")
        # 不标记任何指纹，因为没有文章进入日报
        seen_records = dedup.cleanup(seen_records)
        dedup.save(seen_records, user_id=args.user)
        clear_progress()
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
    update_progress(6, "running", input_count=len(items))
    print_step(6, 7, "LLM Summarize (generate Chinese summaries)")
    logger.info(f"Step 6: LLM Summarize starting with {len(items)} items")
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
    update_progress(6, "completed", input_count=before_llm, output_count=len(items))
    if llm_stats:
        logger.info(f"Step 6: LLM completed - {llm_stats.success}/{llm_stats.total} summarized, {llm_stats.cached} cached")

    # Sort by semantic score (already computed in Step 5)
    results = sort_results(items)

    # ─────────────────────────────────────────────────────────
    # Step 7: Deliver
    # ─────────────────────────────────────────────────────────
    update_progress(7, "running", input_count=len(results))
    print_step(7, 7, "Deliver (output results)")
    logger.info(f"Step 7: Deliver starting with {len(results)} items")

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

    # Save dedup history - 只记录进入日报的文章
    from src.S3_dedup.fingerprint import get_fingerprint
    final_fps = [get_fingerprint(item) for item in results]
    final_fps = [fp for fp in final_fps if fp]  # 过滤空指纹
    dedup.mark_batch(seen_records, final_fps)
    seen_records = dedup.cleanup(seen_records)
    dedup.save(seen_records, user_id=args.user)
    print_detail("Dedup history", f"recorded {len(final_fps)} items")

    # Archive daily report
    from src.archive import archive_daily
    archive_stats = {
        "total": total,
        "after_dedup": after_dedup,
        "after_filter": after_filter,
    }
    archive_info = archive_daily(results, archive_stats)
    print_detail("Archived", f"v{archive_info['version']} -> output/archive/")
    logger.info(f"Archived v{archive_info['version']} with {len(results)} articles")

    # Clean up intermediate data, old raw cache, and old logs
    try:
        from src.cleanup import cleanup_intermediate_data, cleanup_old_raw_data, cleanup_old_logs
        today_str = datetime.now().strftime("%Y-%m-%d")
        cleanup_stats = cleanup_intermediate_data(date=today_str)
        raw_cleanup_stats = cleanup_old_raw_data()
        log_cleanup_stats = cleanup_old_logs()
        total_cleaned = cleanup_stats['dirs_removed'] + raw_cleanup_stats['dirs_removed']
        logs_removed = log_cleanup_stats['files_removed']
        print_detail("Cleanup", f"{total_cleaned} dirs, {logs_removed} logs removed")
        logger.info(f"Cleanup completed: {total_cleaned} dirs, {logs_removed} logs removed")
    except Exception as e:
        logger.warning(f"Cleanup failed (non-critical): {e}")
        print_detail("Cleanup", "skipped due to error")

    print_step_end(len(results), len(results))
    update_progress(7, "completed", input_count=len(results), output_count=len(results))
    clear_progress()
    logger.info("Step 7: Deliver completed")

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
