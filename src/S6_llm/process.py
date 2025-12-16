"""LLM processing - generate Chinese summaries for each article."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from ..models import NewsItem
from ..infra import chat, get_concurrency

PROMPT_PATH = Path("config/llm_prompt_template.txt")
if not PROMPT_PATH.exists():
    raise FileNotFoundError(f"Prompt template not found: {PROMPT_PATH}")
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class SummarizeStats:
    """Stats for summarization step."""
    total: int
    success: int
    failed: int


def summarize_single(item: NewsItem) -> NewsItem:
    """
    Generate Chinese summary for a single article.

    Args:
        item: NewsItem to summarize

    Returns:
        NewsItem with summary field populated
    """
    prompt = PROMPT_TEMPLATE.format(
        title=item.title or "",
        source=item.source_name or "",
        content=(item.content or "")[:1500],  # Limit content length
    )

    try:
        response = chat(prompt, max_tokens=2000)
        item.summary = response.strip() if response else ""
    except Exception as e:
        item.summary = ""
        item._summarize_error = str(e)

    return item


def summarize_batch(
    items: list[NewsItem],
    concurrency: int | None = None,
) -> tuple[list[NewsItem], SummarizeStats]:
    """
    Generate Chinese summaries for all articles (concurrent).

    Args:
        items: List of NewsItems to summarize
        concurrency: Number of concurrent LLM calls (default from config)

    Returns:
        (items_with_summaries, stats)
    """
    if not items:
        return [], SummarizeStats(0, 0, 0)

    if concurrency is None:
        concurrency = get_concurrency()

    total = len(items)
    success = 0
    failed = 0
    completed = 0
    lock = Lock()

    def process_item(idx: int, item: NewsItem) -> tuple[int, NewsItem]:
        return idx, summarize_single(item)

    print(f"      Summarizing {total} items (concurrency={concurrency})...")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(process_item, i, item): i
            for i, item in enumerate(items)
        }

        for future in as_completed(futures):
            idx, item = future.result()
            items[idx] = item

            with lock:
                completed += 1
                if item.summary:
                    success += 1
                else:
                    failed += 1

                # Progress update every N items (N=concurrency) or at the end
                if completed % concurrency == 0 or completed == total:
                    print(f"      [{completed}/{total}] done")

    stats = SummarizeStats(total=total, success=success, failed=failed)
    return items, stats


# Keep old function name for backwards compatibility
def process_batch(items: list[NewsItem], score_threshold: float = 6.0) -> list[NewsItem]:
    """
    Legacy function - now just calls summarize_batch.
    score_threshold is ignored (no filtering).
    """
    results, _ = summarize_batch(items)
    return results
