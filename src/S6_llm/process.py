"""LLM processing - generate Chinese summaries for each article."""

import hashlib
import json
import logging
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import yaml

logger = logging.getLogger(__name__)

from ..models import NewsItem
from ..infra import chat, get_concurrency

PROMPT_PATH = Path("config/llm_prompt_template.txt")
CACHE_DB_PATH = Path("data/llm_cache.db")
USER_PROFILE_PATH = Path("config/user_profile.yaml")

if not PROMPT_PATH.exists():
    raise FileNotFoundError(f"Prompt template not found: {PROMPT_PATH}")
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# LLM Cache
# ─────────────────────────────────────────────────────────────

class LLMCache:
    """SQLite-based cache for LLM summaries."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._db_lock = Lock()
                    cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(CACHE_DB_PATH), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                cache_key TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _make_key(self, item: "NewsItem") -> str:
        """Generate cache key from DOI or title hash."""
        if item.doi:
            return f"doi:{item.doi.lower()}"
        title = (item.title or "").strip().lower()
        return f"title:{hashlib.md5(title.encode()).hexdigest()}"

    def get(self, item: "NewsItem") -> str | None:
        """Get cached summary, returns None if not found."""
        key = self._make_key(item)
        with self._db_lock:
            cur = self.conn.execute(
                "SELECT summary FROM summaries WHERE cache_key = ?", (key,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, item: "NewsItem", summary: str):
        """Cache a summary."""
        key = self._make_key(item)
        with self._db_lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO summaries (cache_key, summary) VALUES (?, ?)",
                (key, summary)
            )
            self.conn.commit()


def _get_cache() -> LLMCache:
    return LLMCache()

# 质量检查常量
MIN_SUMMARY_LENGTH = 50
MAX_SUMMARY_LENGTH = 300
MAX_RETRIES = 2

# VIP 关键词（用于检查是否需要【关联】标注）
VIP_KEYWORDS = ["tRNA", "RtcB", "RTCB", "XBP1", "IRE1", "UPR", "RNA ligase", "RNA连接"]


@lru_cache(maxsize=1)
def _load_user_profile() -> str:
    """Load user profile and format as context string."""
    if not USER_PROFILE_PATH.exists():
        return ""

    try:
        profile = yaml.safe_load(USER_PROFILE_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""

    context_parts = []

    # 研究背景
    if research_context := profile.get("research_context"):
        context_parts.append(f"## 用户研究背景\n{research_context.strip()}")

    # 关联提示
    if relevance_instruction := profile.get("relevance_instruction"):
        context_parts.append(f"## 关联判断指引\n{relevance_instruction.strip()}")

    return "\n\n".join(context_parts)


def _smart_truncate(content: str, max_length: int = 1500) -> str:
    """
    智能截断内容，优先保留摘要和结论部分。

    策略：
    1. 如果内容足够短，直接返回
    2. 尝试识别关键段落（Abstract, Conclusion, Results）
    3. 优先保留开头和关键段落
    """
    if not content or len(content) <= max_length:
        return content

    # 尝试找到 Abstract 或摘要段落
    abstract_patterns = [
        r"(?i)abstract[:\s]*(.{100,800}?)(?=\n\n|introduction|background|$)",
        r"(?i)summary[:\s]*(.{100,800}?)(?=\n\n|introduction|$)",
    ]

    abstract_text = ""
    for pattern in abstract_patterns:
        match = re.search(pattern, content)
        if match:
            abstract_text = match.group(1).strip()
            break

    # 尝试找到 Conclusion 或结论段落
    conclusion_patterns = [
        r"(?i)conclusion[s]?[:\s]*(.{100,600}?)(?=\n\n|reference|acknowledge|$)",
        r"(?i)in summary[,:\s]*(.{100,400}?)(?=\n\n|$)",
    ]

    conclusion_text = ""
    for pattern in conclusion_patterns:
        match = re.search(pattern, content)
        if match:
            conclusion_text = match.group(1).strip()
            break

    # 组合策略
    if abstract_text and conclusion_text:
        # 有摘要和结论：摘要 + 结论 + 开头填充
        combined = f"{abstract_text}\n\n{conclusion_text}"
        if len(combined) < max_length:
            # 还有空间，加入开头内容
            remaining = max_length - len(combined) - 50
            if remaining > 100:
                combined = f"{content[:remaining]}...\n\n{combined}"
        return combined[:max_length]

    elif abstract_text:
        # 只有摘要：摘要 + 开头
        remaining = max_length - len(abstract_text) - 50
        if remaining > 200:
            return f"{content[:remaining]}...\n\n[Abstract] {abstract_text}"
        return abstract_text[:max_length]

    else:
        # 无特殊结构：取开头，尽量在句子边界截断
        truncated = content[:max_length]
        # 尝试在句号处截断
        last_period = max(
            truncated.rfind(". "),
            truncated.rfind("。"),
            truncated.rfind(".\n"),
        )
        if last_period > max_length * 0.7:
            truncated = truncated[:last_period + 1]
        return truncated


def _check_vip_content(content: str, title: str) -> bool:
    """检查内容是否包含 VIP 关键词。"""
    text = f"{title} {content}".lower()
    for kw in VIP_KEYWORDS:
        if kw.lower() in text:
            return True
    return False


def _post_process_summary(summary: str, is_vip_content: bool) -> str:
    """
    后处理摘要：
    1. 清理多余空白
    2. 如果是 VIP 内容但缺少【关联】标注，添加之
    """
    if not summary:
        return summary

    # 清理
    summary = summary.strip()
    summary = re.sub(r"\n{3,}", "\n\n", summary)

    # 移除可能的前缀
    prefixes_to_remove = [
        "摘要：", "摘要:", "中文摘要：", "中文摘要:",
        "输出：", "输出:", "翻译：", "翻译:",
    ]
    for prefix in prefixes_to_remove:
        if summary.startswith(prefix):
            summary = summary[len(prefix):].strip()

    # VIP 内容检查【关联】标注
    if is_vip_content and "【关联】" not in summary:
        summary = summary.rstrip("。.") + "。【关联】"

    return summary


@dataclass
class SummarizeStats:
    """Stats for summarization step."""
    total: int
    success: int
    failed: int
    retried: int = 0
    cached: int = 0  # Number of cache hits


def summarize_single(item: NewsItem, user_context: str = "", use_cache: bool = True) -> NewsItem:
    """
    Generate Chinese summary for a single article with retry and quality check.

    Args:
        item: NewsItem to summarize
        user_context: User profile context string
        use_cache: Whether to use cache (default True)

    Returns:
        NewsItem with summary field populated
    """
    cache = _get_cache()

    # Check cache first
    if use_cache:
        cached = cache.get(item)
        if cached:
            item.summary = cached
            item._from_cache = True
            item._retries = 0
            return item

    # 智能截断内容
    content = _smart_truncate(item.content or "", max_length=1500)

    # 检查是否 VIP 内容
    is_vip = _check_vip_content(content, item.title or "")

    prompt = PROMPT_TEMPLATE.format(
        title=item.title or "",
        source=item.source_name or "",
        content=content,
        user_context=user_context,
    )

    last_error = None
    retries = 0

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = chat(prompt, max_tokens=2000)
            summary = response.strip() if response else ""

            # 质量检查
            if summary and len(summary) >= MIN_SUMMARY_LENGTH:
                # 后处理
                item.summary = _post_process_summary(summary, is_vip)
                item._retries = retries
                item._from_cache = False
                # Save to cache
                if use_cache:
                    cache.set(item, item.summary)
                return item

            # 摘要过短，重试
            retries += 1
            last_error = f"Summary too short: {len(summary)} chars"

        except Exception as e:
            retries += 1
            last_error = str(e)

    # 所有重试都失败
    item.summary = ""
    item._summarize_error = last_error
    item._retries = retries
    item._from_cache = False
    return item


def summarize_batch(
    items: list[NewsItem],
    concurrency: int | None = None,
    use_cache: bool = True,
) -> tuple[list[NewsItem], SummarizeStats]:
    """
    Generate Chinese summaries for all articles (concurrent).

    Args:
        items: List of NewsItems to summarize
        concurrency: Number of concurrent LLM calls (default from config)
        use_cache: Whether to use cache (default True)

    Returns:
        (items_with_summaries, stats)
    """
    if not items:
        return [], SummarizeStats(0, 0, 0, 0, 0)

    if concurrency is None:
        concurrency = get_concurrency()

    # 预加载用户配置
    user_context = _load_user_profile()

    total = len(items)
    success = 0
    failed = 0
    retried = 0
    cached = 0
    completed = 0
    lock = Lock()

    def process_item(idx: int, item: NewsItem) -> tuple[int, NewsItem]:
        return idx, summarize_single(item, user_context, use_cache=use_cache)

    logger.info(f"Summarizing {total} items (concurrency={concurrency}, cache={'on' if use_cache else 'off'})")

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
                if hasattr(item, "_retries") and item._retries > 0:
                    retried += 1
                if hasattr(item, "_from_cache") and item._from_cache:
                    cached += 1

                # Progress update every N items (N=concurrency) or at the end
                if completed % concurrency == 0 or completed == total:
                    cache_info = f", cached={cached}" if cached > 0 else ""
                    logger.info(f"LLM progress: [{completed}/{total}] done (success={success}, failed={failed}{cache_info})")

    stats = SummarizeStats(total=total, success=success, failed=failed, retried=retried, cached=cached)
    return items, stats


# Keep old function name for backwards compatibility
def process_batch(items: list[NewsItem], score_threshold: float = 6.0) -> list[NewsItem]:
    """
    Legacy function - now just calls summarize_batch.
    score_threshold is ignored (no filtering).
    """
    results, _ = summarize_batch(items)
    return results
