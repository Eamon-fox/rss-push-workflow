"""Enrich module - 补充完整摘要和元数据."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from ..models import NewsItem

# 导入已有的PubMed enrichment功能
from ..S1_aggregate.pubmed import enrich_items_by_doi


@dataclass
class UserProfile:
    """用户研究兴趣配置."""

    research_context: str = ""
    core_topics: list[str] = None
    highlight_keywords: list[str] = None
    relevance_instruction: str = ""

    def __post_init__(self):
        if self.core_topics is None:
            self.core_topics = []
        if self.highlight_keywords is None:
            self.highlight_keywords = []


@dataclass
class EnrichStats:
    """Enrichment统计."""

    total: int = 0
    need_enrich: int = 0
    enriched: int = 0
    failed: int = 0


def load_user_profile(config_path: str | Path = "config/user_profile.yaml") -> UserProfile:
    """加载用户研究兴趣配置."""
    path = Path(config_path)
    if not path.exists():
        print(f"  [Enrich] User profile not found: {path}, using defaults")
        return UserProfile()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return UserProfile(
            research_context=data.get("research_context", ""),
            core_topics=data.get("core_topics", []),
            highlight_keywords=data.get("highlight_keywords", []),
            relevance_instruction=data.get("relevance_instruction", ""),
        )
    except Exception as e:
        print(f"  [Enrich] Failed to load user profile: {e}")
        return UserProfile()


def enrich_batch(
    items: list["NewsItem"],
    *,
    min_content_len: int = 150,
    use_crossref: bool = False,
) -> tuple[list["NewsItem"], EnrichStats]:
    """
    批量增强条目，补充完整摘要.

    Args:
        items: 待增强的条目列表
        min_content_len: 内容长度阈值，低于此值的条目需要增强
        use_crossref: 是否使用Crossref作为备选源（暂未实现）

    Returns:
        (enriched_items, stats)
    """
    if not items:
        return items, EnrichStats()

    stats = EnrichStats(total=len(items))

    # 统计需要enrich的条目数
    for item in items:
        content = (item.content or "").strip()
        if len(content) < min_content_len and item.doi:
            stats.need_enrich += 1

    if stats.need_enrich == 0:
        print(f"  [Enrich] All {stats.total} items have sufficient content")
        return items, stats

    print(f"  [Enrich] {stats.need_enrich}/{stats.total} items need enrichment")

    # 使用已有的PubMed enrichment
    try:
        enriched = enrich_items_by_doi(
            items,
            min_content_len=min_content_len,
        )

        # 统计实际enriched数量，并标记is_enriched
        result = []
        for orig, new in zip(items, enriched):
            orig_len = len((orig.content or "").strip())
            new_len = len((new.content or "").strip())
            if new_len > orig_len:
                stats.enriched += 1
                # 标记为已增强
                new = new.model_copy(update={
                    "is_enriched": True,
                    "original_content_len": orig_len,
                })
            result.append(new)

        stats.failed = stats.need_enrich - stats.enriched

        print(f"  [Enrich] Success: {stats.enriched}, Failed: {stats.failed}")
        return result, stats

    except Exception as e:
        print(f"  [Enrich] Batch enrichment failed: {e}")
        stats.failed = stats.need_enrich
        return items, stats


def enrich_with_crossref(
    items: list["NewsItem"],
    *,
    min_content_len: int = 150,
) -> list["NewsItem"]:
    """
    使用Crossref API补充摘要（备选方案）.

    Crossref覆盖面更广，但摘要质量可能不如PubMed.
    """
    # TODO: 实现Crossref enrichment
    # API: https://api.crossref.org/works/{doi}
    # 返回的abstract字段可能是HTML格式
    return items
