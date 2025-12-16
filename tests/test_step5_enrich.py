#!/usr/bin/env python
"""Step 5: Enrich - 测试摘要补充功能"""

import importlib
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

enrich = importlib.import_module("src.S5_enrich")
from src.models import NewsItem


def create_test_items() -> list[NewsItem]:
    """创建测试数据 - 包含需要补充和不需要补充的"""
    return [
        # 短摘要，有 DOI，需要补充
        NewsItem(
            title="New CRISPR discovery",
            content="Brief summary.",
            doi="10.1038/s41586-024-00001-1",
            source_name="Nature",
        ),
        # 长摘要，不需要补充
        NewsItem(
            title="Protein structure analysis",
            content="This is a comprehensive study about protein folding mechanisms. " * 20,
            doi="10.1126/science.abc1234",
            source_name="Science",
        ),
        # 无 DOI，无法补充
        NewsItem(
            title="General science news",
            content="Short content without DOI.",
            source_name="ScienceDaily",
        ),
    ]


def test_enrich():
    """测试摘要补充功能"""
    print("=" * 60)
    print("Step 5: Enrich 测试")
    print("=" * 60)

    # 创建测试数据
    print("\n[1] 创建测试数据...")
    items = create_test_items()
    print(f"    创建了 {len(items)} 条测试数据")

    for i, item in enumerate(items, 1):
        doi_tag = f" (DOI: {item.doi})" if item.doi else " (无DOI)"
        print(f"    [{i}] {item.title} - 内容长度: {len(item.content)}{doi_tag}")

    # 执行补充
    print("\n[2] 执行摘要补充 (min_content_len=150)...")
    enriched_items, stats = enrich.enrich_batch(items, min_content_len=150)

    print(f"\n[3] 补充结果:")
    print(f"    需要补充: {stats.need_enrich}")
    print(f"    成功补充: {stats.enriched}")
    print(f"    跳过 (无DOI): {stats.skipped_no_doi}")
    print(f"    失败: {stats.failed}")

    # 展示补充后的结果
    print("\n[4] 补充后内容长度:")
    for i, item in enumerate(enriched_items, 1):
        enriched_tag = " [已补充]" if item.is_enriched else ""
        print(f"    [{i}] {item.title}: {len(item.content)} chars{enriched_tag}")

    print("\n" + "=" * 60)
    print("✓ Step 5 测试完成")
    print("=" * 60)

    return enriched_items, stats


if __name__ == "__main__":
    test_enrich()
