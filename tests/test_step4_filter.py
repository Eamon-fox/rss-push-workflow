#!/usr/bin/env python
"""Step 4: Filter - 测试混合过滤功能"""

import importlib
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

filt = importlib.import_module("src.S4_filter")
from src.models import NewsItem


def load_deduped_data() -> list[NewsItem]:
    """从data/deduped目录加载最新的去重后数据"""
    deduped_dir = Path("data/deduped")
    if not deduped_dir.exists():
        return []

    date_dirs = sorted([d for d in deduped_dir.iterdir() if d.is_dir()], reverse=True)
    if not date_dirs:
        return []

    latest = date_dirs[0] / "all.json"
    if not latest.exists():
        return []

    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [NewsItem(**item) for item in data]


def create_test_items() -> list[NewsItem]:
    """创建测试数据"""
    return [
        # VIP 关键词命中
        NewsItem(
            title="RTCB ligase in tRNA splicing",
            content="Study reveals RTCB role in RNA processing and cellular stress response.",
            source_name="Nature",
        ),
        # 生物关键词命中，高语义相关
        NewsItem(
            title="CRISPR gene editing advances",
            content="New CRISPR technique enables precise genome editing in mammalian cells.",
            source_name="Science",
        ),
        # 生物关键词命中，低语义相关
        NewsItem(
            title="Plant biology discovery",
            content="Researchers found new chloroplast mechanism in plant photosynthesis.",
            source_name="Cell",
        ),
        # 非生物内容
        NewsItem(
            title="Climate change report",
            content="Global temperatures continue to rise according to new climate data.",
            source_name="News",
        ),
    ]


def test_filter():
    """测试混合过滤功能"""
    print("=" * 60)
    print("Step 4: Hybrid Filter 测试")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    items = load_deduped_data()

    if items:
        print(f"    从 data/deduped 加载了 {len(items)} 条数据")
    else:
        print("    未找到实际数据，使用模拟数据")
        items = create_test_items()
        print(f"    创建了 {len(items)} 条模拟数据")

    # 执行过滤
    print("\n[2] 执行混合过滤...")
    kept, dropped, stats = filt.filter_hybrid(items, save=False)

    print(f"\n[3] 过滤结果:")
    print(f"    总数: {stats.total}")
    print(f"    Layer1 丢弃 (非生物): {stats.layer1_dropped}")
    print(f"    Layer2 VIP 保留: {stats.layer2_vip_kept}")
    print(f"    Layer2 语义保留: {stats.layer2_semantic_kept}")
    print(f"    Layer2 语义丢弃: {stats.layer2_semantic_dropped}")
    print(f"    最终保留: {len(kept)}")

    # 展示保留的文章
    if kept:
        print("\n[4] 保留的文章:")
        for i, item in enumerate(kept[:5], 1):
            vip_tag = f" [VIP: {', '.join(item.vip_keywords)}]" if item.is_vip else ""
            sem_score = f" (sem={item.semantic_score:.3f})" if item.semantic_score else ""
            print(f"    [{i}] {item.title[:45]}...{vip_tag}{sem_score}")

    print("\n" + "=" * 60)
    print("✓ Step 4 测试完成")
    print("=" * 60)

    return kept, stats


if __name__ == "__main__":
    test_filter()
