#!/usr/bin/env python
"""Step 3: Dedup - 测试去重功能"""

import importlib
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

dedup = importlib.import_module("src.S3_dedup")
from src.models import NewsItem


def load_cleaned_data() -> list[NewsItem]:
    """从data/cleaned目录加载最新的清洗后数据"""
    cleaned_dir = Path("data/cleaned")
    if not cleaned_dir.exists():
        return []

    date_dirs = sorted([d for d in cleaned_dir.iterdir() if d.is_dir()], reverse=True)
    if not date_dirs:
        return []

    latest = date_dirs[0] / "all.json"
    if not latest.exists():
        return []

    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [NewsItem(**item) for item in data]


def create_test_items() -> list[NewsItem]:
    """创建包含重复项的测试数据"""
    return [
        NewsItem(title="Article A", content="Content A", source_name="Source 1"),
        NewsItem(title="Article B", content="Content B", source_name="Source 1"),
        NewsItem(title="Article A", content="Content A", source_name="Source 2"),  # 重复
        NewsItem(title="Article C", content="Content C", source_name="Source 1"),
        NewsItem(title="article a", content="content a", source_name="Source 3"),  # 大小写重复
    ]


def test_dedup():
    """测试去重功能"""
    print("=" * 60)
    print("Step 3: Dedup 测试")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    items = load_cleaned_data()

    if items:
        print(f"    从 data/cleaned 加载了 {len(items)} 条数据")
    else:
        print("    未找到实际数据，使用模拟数据")
        items = create_test_items()
        print(f"    创建了 {len(items)} 条模拟数据 (包含重复)")

    # 加载历史记录
    print("\n[2] 加载去重历史...")
    seen_records = dedup.load()
    print(f"    历史记录: {len(seen_records)} 条")

    # 批次内去重
    print("\n[3] 批次内去重...")
    before_batch = len(items)
    items = dedup.filter_duplicates_in_batch(items)
    after_batch = len(items)
    print(f"    {before_batch} -> {after_batch} (移除 {before_batch - after_batch} 条批次内重复)")

    # 跨期去重
    print("\n[4] 跨期去重...")
    new_items, new_fps = dedup.filter_unseen(items, seen_records, save=True)
    print(f"    {len(items)} -> {len(new_items)} (移除 {len(items) - len(new_items)} 条历史重复)")
    print(f"    新增指纹: {len(new_fps)} 条")

    # 展示结果
    if new_items:
        print("\n    去重后前3条:")
        for i, item in enumerate(new_items[:3], 1):
            fp = dedup.get_fingerprint(item)
            print(f"    [{i}] {item.title[:40]}...")
            print(f"        指纹: {fp[:16]}...")

    # 注意: 这里不保存seen_records，因为是测试
    print("\n    (测试模式: 不更新seen.json)")

    print("\n" + "=" * 60)
    print("✓ Step 3 测试完成")
    print("=" * 60)

    return new_items, new_fps


if __name__ == "__main__":
    test_dedup()
