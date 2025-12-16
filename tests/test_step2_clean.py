#!/usr/bin/env python
"""Step 2: Clean - 测试内容清洗功能"""

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

clean = importlib.import_module("src.S2_clean")
from src.models import NewsItem


def load_raw_data() -> list[NewsItem]:
    """从data/raw目录加载最新的原始数据"""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return []

    # 找最新日期的目录
    date_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()], reverse=True)
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
        NewsItem(
            title="  <b>Test Article</b> with &amp; HTML entities  ",
            content="<p>This is a <strong>test</strong> paragraph.</p>\n\n\n  Multiple   spaces here.",
            link="  https://example.com/article1  ",
            source_name="Test Source",
        ),
        NewsItem(
            title="Normal Title",
            content="A" * 3000,  # 超长内容，应该被截断
            link="https://example.com/article2",
            source_name="Test Source",
        ),
    ]


def test_clean():
    """测试内容清洗"""
    print("=" * 60)
    print("Step 2: Clean 测试")
    print("=" * 60)

    # 尝试从实际数据加载
    print("\n[1] 加载测试数据...")
    items = load_raw_data()

    if items:
        print(f"    从 data/raw 加载了 {len(items)} 条数据")
    else:
        print("    未找到实际数据，使用模拟数据")
        items = create_test_items()
        print(f"    创建了 {len(items)} 条模拟数据")

    # 执行清洗
    print("\n[2] 执行清洗...")
    cleaned_items = clean.batch_clean(items, save=True)

    # 结果对比
    print(f"\n[3] 清洗结果: {len(cleaned_items)} 条")

    if cleaned_items:
        print("\n    清洗效果示例 (前2条):")
        for i, (orig, cln) in enumerate(zip(items[:2], cleaned_items[:2]), 1):
            print(f"\n    [{i}] 标题清洗:")
            print(f"        前: {repr(orig.title[:50])}")
            print(f"        后: {repr(cln.title[:50])}")

            if orig.content:
                print(f"        内容长度: {len(orig.content)} -> {len(cln.content)}")

    print("\n" + "=" * 60)
    print("✓ Step 2 测试完成")
    print("=" * 60)

    return cleaned_items


if __name__ == "__main__":
    test_clean()
