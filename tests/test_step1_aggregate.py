#!/usr/bin/env python
"""Step 1: Aggregate - 测试RSS抓取功能"""

import importlib
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

aggregate = importlib.import_module("src.S1_aggregate")


def test_aggregate():
    """测试从所有RSS源抓取数据"""
    print("=" * 60)
    print("Step 1: Aggregate 测试")
    print("=" * 60)

    # 加载源配置
    print("\n[1] 加载数据源配置...")
    sources = aggregate.load_sources()
    print(f"    发现 {len(sources)} 个数据源:")
    for src in sources:
        print(f"    - {src.get('name', 'unknown')}: {src.get('type', 'rss')}")

    # 抓取所有数据
    print("\n[2] 开始抓取数据...")
    items = aggregate.fetch_all(save_raw=True)

    # 结果汇总
    print("\n[3] 抓取结果:")
    print(f"    总计: {len(items)} 条")

    if items:
        # 按来源统计
        by_source = {}
        for item in items:
            src = item.source_name
            by_source[src] = by_source.get(src, 0) + 1

        print("\n    按来源统计:")
        for src, count in sorted(by_source.items()):
            print(f"    - {src}: {count} 条")

        # 展示前3条
        print("\n    前3条示例:")
        for i, item in enumerate(items[:3], 1):
            print(f"\n    [{i}] {item.title[:60]}...")
            print(f"        来源: {item.source_name}")
            print(f"        链接: {item.link[:60]}..." if item.link else "        链接: (无)")

    print("\n" + "=" * 60)
    print("✓ Step 1 测试完成")
    print("=" * 60)

    return items


if __name__ == "__main__":
    test_aggregate()
