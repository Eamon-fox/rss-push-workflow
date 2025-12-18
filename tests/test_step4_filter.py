#!/usr/bin/env python
"""Step 4: Hybrid Filter - 单元测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import NewsItem
from src.S4_filter.hybrid import (
    _item_text,
    _find_vip_keywords,
    _aggregate_semantic_score,
    _sort_by_semantic,
    HybridFilterStats,
)
from src.S4_filter.config import (
    GENERAL_BIO_KEYWORDS,
    VIP_KEYWORDS,
    SEMANTIC_ANCHORS,
    THRESHOLD_NORMAL,
)


class TestItemText:
    """测试文本提取"""

    def test_combines_title_and_content(self):
        """应合并标题和内容"""
        item = NewsItem(
            title="Test Title",
            content="Test Content",
            source_name="Test",
        )
        text = _item_text(item)
        assert "Test Title" in text
        assert "Test Content" in text

    def test_handles_empty_content(self):
        """应处理空内容"""
        item = NewsItem(
            title="Test Title",
            content="",
            source_name="Test",
        )
        text = _item_text(item)
        assert "Test Title" in text

    def test_truncates_long_text(self):
        """应截断长文本"""
        item = NewsItem(
            title="Title",
            content="x" * 3000,
            source_name="Test",
        )
        text = _item_text(item, max_len=100)
        assert len(text) <= 100


class TestFindVipKeywords:
    """测试 VIP 关键词查找"""

    def test_finds_rtcb(self):
        """应找到 RTCB 关键词"""
        text = "RTCB is an essential RNA ligase enzyme"
        keywords = _find_vip_keywords(text)
        assert len(keywords) > 0  # RTCB 是 VIP 关键词

    def test_finds_ire1(self):
        """应找到 IRE1 关键词"""
        text = "IRE1 activation during ER stress"
        keywords = _find_vip_keywords(text)
        assert len(keywords) > 0  # IRE1 是 VIP 关键词

    def test_finds_xbp1(self):
        """应找到 XBP1 关键词"""
        text = "XBP1 splicing in unfolded protein response"
        keywords = _find_vip_keywords(text)
        assert len(keywords) > 0  # XBP1 是 VIP 关键词

    def test_no_match(self):
        """无匹配应返回空列表"""
        text = "General biology research on plants"
        keywords = _find_vip_keywords(text)
        assert keywords == []


class TestAggregateSemanticScore:
    """测试语义分数聚合"""

    def test_single_score(self):
        """单个分数应直接返回"""
        score = _aggregate_semantic_score([0.8])
        assert score == 0.8

    def test_empty_scores(self):
        """空分数应返回 0"""
        score = _aggregate_semantic_score([])
        assert score == 0.0

    def test_multiple_scores_boost(self):
        """多个高分应有轻微加成"""
        single = _aggregate_semantic_score([0.7])
        multiple = _aggregate_semantic_score([0.7, 0.6, 0.5])
        # Multiple hits should give a small boost
        assert multiple >= single

    def test_capped_at_one(self):
        """分数应限制在 1.0 以内"""
        score = _aggregate_semantic_score([0.99, 0.95, 0.90])
        assert score <= 1.0


class TestSortBySemantic:
    """测试语义分数排序"""

    def test_sorts_descending(self):
        """应按分数降序排列"""
        items = [
            NewsItem(title="Low", source_name="Test", semantic_score=0.3),
            NewsItem(title="High", source_name="Test", semantic_score=0.9),
            NewsItem(title="Mid", source_name="Test", semantic_score=0.6),
        ]
        sorted_items = _sort_by_semantic(items)
        assert sorted_items[0].title == "High"
        assert sorted_items[1].title == "Mid"
        assert sorted_items[2].title == "Low"

    def test_handles_none_scores(self):
        """应处理 None 分数"""
        items = [
            NewsItem(title="Scored", source_name="Test", semantic_score=0.5),
            NewsItem(title="Unscored", source_name="Test", semantic_score=None),
        ]
        sorted_items = _sort_by_semantic(items)
        assert sorted_items[0].title == "Scored"

    def test_empty_list(self):
        """空列表应返回空列表"""
        assert _sort_by_semantic([]) == []

    def test_no_scores(self):
        """无分数时保持原顺序"""
        items = [
            NewsItem(title="A", source_name="Test"),
            NewsItem(title="B", source_name="Test"),
        ]
        sorted_items = _sort_by_semantic(items)
        assert sorted_items[0].title == "A"
        assert sorted_items[1].title == "B"


class TestHybridFilterStats:
    """测试过滤统计"""

    def test_stats_dataclass(self):
        """应正确创建统计数据"""
        stats = HybridFilterStats(
            total=100,
            layer1_dropped=20,
            layer2_vip_kept=10,
            layer2_semantic_kept=50,
            layer2_semantic_dropped=20,
        )
        assert stats.total == 100
        assert stats.layer1_dropped == 20
        assert stats.layer2_vip_kept == 10

    def test_stats_frozen(self):
        """统计数据应不可变"""
        stats = HybridFilterStats(
            total=100,
            layer1_dropped=20,
            layer2_vip_kept=10,
            layer2_semantic_kept=50,
            layer2_semantic_dropped=20,
        )
        import pytest
        with pytest.raises(Exception):  # FrozenInstanceError
            stats.total = 200


class TestConfig:
    """测试配置常量"""

    def test_bio_keywords_exist(self):
        """应有生物关键词列表"""
        assert isinstance(GENERAL_BIO_KEYWORDS, (list, tuple))
        assert len(GENERAL_BIO_KEYWORDS) > 0

    def test_vip_keywords_exist(self):
        """应有 VIP 关键词列表"""
        assert isinstance(VIP_KEYWORDS, (list, tuple))
        assert len(VIP_KEYWORDS) > 0

    def test_semantic_anchors_exist(self):
        """应有语义锚点"""
        assert isinstance(SEMANTIC_ANCHORS, (list, tuple))

    def test_threshold_valid(self):
        """阈值应在有效范围"""
        assert 0 <= THRESHOLD_NORMAL <= 1


class TestModuleExports:
    """测试模块导出"""

    def test_exports_filter_hybrid(self):
        """应导出 filter_hybrid"""
        from src.S4_filter import filter_hybrid
        assert callable(filter_hybrid)


def test_filter():
    """集成测试"""
    print("=" * 60)
    print("Step 4: Hybrid Filter 单元测试")
    print("=" * 60)

    # Test 1: Item text extraction
    print("\n[1] 测试文本提取...")
    item = NewsItem(
        title="Test Title",
        content="Test Content",
        source_name="Test",
    )
    text = _item_text(item)
    assert "Test Title" in text
    assert "Test Content" in text
    print("    ✓ 文本提取正常")

    # Test 2: VIP keywords
    print("\n[2] 测试 VIP 关键词...")
    vip_text = "tRNA splicing and RTCB ligase"
    keywords = _find_vip_keywords(vip_text)
    print(f"    ✓ 找到 VIP 关键词: {keywords}")

    # Test 3: Semantic score aggregation
    print("\n[3] 测试语义分数聚合...")
    single = _aggregate_semantic_score([0.7])
    multiple = _aggregate_semantic_score([0.7, 0.6, 0.5])
    assert single == 0.7
    assert multiple >= single
    print(f"    ✓ 单分数: {single}, 多分数: {multiple:.4f}")

    # Test 4: Sorting
    print("\n[4] 测试分数排序...")
    items = [
        NewsItem(title="Low", source_name="Test", semantic_score=0.3),
        NewsItem(title="High", source_name="Test", semantic_score=0.9),
    ]
    sorted_items = _sort_by_semantic(items)
    assert sorted_items[0].semantic_score > sorted_items[1].semantic_score
    print("    ✓ 排序正常（降序）")

    # Test 5: Stats
    print("\n[5] 测试统计数据类...")
    stats = HybridFilterStats(
        total=100,
        layer1_dropped=20,
        layer2_vip_kept=10,
        layer2_semantic_kept=50,
        layer2_semantic_dropped=20,
    )
    assert stats.total == 100
    print(f"    ✓ 统计: total={stats.total}, vip={stats.layer2_vip_kept}")

    # Test 6: Config
    print("\n[6] 测试配置...")
    assert len(GENERAL_BIO_KEYWORDS) > 0
    assert len(VIP_KEYWORDS) > 0
    assert 0 <= THRESHOLD_NORMAL <= 1
    print(f"    ✓ 生物关键词: {len(GENERAL_BIO_KEYWORDS)} 个")
    print(f"    ✓ VIP 关键词: {len(VIP_KEYWORDS)} 个")
    print(f"    ✓ 阈值: {THRESHOLD_NORMAL}")

    print("\n" + "=" * 60)
    print("✓ Step 4 单元测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_filter()

    print("\n运行 pytest...")
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
