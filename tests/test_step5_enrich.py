#!/usr/bin/env python
"""Step 5: Enrich - 单元测试"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import NewsItem
from src.S5_enrich.enrich import (
    UserProfile,
    EnrichStats,
    load_user_profile,
    enrich_batch,
)


class TestUserProfile:
    """测试用户配置数据类"""

    def test_default_values(self):
        """默认值应正确"""
        profile = UserProfile()
        assert profile.research_context == ""
        assert profile.core_topics == []
        assert profile.highlight_keywords == []
        assert profile.relevance_instruction == ""

    def test_custom_values(self):
        """自定义值应正确设置"""
        profile = UserProfile(
            research_context="tRNA research",
            core_topics=["tRNA", "RNA splicing"],
            highlight_keywords=["RTCB", "IRE1"],
        )
        assert profile.research_context == "tRNA research"
        assert "tRNA" in profile.core_topics
        assert "RTCB" in profile.highlight_keywords

    def test_none_lists_become_empty(self):
        """None 列表应变为空列表"""
        profile = UserProfile(core_topics=None, highlight_keywords=None)
        assert profile.core_topics == []
        assert profile.highlight_keywords == []


class TestEnrichStats:
    """测试增强统计数据类"""

    def test_default_values(self):
        """默认值应为 0"""
        stats = EnrichStats()
        assert stats.total == 0
        assert stats.need_enrich == 0
        assert stats.enriched == 0
        assert stats.failed == 0

    def test_custom_values(self):
        """自定义值应正确设置"""
        stats = EnrichStats(
            total=100,
            need_enrich=30,
            enriched=25,
            failed=5,
        )
        assert stats.total == 100
        assert stats.need_enrich == 30
        assert stats.enriched == 25
        assert stats.failed == 5


class TestLoadUserProfile:
    """测试用户配置加载"""

    def test_missing_file_returns_default(self):
        """文件不存在应返回默认配置"""
        profile = load_user_profile("nonexistent_file.yaml")
        assert isinstance(profile, UserProfile)
        assert profile.research_context == ""

    def test_loads_valid_yaml(self):
        """应正确加载 YAML 文件"""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "research_context": "Test context",
                "core_topics": ["topic1", "topic2"],
            }, f)
            path = f.name

        try:
            profile = load_user_profile(path)
            assert profile.research_context == "Test context"
            assert "topic1" in profile.core_topics
        finally:
            Path(path).unlink()

    def test_handles_empty_yaml(self):
        """应处理空 YAML 文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name

        try:
            profile = load_user_profile(path)
            assert isinstance(profile, UserProfile)
        finally:
            Path(path).unlink()


class TestEnrichBatch:
    """测试批量增强"""

    def test_empty_list(self):
        """空列表应返回空结果"""
        items, stats = enrich_batch([])
        assert items == []
        assert stats.total == 0

    def test_sufficient_content_skipped(self):
        """内容足够的条目应跳过"""
        items = [
            NewsItem(
                title="Test Article",
                content="x" * 200,  # 超过 min_content_len
                source_name="Test",
            ),
        ]
        result, stats = enrich_batch(items, min_content_len=150)
        assert stats.need_enrich == 0

    def test_short_content_without_doi_skipped(self):
        """无 DOI 的短内容不需要增强"""
        items = [
            NewsItem(
                title="Test Article",
                content="Short",
                source_name="Test",
                # No DOI
            ),
        ]
        result, stats = enrich_batch(items, min_content_len=150)
        assert stats.need_enrich == 0  # No DOI means can't enrich

    def test_short_content_with_doi_needs_enrich(self):
        """有 DOI 的短内容需要增强"""
        items = [
            NewsItem(
                title="Test Article",
                content="Short",
                source_name="Test",
                doi="10.1234/test",
            ),
        ]
        result, stats = enrich_batch(items, min_content_len=150)
        assert stats.need_enrich == 1

    def test_returns_same_length(self):
        """应返回相同长度的列表"""
        items = [
            NewsItem(title=f"Article {i}", content="x" * 200, source_name="Test")
            for i in range(5)
        ]
        result, stats = enrich_batch(items, min_content_len=150)
        assert len(result) == len(items)


class TestModuleExports:
    """测试模块导出"""

    def test_exports_enrich_batch(self):
        """应导出 enrich_batch"""
        from src.S5_enrich import enrich_batch
        assert callable(enrich_batch)

    def test_exports_load_user_profile(self):
        """应导出 load_user_profile"""
        from src.S5_enrich import load_user_profile
        assert callable(load_user_profile)


def test_enrich():
    """集成测试"""
    print("=" * 60)
    print("Step 5: Enrich 单元测试")
    print("=" * 60)

    # Test 1: UserProfile
    print("\n[1] 测试 UserProfile 数据类...")
    profile = UserProfile(
        research_context="tRNA research",
        core_topics=["tRNA", "splicing"],
    )
    assert profile.research_context == "tRNA research"
    print(f"    ✓ 研究背景: {profile.research_context}")
    print(f"    ✓ 核心主题: {profile.core_topics}")

    # Test 2: EnrichStats
    print("\n[2] 测试 EnrichStats 数据类...")
    stats = EnrichStats(total=100, need_enrich=30, enriched=25, failed=5)
    assert stats.total == 100
    print(f"    ✓ 统计: total={stats.total}, enriched={stats.enriched}")

    # Test 3: Load profile
    print("\n[3] 测试配置加载...")
    profile = load_user_profile("nonexistent.yaml")
    assert isinstance(profile, UserProfile)
    print("    ✓ 缺失文件返回默认配置")

    # Test 4: Empty batch
    print("\n[4] 测试空列表...")
    items, stats = enrich_batch([])
    assert items == []
    assert stats.total == 0
    print("    ✓ 空列表返回空结果")

    # Test 5: Sufficient content
    print("\n[5] 测试内容足够的条目...")
    items = [
        NewsItem(title="Test", content="x" * 200, source_name="Test"),
    ]
    result, stats = enrich_batch(items, min_content_len=150)
    assert stats.need_enrich == 0
    print("    ✓ 内容足够时跳过增强")

    # Test 6: Short content with DOI
    print("\n[6] 测试短内容条目...")
    items = [
        NewsItem(title="Test", content="Short", source_name="Test", doi="10.1234/test"),
    ]
    result, stats = enrich_batch(items, min_content_len=150)
    assert stats.need_enrich == 1
    print(f"    ✓ 需要增强: {stats.need_enrich} 条")

    print("\n" + "=" * 60)
    print("✓ Step 5 单元测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_enrich()

    print("\n运行 pytest...")
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
