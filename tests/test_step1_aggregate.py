#!/usr/bin/env python
"""Step 1: Aggregate - 单元测试"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
aggregate = importlib.import_module("src.S1_aggregate")


class TestLoadSources:
    """测试数据源加载"""

    def test_load_sources_returns_list(self):
        """load_sources 应返回列表"""
        sources = aggregate.load_sources()
        assert isinstance(sources, list)

    def test_sources_have_required_fields(self):
        """每个源应有必要字段"""
        sources = aggregate.load_sources()
        for src in sources:
            assert "name" in src or "url" in src


class TestResolveMaxWorkers:
    """测试并发数解析"""

    def test_default_max_workers(self):
        """默认应返回 DEFAULT_MAX_WORKERS"""
        result = aggregate._resolve_max_workers(10)
        assert result <= aggregate.DEFAULT_MAX_WORKERS

    def test_max_workers_capped_by_sources(self):
        """并发数不应超过源数量"""
        result = aggregate._resolve_max_workers(2)
        assert result <= 2

    def test_min_workers_is_one(self):
        """最少应有 1 个 worker"""
        result = aggregate._resolve_max_workers(1)
        assert result >= 1

    def test_env_override(self):
        """环境变量应能覆盖默认值"""
        import os
        old_val = os.environ.get("AGGREGATE_MAX_WORKERS")
        try:
            os.environ["AGGREGATE_MAX_WORKERS"] = "2"
            result = aggregate._resolve_max_workers(10)
            assert result == 2
        finally:
            if old_val:
                os.environ["AGGREGATE_MAX_WORKERS"] = old_val
            else:
                os.environ.pop("AGGREGATE_MAX_WORKERS", None)


class TestFetchSingleSource:
    """测试单源抓取"""

    def test_unknown_source_type_raises(self):
        """未知源类型应抛出异常"""
        import pytest
        with pytest.raises(ValueError, match="Unknown source type"):
            aggregate._fetch_single_source({"type": "unknown"}, save_raw=False)


class TestFetchAll:
    """测试批量抓取"""

    def test_empty_sources_returns_empty(self):
        """空源列表应返回空结果"""
        result = aggregate.fetch_all(sources=[], save_raw=False)
        assert result == []

    def test_fetch_all_returns_list(self):
        """fetch_all 应返回列表"""
        # Mock both rss.fetch and pubmed.fetch
        with patch.object(aggregate.rss, 'fetch', return_value=[]):
            with patch.object(aggregate.pubmed, 'fetch', return_value=[]):
                result = aggregate.fetch_all(
                    sources=[{"name": "test", "type": "rss", "url": "http://test.com"}],
                    save_raw=False
                )
                assert isinstance(result, list)


class TestModuleExports:
    """测试模块导出"""

    def test_exports_fetch_all(self):
        """应导出 fetch_all"""
        assert hasattr(aggregate, "fetch_all")
        assert callable(aggregate.fetch_all)

    def test_exports_load_sources(self):
        """应导出 load_sources"""
        assert hasattr(aggregate, "load_sources")
        assert callable(aggregate.load_sources)


def test_aggregate():
    """集成测试"""
    print("=" * 60)
    print("Step 1: Aggregate 单元测试")
    print("=" * 60)

    # Test 1: Load sources
    print("\n[1] 测试加载数据源...")
    sources = aggregate.load_sources()
    assert isinstance(sources, list)
    print(f"    ✓ 加载了 {len(sources)} 个数据源")

    # Test 2: Max workers
    print("\n[2] 测试并发数计算...")
    workers = aggregate._resolve_max_workers(10)
    assert 1 <= workers <= 10
    print(f"    ✓ 并发数: {workers}")

    # Test 3: Empty fetch
    print("\n[3] 测试空源列表...")
    result = aggregate.fetch_all(sources=[], save_raw=False)
    assert result == []
    print("    ✓ 空源返回空列表")

    print("\n" + "=" * 60)
    print("✓ Step 1 单元测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_aggregate()

    print("\n运行 pytest...")
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
