#!/usr/bin/env python
"""Step 3: Dedup - 单元测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import NewsItem
from src.S3_dedup.fingerprint import (
    normalize_doi,
    normalize_title,
    hash_title,
    get_fingerprint,
)
from src.S3_dedup import (
    filter_duplicates_in_batch,
    filter_unseen,
    load,
    mark_seen,
    mark_batch,
    cleanup,
)


class TestNormalizeDoi:
    """测试 DOI 标准化"""

    def test_removes_doi_prefix(self):
        """应移除 doi: 前缀"""
        assert normalize_doi("doi:10.1038/xxx") == "10.1038/xxx"
        assert normalize_doi("DOI:10.1038/xxx") == "10.1038/xxx"

    def test_removes_url_prefix(self):
        """应移除 URL 前缀"""
        assert normalize_doi("https://doi.org/10.1038/xxx") == "10.1038/xxx"
        assert normalize_doi("http://doi.org/10.1038/xxx") == "10.1038/xxx"

    def test_lowercases(self):
        """应转为小写"""
        assert normalize_doi("10.1038/XXX") == "10.1038/xxx"

    def test_strips_whitespace(self):
        """应去除首尾空白"""
        assert normalize_doi("  10.1038/xxx  ") == "10.1038/xxx"

    def test_empty_input(self):
        """空输入应返回空字符串"""
        assert normalize_doi("") == ""
        assert normalize_doi(None) == ""


class TestNormalizeTitle:
    """测试标题标准化"""

    def test_lowercases(self):
        """应转为小写"""
        assert "hello" in normalize_title("Hello World")

    def test_removes_punctuation(self):
        """应移除标点"""
        result = normalize_title("Hello, World!")
        assert "," not in result
        assert "!" not in result

    def test_collapses_whitespace(self):
        """应合并空白"""
        result = normalize_title("hello   world")
        assert "  " not in result

    def test_empty_input(self):
        """空输入应返回空字符串"""
        assert normalize_title("") == ""
        assert normalize_title(None) == ""


class TestHashTitle:
    """测试标题哈希"""

    def test_returns_md5(self):
        """应返回 MD5 哈希"""
        result = hash_title("Hello World")
        assert len(result) == 32  # MD5 hex length

    def test_same_title_same_hash(self):
        """相同标题应返回相同哈希"""
        h1 = hash_title("Hello World")
        h2 = hash_title("Hello World")
        assert h1 == h2

    def test_case_insensitive(self):
        """大小写不敏感"""
        h1 = hash_title("Hello World")
        h2 = hash_title("hello world")
        assert h1 == h2

    def test_punctuation_insensitive(self):
        """标点不敏感"""
        h1 = hash_title("Hello, World!")
        h2 = hash_title("Hello World")
        assert h1 == h2

    def test_empty_input(self):
        """空输入应返回空字符串"""
        assert hash_title("") == ""
        assert hash_title(None) == ""


class TestGetFingerprint:
    """测试指纹获取"""

    def test_prefers_doi(self):
        """有 DOI 时应优先使用 DOI"""
        item = NewsItem(
            title="Test Article",
            doi="10.1038/test",
            source_name="Test",
        )
        fp = get_fingerprint(item)
        assert fp == "10.1038/test"

    def test_fallback_to_title_hash(self):
        """无 DOI 时应使用标题哈希"""
        item = NewsItem(
            title="Test Article",
            source_name="Test",
        )
        fp = get_fingerprint(item)
        assert len(fp) == 32  # MD5 hash

    def test_empty_item(self):
        """空条目应返回空字符串"""
        item = NewsItem(
            title="",
            source_name="Test",
        )
        fp = get_fingerprint(item)
        assert fp == ""


class TestFilterDuplicatesInBatch:
    """测试批次内去重"""

    def test_removes_exact_duplicates(self):
        """应移除完全重复"""
        items = [
            NewsItem(title="Article A", content="Content", source_name="S1"),
            NewsItem(title="Article B", content="Content", source_name="S1"),
            NewsItem(title="Article A", content="Content", source_name="S2"),
        ]
        result = filter_duplicates_in_batch(items)
        assert len(result) == 2

    def test_case_insensitive(self):
        """大小写不敏感"""
        items = [
            NewsItem(title="Article A", content="Content", source_name="S1"),
            NewsItem(title="article a", content="Content", source_name="S2"),
        ]
        result = filter_duplicates_in_batch(items)
        assert len(result) == 1

    def test_empty_list(self):
        """空列表应返回空列表"""
        result = filter_duplicates_in_batch([])
        assert result == []

    def test_no_duplicates(self):
        """无重复时应保留全部"""
        items = [
            NewsItem(title="Article A", content="Content", source_name="S1"),
            NewsItem(title="Article B", content="Content", source_name="S1"),
        ]
        result = filter_duplicates_in_batch(items)
        assert len(result) == 2


class TestFilterUnseen:
    """测试跨期去重"""

    def test_filters_seen_items(self):
        """应过滤已见条目"""
        items = [
            NewsItem(title="New Article", source_name="Test"),
            NewsItem(title="Old Article", source_name="Test"),
        ]
        # 预先标记一个为已见
        old_fp = get_fingerprint(items[1])
        seen = {old_fp: "2024-01-01"}

        new_items, new_fps = filter_unseen(items, seen, save=False)
        assert len(new_items) == 1
        assert new_items[0].title == "New Article"

    def test_returns_new_fingerprints(self):
        """应返回新指纹列表"""
        items = [
            NewsItem(title="New Article", source_name="Test"),
        ]
        seen = {}

        new_items, new_fps = filter_unseen(items, seen, save=False)
        assert len(new_fps) == 1

    def test_empty_seen(self):
        """空历史应全部通过"""
        items = [
            NewsItem(title="Article A", source_name="Test"),
            NewsItem(title="Article B", source_name="Test"),
        ]
        new_items, new_fps = filter_unseen(items, {}, save=False)
        assert len(new_items) == 2


class TestSeenRecords:
    """测试已见记录管理"""

    def test_mark_seen(self):
        """应标记为已见"""
        seen = {}
        mark_seen(seen, "test_fp")
        assert "test_fp" in seen

    def test_mark_batch(self):
        """应批量标记"""
        seen = {}
        fps = ["fp1", "fp2", "fp3"]
        mark_batch(seen, fps)
        assert all(fp in seen for fp in fps)

    def test_cleanup_old_records(self):
        """应清理过期记录"""
        from datetime import datetime, timedelta

        seen = {
            "new_fp": datetime.now().strftime("%Y-%m-%d"),
            "old_fp": (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d"),
        }
        cleaned = cleanup(seen, max_age_days=30)
        assert "new_fp" in cleaned
        assert "old_fp" not in cleaned


class TestModuleExports:
    """测试模块导出"""

    def test_exports_get_fingerprint(self):
        """应导出 get_fingerprint"""
        from src.S3_dedup import get_fingerprint
        assert callable(get_fingerprint)

    def test_exports_filter_functions(self):
        """应导出过滤函数"""
        from src.S3_dedup import filter_unseen, filter_duplicates_in_batch
        assert callable(filter_unseen)
        assert callable(filter_duplicates_in_batch)


def test_dedup():
    """集成测试"""
    print("=" * 60)
    print("Step 3: Dedup 单元测试")
    print("=" * 60)

    # Test 1: DOI normalization
    print("\n[1] 测试 DOI 标准化...")
    assert normalize_doi("doi:10.1038/XXX") == "10.1038/xxx"
    assert normalize_doi("https://doi.org/10.1038/test") == "10.1038/test"
    print("    ✓ DOI 标准化正常")

    # Test 2: Title hashing
    print("\n[2] 测试标题哈希...")
    h1 = hash_title("Hello, World!")
    h2 = hash_title("hello world")
    assert h1 == h2
    print(f"    ✓ 标题哈希: {h1[:16]}...")

    # Test 3: Fingerprint
    print("\n[3] 测试指纹计算...")
    item_with_doi = NewsItem(title="Test", doi="10.1038/test", source_name="Test")
    item_no_doi = NewsItem(title="Test Article", source_name="Test")
    fp1 = get_fingerprint(item_with_doi)
    fp2 = get_fingerprint(item_no_doi)
    assert fp1 == "10.1038/test"
    assert len(fp2) == 32
    print(f"    ✓ DOI 指纹: {fp1}")
    print(f"    ✓ 标题指纹: {fp2[:16]}...")

    # Test 4: Batch dedup
    print("\n[4] 测试批次内去重...")
    items = [
        NewsItem(title="Article A", source_name="S1"),
        NewsItem(title="Article B", source_name="S1"),
        NewsItem(title="Article A", source_name="S2"),
        NewsItem(title="article a", source_name="S3"),
    ]
    result = filter_duplicates_in_batch(items)
    assert len(result) == 2
    print(f"    ✓ {len(items)} 条去重后剩 {len(result)} 条")

    # Test 5: Cross-period dedup
    print("\n[5] 测试跨期去重...")
    seen = {get_fingerprint(items[0]): "2024-01-01"}
    new_items, new_fps = filter_unseen(result, seen, save=False)
    print(f"    ✓ {len(result)} 条过滤后剩 {len(new_items)} 条新条目")

    print("\n" + "=" * 60)
    print("✓ Step 3 单元测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_dedup()

    print("\n运行 pytest...")
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
