#!/usr/bin/env python
"""Step 6: LLM Summarize - 测试 LLM 摘要生成功能"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import NewsItem
from src.S6_llm.process import (
    _smart_truncate,
    _check_vip_content,
    _post_process_summary,
    summarize_single,
    summarize_batch,
    SummarizeStats,
)


class TestSmartTruncate:
    """测试智能截断功能"""

    def test_short_content_unchanged(self):
        """短内容不应被截断"""
        content = "This is a short abstract."
        result = _smart_truncate(content, max_length=1500)
        assert result == content

    def test_empty_content(self):
        """空内容返回空字符串"""
        assert _smart_truncate("", max_length=100) == ""
        assert _smart_truncate(None, max_length=100) is None

    def test_long_content_truncated(self):
        """长内容应被截断到指定长度"""
        content = "x" * 2000
        result = _smart_truncate(content, max_length=500)
        assert len(result) <= 500

    def test_abstract_extraction(self):
        """应提取 Abstract 部分"""
        content = """
        Some introduction text here.

        Abstract: This is a groundbreaking study that investigates the molecular mechanisms of RNA splicing in mammalian cells. We discovered novel interactions.

        Introduction: More detailed background information follows here.
        """
        result = _smart_truncate(content, max_length=300)
        assert "groundbreaking study" in result or "RNA splicing" in result


class TestCheckVipContent:
    """测试 VIP 关键词检测"""

    def test_vip_keyword_in_title(self):
        """标题中的 VIP 关键词应被检测"""
        assert _check_vip_content("", "tRNA splicing mechanism") is True
        assert _check_vip_content("", "RTCB ligase function") is True

    def test_vip_keyword_in_content(self):
        """内容中的 VIP 关键词应被检测"""
        assert _check_vip_content("Study of IRE1 activation pathway", "Normal title") is True
        assert _check_vip_content("XBP1 splicing in ER stress", "Another title") is True

    def test_case_insensitive(self):
        """大小写不敏感"""
        assert _check_vip_content("TRNA modification", "") is True
        assert _check_vip_content("rtcb enzyme", "") is True

    def test_no_vip_keyword(self):
        """无 VIP 关键词应返回 False"""
        assert _check_vip_content("General biology research", "Normal science title") is False


class TestPostProcessSummary:
    """测试摘要后处理"""

    def test_strip_whitespace(self):
        """应去除首尾空白"""
        result = _post_process_summary("  Some summary text.  ", is_vip_content=False)
        assert result == "Some summary text."

    def test_remove_prefix(self):
        """应移除常见前缀"""
        result = _post_process_summary("摘要：这是一段摘要。", is_vip_content=False)
        assert not result.startswith("摘要")

    def test_add_vip_tag(self):
        """VIP 内容应添加【关联】标注"""
        result = _post_process_summary("这是关于 tRNA 的研究", is_vip_content=True)
        assert "【关联】" in result

    def test_no_duplicate_vip_tag(self):
        """已有【关联】标注不应重复添加"""
        result = _post_process_summary("这是研究成果。【关联】", is_vip_content=True)
        assert result.count("【关联】") == 1

    def test_empty_summary(self):
        """空摘要应返回空字符串"""
        assert _post_process_summary("", is_vip_content=False) == ""


class TestSummarizeSingle:
    """测试单条摘要生成 - 需要 mock chat 函数"""

    def test_successful_summary_logic(self):
        """测试摘要成功时的逻辑（检查函数存在且可调用）"""
        # 验证函数签名和基本结构
        import inspect
        sig = inspect.signature(summarize_single)
        params = list(sig.parameters.keys())
        assert "item" in params
        assert "user_context" in params

    def test_retry_logic_constants(self):
        """测试重试相关常量"""
        from src.S6_llm.process import MIN_SUMMARY_LENGTH, MAX_RETRIES
        assert MIN_SUMMARY_LENGTH == 50
        assert MAX_RETRIES == 2

    def test_handle_api_error(self):
        """API 错误应被捕获"""
        import src.S6_llm.process as proc
        original_chat = proc.chat

        def mock_chat(*args, **kwargs):
            raise Exception("API Error")

        proc.chat = mock_chat
        try:
            item = NewsItem(
                title="Test Article",
                content="Test content",
                source_name="Cell",
            )
            result = proc.summarize_single(item)
            assert result.summary == ""
            assert hasattr(result, "_summarize_error")
        finally:
            proc.chat = original_chat


class TestSummarizeBatch:
    """测试批量摘要生成"""

    def test_empty_batch(self):
        """空列表应返回空结果"""
        items, stats = summarize_batch([])
        assert items == []
        assert stats.total == 0
        assert stats.success == 0

    def test_batch_stats_dataclass(self):
        """测试 SummarizeStats 数据类"""
        stats = SummarizeStats(total=10, success=8, failed=2, retried=1)
        assert stats.total == 10
        assert stats.success == 8
        assert stats.failed == 2
        assert stats.retried == 1

    def test_batch_function_signature(self):
        """测试 summarize_batch 函数签名"""
        import inspect
        sig = inspect.signature(summarize_batch)
        params = list(sig.parameters.keys())
        assert "items" in params
        assert "concurrency" in params

    def test_concurrency_config(self):
        """测试并发配置"""
        from src.S6_llm.process import get_concurrency
        concurrency = get_concurrency()
        assert isinstance(concurrency, int)
        assert concurrency >= 1


def test_filter():
    """集成测试：检验模块导入和基本功能"""
    print("=" * 60)
    print("Step 6: LLM Summarize 测试")
    print("=" * 60)

    # Test 1: Smart truncate
    print("\n[1] 测试智能截断...")
    test_content = "x" * 2000
    truncated = _smart_truncate(test_content, max_length=500)
    assert len(truncated) <= 500
    print(f"    ✓ 2000字符截断到 {len(truncated)} 字符")

    # Test 2: VIP detection
    print("\n[2] 测试 VIP 关键词检测...")
    assert _check_vip_content("tRNA splicing study", "") is True
    assert _check_vip_content("general biology", "") is False
    print("    ✓ VIP 关键词检测正常")

    # Test 3: Post-process
    print("\n[3] 测试摘要后处理...")
    processed = _post_process_summary("摘要：测试内容", is_vip_content=False)
    assert not processed.startswith("摘要")
    print("    ✓ 前缀移除正常")

    processed_vip = _post_process_summary("关于 tRNA 的研究发现", is_vip_content=True)
    assert "【关联】" in processed_vip
    print("    ✓ VIP 标注添加正常")

    # Test 4: Stats dataclass
    print("\n[4] 测试统计类...")
    stats = SummarizeStats(total=10, success=8, failed=2, retried=1)
    assert stats.total == 10
    assert stats.success == 8
    print(f"    ✓ 统计: total={stats.total}, success={stats.success}, failed={stats.failed}")

    print("\n" + "=" * 60)
    print("✓ Step 6 单元测试全部通过")
    print("=" * 60)


if __name__ == "__main__":
    # 运行基础测试
    test_filter()

    # 运行 pytest 风格测试
    print("\n运行详细单元测试...")
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
