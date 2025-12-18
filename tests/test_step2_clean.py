#!/usr/bin/env python
"""Step 2: Clean - 单元测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import NewsItem
from src.S2_clean.html import (
    clean_html,
    clean_whitespace,
    truncate,
    clean_text,
    clean,
    batch_clean,
)


class TestCleanHtml:
    """测试 HTML 清理"""

    def test_removes_tags(self):
        """应移除 HTML 标签"""
        result = clean_html("<p>Hello <b>World</b></p>")
        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "World" in result

    def test_decodes_entities(self):
        """应解码 HTML 实体"""
        assert clean_html("&amp;") == "&"
        assert clean_html("&lt;") == "<"
        assert clean_html("&gt;") == ">"
        assert clean_html("&nbsp;") == "\xa0"  # non-breaking space
        assert clean_html("&quot;") == '"'

    def test_empty_input(self):
        """空输入应返回空字符串"""
        assert clean_html("") == ""
        assert clean_html(None) == ""

    def test_preserves_text(self):
        """应保留纯文本"""
        text = "Hello World"
        assert clean_html(text) == text


class TestCleanWhitespace:
    """测试空白符清理"""

    def test_collapses_spaces(self):
        """应合并多个空格"""
        assert clean_whitespace("hello   world") == "hello world"

    def test_strips_edges(self):
        """应去除首尾空白"""
        assert clean_whitespace("  hello  ") == "hello"

    def test_handles_newlines(self):
        """应处理换行符"""
        assert clean_whitespace("hello\n\nworld") == "hello world"

    def test_handles_tabs(self):
        """应处理制表符"""
        assert clean_whitespace("hello\t\tworld") == "hello world"

    def test_empty_input(self):
        """空输入应返回空字符串"""
        assert clean_whitespace("") == ""
        assert clean_whitespace(None) == ""


class TestTruncate:
    """测试文本截断"""

    def test_short_text_unchanged(self):
        """短文本不应被截断"""
        text = "Hello World"
        assert truncate(text, max_length=100) == text

    def test_long_text_truncated(self):
        """长文本应被截断"""
        text = "x" * 100
        result = truncate(text, max_length=50)
        assert len(result) <= 53  # 50 + "..."

    def test_adds_ellipsis(self):
        """截断后应添加省略号"""
        text = "x" * 100
        result = truncate(text, max_length=50)
        assert result.endswith("...")

    def test_empty_input(self):
        """空输入应返回空字符串"""
        assert truncate("", max_length=100) == ""
        assert truncate(None, max_length=100) is None

    def test_exact_length(self):
        """刚好等于限制长度不截断"""
        text = "x" * 50
        result = truncate(text, max_length=50)
        assert result == text
        assert "..." not in result


class TestCleanText:
    """测试完整清理流程"""

    def test_full_pipeline(self):
        """应执行完整清理流程"""
        text = "<p>  Hello   <b>World</b> &amp; More  </p>"
        result = clean_text(text, max_length=100)

        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "World" in result
        assert "&" in result
        assert "  " not in result  # no double spaces

    def test_truncates_after_cleaning(self):
        """应在清理后截断"""
        text = "<p>" + "x" * 3000 + "</p>"
        result = clean_text(text, max_length=100)
        assert len(result) <= 103


class TestCleanItem:
    """测试 NewsItem 清理"""

    def test_cleans_title(self):
        """应清理标题空白"""
        item = NewsItem(
            title="  Test Title  ",
            content="Content",
            source_name="Test",
        )
        result = clean(item)
        assert result.title == "Test Title"  # clean() 只处理空白，不处理 HTML

    def test_cleans_content(self):
        """应清理内容"""
        item = NewsItem(
            title="Title",
            content="<p>Test &amp; Content</p>",
            source_name="Test",
        )
        result = clean(item)
        assert "<p>" not in result.content
        assert "&" in result.content

    def test_strips_link(self):
        """应去除链接首尾空白"""
        item = NewsItem(
            title="Title",
            content="Content",
            link="  https://example.com  ",
            source_name="Test",
        )
        result = clean(item)
        assert result.link == "https://example.com"

    def test_preserves_metadata(self):
        """应保留元数据"""
        item = NewsItem(
            title="Title",
            content="Content",
            source_name="Nature",
            authors=["Author A", "Author B"],
            doi="10.1234/test",
        )
        result = clean(item)
        assert result.source_name == "Nature"
        assert result.authors == ["Author A", "Author B"]
        assert result.doi == "10.1234/test"


class TestBatchClean:
    """测试批量清理"""

    def test_cleans_all_items(self):
        """应清理所有条目"""
        items = [
            NewsItem(title="  Title 1  ", content="Content 1", source_name="Test"),
            NewsItem(title="  Title 2  ", content="Content 2", source_name="Test"),
        ]
        results = batch_clean(items, save=False)

        assert len(results) == 2
        assert results[0].title == "Title 1"
        assert results[1].title == "Title 2"

    def test_empty_list(self):
        """空列表应返回空列表"""
        results = batch_clean([], save=False)
        assert results == []


class TestModuleExports:
    """测试模块导出"""

    def test_exports_clean(self):
        """应导出 clean 函数"""
        from src.S2_clean import clean
        assert callable(clean)

    def test_exports_batch_clean(self):
        """应导出 batch_clean 函数"""
        from src.S2_clean import batch_clean
        assert callable(batch_clean)


def test_clean():
    """集成测试"""
    print("=" * 60)
    print("Step 2: Clean 单元测试")
    print("=" * 60)

    # Test 1: HTML cleaning
    print("\n[1] 测试 HTML 清理...")
    html_input = "<p>Hello &amp; <b>World</b></p>"
    result = clean_html(html_input)
    assert "Hello" in result
    assert "&" in result
    assert "<" not in result
    print(f"    ✓ '{html_input}' -> '{result}'")

    # Test 2: Whitespace
    print("\n[2] 测试空白符清理...")
    ws_input = "  hello   world  "
    result = clean_whitespace(ws_input)
    assert result == "hello world"
    print(f"    ✓ '{ws_input}' -> '{result}'")

    # Test 3: Truncate
    print("\n[3] 测试截断...")
    long_text = "x" * 100
    result = truncate(long_text, max_length=50)
    assert len(result) <= 53
    print(f"    ✓ 100字符截断到 {len(result)} 字符")

    # Test 4: Item cleaning
    print("\n[4] 测试 NewsItem 清理...")
    item = NewsItem(
        title="  Test Title  ",
        content="<p>Content with  spaces</p>",
        link="  https://example.com  ",
        source_name="Test",
    )
    cleaned = clean(item)
    assert cleaned.title == "Test Title"  # 空白已清理
    assert "<p>" not in cleaned.content  # HTML 已清理
    assert cleaned.link == "https://example.com"
    print("    ✓ NewsItem 清理正常")

    # Test 5: Batch
    print("\n[5] 测试批量清理...")
    items = [
        NewsItem(title=f"  Title {i}  ", content=f"Content {i}", source_name="Test")
        for i in range(3)
    ]
    results = batch_clean(items, save=False)
    assert len(results) == 3
    assert all(r.title.startswith("Title") for r in results)
    print(f"    ✓ 批量清理 {len(results)} 条")

    print("\n" + "=" * 60)
    print("✓ Step 2 单元测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_clean()

    print("\n运行 pytest...")
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
