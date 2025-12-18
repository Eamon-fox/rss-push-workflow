#!/usr/bin/env python
"""Step 7: Deliver - 测试输出模块（JSON, Markdown, HTML）"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import NewsItem
from src.S7_deliver import to_json, to_markdown, to_html, to_html_file
from src.S7_deliver.html import _render_card, _render_stats, _get_css


def create_test_items() -> list[NewsItem]:
    """创建测试数据"""
    return [
        NewsItem(
            title="CRISPR-Cas9 enables precise genome editing in mammalian cells",
            content="A novel CRISPR technique developed by researchers allows for highly specific gene modifications with minimal off-target effects.",
            source_name="Nature",
            authors=["Zhang F", "Doudna JA", "Charpentier E"],
            doi="10.1038/nature12345",
            link="https://nature.com/articles/12345",
            semantic_score=0.85,
            is_vip=False,
            published_at=datetime(2024, 12, 15),
            summary="本研究开发了一种新型CRISPR技术，能够在哺乳动物细胞中实现精确的基因编辑，脱靶效应极低。",
        ),
        NewsItem(
            title="tRNA splicing mechanism revealed by structural biology",
            content="Cryo-EM structures reveal the molecular mechanism of RTCB-mediated tRNA ligation.",
            source_name="Science",
            authors=["Smith J", "Johnson K"],
            doi="10.1126/science.abc1234",
            link="https://science.org/doi/10.1126/science.abc1234",
            semantic_score=0.92,
            is_vip=True,
            vip_keywords=["tRNA", "RTCB"],
            published_at=datetime(2024, 12, 14),
            summary="通过冷冻电镜技术揭示了RTCB介导的tRNA连接的分子机制。【关联】",
        ),
        NewsItem(
            title="AI model predicts protein structure with high accuracy",
            content="Deep learning approach achieves near-experimental accuracy in protein structure prediction.",
            source_name="Cell",
            link="https://cell.com/article/12345",
            semantic_score=0.65,
            is_vip=False,
            summary="深度学习方法在蛋白质结构预测中达到接近实验精度的准确率。",
        ),
    ]


class TestToJson:
    """测试 JSON 输出"""

    def test_output_to_file(self):
        """应正确输出到文件"""
        items = create_test_items()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        to_json(items, path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["title"] == items[0].title
        Path(path).unlink()

    def test_empty_list(self):
        """空列表应输出空数组"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        to_json([], path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data == []
        Path(path).unlink()


class TestToMarkdown:
    """测试 Markdown 输出"""

    def test_generates_markdown(self):
        """应生成 Markdown 格式"""
        items = create_test_items()
        md = to_markdown(items)

        assert "# ScholarPipe" in md or "##" in md
        assert items[0].title in md
        assert items[1].title in md

    def test_includes_summary(self):
        """应包含摘要"""
        items = create_test_items()
        md = to_markdown(items)

        assert "CRISPR" in md or "基因编辑" in md

    def test_empty_list(self):
        """空列表应返回提示信息"""
        md = to_markdown([])
        assert md != "" or "empty" in md.lower() or "No" in md


class TestToHtml:
    """测试 HTML 输出"""

    def test_generates_valid_html(self):
        """应生成有效的 HTML"""
        items = create_test_items()
        html = to_html(items)

        assert html.startswith("<!DOCTYPE html>")
        assert "<html" in html
        assert "</html>" in html
        assert "ScholarPipe" in html

    def test_includes_all_items(self):
        """应包含所有条目"""
        items = create_test_items()
        html = to_html(items)

        for item in items:
            # 标题应该存在（可能被 HTML 转义）
            assert item.title[:20] in html or item.source_name in html

    def test_vip_styling(self):
        """VIP 条目应有特殊样式"""
        items = create_test_items()
        html = to_html(items)

        assert "vip" in html.lower()
        assert "tRNA" in html or "RTCB" in html

    def test_semantic_score_display(self):
        """应显示语义相关度分数"""
        items = create_test_items()
        html = to_html(items)

        # 检查是否有 score 相关的 CSS class
        assert "score" in html

    def test_with_stats(self):
        """带统计信息应正确显示"""
        items = create_test_items()
        stats = {"total": 100, "after_dedup": 50, "after_filter": 10}
        html = to_html(items, stats)

        assert "100" in html
        assert "50" in html

    def test_empty_list(self):
        """空列表应显示提示"""
        html = to_html([])

        assert "<!DOCTYPE html>" in html
        assert "No articles" in html or "empty" in html.lower()

    def test_includes_css(self):
        """应包含 CSS 样式"""
        items = create_test_items()
        html = to_html(items)

        assert "<style>" in html
        assert "container" in html
        assert "card" in html


class TestToHtmlFile:
    """测试 HTML 文件输出"""

    def test_creates_file(self):
        """应创建文件"""
        items = create_test_items()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = f.name

        to_html_file(items, path)

        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        Path(path).unlink()

    def test_creates_parent_dirs(self):
        """应创建父目录"""
        items = create_test_items()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "output.html"
            to_html_file(items, str(path))

            assert path.exists()


class TestRenderCard:
    """测试卡片渲染"""

    def test_basic_card(self):
        """基本卡片渲染"""
        item = NewsItem(
            title="Test Article",
            content="Test content here.",
            source_name="Nature",
            link="https://example.com",
        )

        card = _render_card(item, 1)

        assert "article" in card
        assert "card" in card
        assert "Test Article" in card
        assert "Nature" in card

    def test_vip_card(self):
        """VIP 卡片应有特殊样式"""
        item = NewsItem(
            title="tRNA Study",
            content="Content",
            source_name="Science",
            link="https://example.com",
            is_vip=True,
            vip_keywords=["tRNA"],
        )

        card = _render_card(item, 1)

        assert "vip" in card.lower()
        assert "tRNA" in card

    def test_with_doi(self):
        """带 DOI 应显示链接"""
        item = NewsItem(
            title="Test",
            content="Content",
            source_name="Cell",
            link="https://example.com",
            doi="10.1234/test",
        )

        card = _render_card(item, 1)

        assert "doi.org" in card
        assert "10.1234/test" in card

    def test_with_authors(self):
        """带作者应显示"""
        item = NewsItem(
            title="Test",
            content="Content",
            source_name="PNAS",
            link="https://example.com",
            authors=["Smith J", "Johnson K", "Williams M"],
        )

        card = _render_card(item, 1)

        assert "Smith" in card

    def test_many_authors_truncated(self):
        """作者过多应截断"""
        item = NewsItem(
            title="Test",
            content="Content",
            source_name="PNAS",
            link="https://example.com",
            authors=[f"Author{i}" for i in range(10)],
        )

        card = _render_card(item, 1)

        assert "et al" in card

    def test_semantic_score_classes(self):
        """语义分数应有不同样式类"""
        high_score_item = NewsItem(
            title="High", content="", source_name="Test",
            link="https://example.com", semantic_score=0.85
        )
        medium_score_item = NewsItem(
            title="Medium", content="", source_name="Test",
            link="https://example.com", semantic_score=0.55
        )
        low_score_item = NewsItem(
            title="Low", content="", source_name="Test",
            link="https://example.com", semantic_score=0.25
        )

        high_card = _render_card(high_score_item, 1)
        medium_card = _render_card(medium_score_item, 1)
        low_card = _render_card(low_score_item, 1)

        assert "high" in high_card
        assert "medium" in medium_card
        assert "low" in low_card


class TestRenderStats:
    """测试统计渲染"""

    def test_renders_all_stats(self):
        """应渲染所有统计数据"""
        stats = {"total": 100, "after_dedup": 50, "after_filter": 10}
        html = _render_stats(stats)

        assert "100" in html
        assert "50" in html
        assert "10" in html

    def test_handles_missing_keys(self):
        """缺少键时不应报错"""
        stats = {"total": 100}
        html = _render_stats(stats)

        assert "100" in html


class TestGetCss:
    """测试 CSS 生成"""

    def test_returns_css(self):
        """应返回 CSS 字符串"""
        css = _get_css()

        assert ":root" in css
        assert ".card" in css
        assert ".vip" in css

    def test_responsive_styles(self):
        """应包含响应式样式"""
        css = _get_css()

        assert "@media" in css


def test_deliver():
    """集成测试"""
    print("=" * 60)
    print("Step 7: Deliver 测试")
    print("=" * 60)

    items = create_test_items()

    # Test 1: JSON
    print("\n[1] 测试 JSON 输出...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_path = f.name
    to_json(items, json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert len(json_data) == 3
    print(f"    ✓ JSON 输出 {len(json_data)} 条记录")
    Path(json_path).unlink()

    # Test 2: Markdown
    print("\n[2] 测试 Markdown 输出...")
    md = to_markdown(items)
    assert len(md) > 100
    print(f"    ✓ Markdown 输出 {len(md)} 字符")

    # Test 3: HTML
    print("\n[3] 测试 HTML 输出...")
    html = to_html(items)
    assert "<!DOCTYPE html>" in html
    assert "ScholarPipe" in html
    print(f"    ✓ HTML 输出 {len(html)} 字符")

    # Test 4: HTML with stats
    print("\n[4] 测试带统计的 HTML 输出...")
    stats = {"total": 100, "after_dedup": 50, "after_filter": 3}
    html_with_stats = to_html(items, stats)
    assert "100" in html_with_stats
    print("    ✓ 统计信息正确渲染")

    # Test 5: HTML file
    print("\n[5] 测试 HTML 文件输出...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        html_path = f.name
    to_html_file(items, html_path, stats)
    assert Path(html_path).exists()
    file_size = Path(html_path).stat().st_size
    print(f"    ✓ HTML 文件 {file_size} 字节")
    Path(html_path).unlink()

    # Test 6: VIP rendering
    print("\n[6] 测试 VIP 条目渲染...")
    vip_item = [i for i in items if i.is_vip][0]
    card = _render_card(vip_item, 1)
    assert "vip" in card.lower()
    assert vip_item.vip_keywords[0] in card
    print(f"    ✓ VIP 关键词 {vip_item.vip_keywords} 正确显示")

    # Test 7: CSS
    print("\n[7] 测试 CSS 样式...")
    css = _get_css()
    assert ".card" in css
    assert "@media" in css
    print(f"    ✓ CSS 包含响应式样式 ({len(css)} 字符)")

    print("\n" + "=" * 60)
    print("✓ Step 7 单元测试全部通过")
    print("=" * 60)


if __name__ == "__main__":
    # 运行基础测试
    test_deliver()

    # 运行 pytest 风格测试
    print("\n运行详细单元测试...")
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
