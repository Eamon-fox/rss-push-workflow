"""Step 4: Deliver/output the processed news items."""

import json
from datetime import datetime
from pathlib import Path

from .models import NewsItem


def to_console(items: list[NewsItem], stats: dict | None = None) -> None:
    """
    Print formatted output to console.

    Args:
        items: Items to display
        stats: Optional stats dict with counts
    """
    today = datetime.now().strftime("%Y-%m-%d")

    print()
    print("=" * 66)
    print(f"  ScholarPipe 学术早报 | {today}")
    print("=" * 66)
    print()

    for item in items:
        score_str = f"{item.score:.1f}" if item.score else "N/A"

        print("-" * 66)
        print(f"[{score_str}] {item.title[:55]}...")
        print(f"来源: {item.source_name}")
        print()
        print(item.summary if item.summary else "(无摘要)")
        print()
        print(f"链接: {item.link}")
        print("-" * 66)
        print()

    if stats:
        print("=" * 66)
        print(f" 共抓取 {stats.get('total', 0)} 条 | "
              f"去重后 {stats.get('after_dedup', 0)} 条 | "
              f"推荐 {stats.get('recommended', 0)} 条")
        print("=" * 66)


def to_json(items: list[NewsItem], path: str) -> None:
    """
    Export items to JSON file.

    Args:
        items: Items to export
        path: Output file path
    """
    data = [item.model_dump() for item in items]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def to_markdown(items: list[NewsItem]) -> str:
    """
    Generate markdown formatted output.

    Args:
        items: Items to format

    Returns:
        Markdown string
    """
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# ScholarPipe 学术早报 | {today}\n"]

    for item in items:
        score_str = f"{item.score:.1f}" if item.score else "N/A"
        lines.append(f"## [{score_str}] {item.title}\n")
        lines.append(f"**来源**: {item.source_name}\n")
        lines.append(f"{item.summary}\n")
        lines.append(f"[阅读原文]({item.link})\n")
        lines.append("---\n")

    return "\n".join(lines)
