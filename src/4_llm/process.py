"""LLM processing - dedup, score, summarize."""

from ..models import NewsItem
from ..infra import chat


def process_batch(
    items: list[NewsItem],
    score_threshold: float = 6.0
) -> list[NewsItem]:
    """
    Process items with LLM:
    - Semantic deduplication
    - Scoring (0-10)
    - Summarization (~200 chars Chinese)

    Args:
        items: Items to process
        score_threshold: Minimum score to include

    Returns:
        Processed items above threshold
    """
    if not items:
        return []

    prompt = _build_prompt(items, score_threshold)
    response = chat(prompt)
    return _parse_response(response, items)


def _build_prompt(items: list[NewsItem], score_threshold: float) -> str:
    """Build the processing prompt."""
    items_text = ""
    for i, item in enumerate(items, 1):
        items_text += f"""
[{i}]
标题: {item.title}
来源: {item.source_name}
内容: {item.content[:500]}
链接: {item.link}
"""

    return f"""你是一个学术资讯筛选助手。以下是今天抓取的学术资讯列表。

请完成以下任务：

1. **语义去重**: 识别报道同一研究的条目，合并为一条

2. **评分**: 为每条独立研究评分 (0-10)

3. **摘要**: 为评分 >= {score_threshold} 的条目生成约200字的中文摘要

## 输入条目

{items_text}

## 输出格式 (JSON)

```json
[
  {{
    "indices": [1, 3],
    "title": "标题",
    "link": "链接",
    "source": "来源",
    "score": 8.5,
    "summary": "200字中文摘要"
  }}
]
```

只输出评分 >= {score_threshold} 的条目。只输出 JSON，不要其他内容。
"""


def _parse_response(response: str, original_items: list[NewsItem]) -> list[NewsItem]:
    """Parse LLM response into NewsItem list."""
    import json
    import re

    # Extract JSON from response
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    results = []
    for item in data:
        idx = item.get("indices", [1])[0] - 1
        if 0 <= idx < len(original_items):
            orig = original_items[idx]
            results.append(NewsItem(
                title=item.get("title", orig.title),
                content=orig.content,
                link=item.get("link", orig.link),
                source_name=item.get("source", orig.source_name),
                source_url=orig.source_url,
                score=item.get("score"),
                summary=item.get("summary", ""),
            ))

    return results
