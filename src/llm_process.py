"""Step 3: LLM processing - dedup, score, and summarize in one pass."""

from .models import NewsItem


def process_batch(
    items: list[NewsItem],
    api_key: str | None = None,
    score_threshold: float = 6.0
) -> list[NewsItem]:
    """
    Process all items with LLM in one pass:
    - Semantic deduplication (merge same research from different sources)
    - Scoring (0-10)
    - Summarization (~200 chars Chinese)

    Args:
        items: Raw items to process
        api_key: LLM API key
        score_threshold: Minimum score to include

    Returns:
        Deduplicated, scored, summarized items above threshold
    """
    if not items:
        return []

    # Build prompt
    prompt = _build_prompt(items, score_threshold)

    # Call LLM
    # TODO: Implement actual LLM call
    # response = call_llm(prompt, api_key)
    # results = parse_response(response)

    # For now, return placeholder
    pass


def _build_prompt(items: list[NewsItem], score_threshold: float) -> str:
    """Build the processing prompt."""

    items_text = ""
    for i, item in enumerate(items, 1):
        items_text += f"""
[{i}]
标题: {item.title}
来源: {item.source_name}
内容: {item.content[:500]}...
链接: {item.link}
"""

    return f"""你是一个学术资讯筛选助手。以下是今天抓取的学术资讯列表。

请完成以下任务：

1. **语义去重**: 识别报道同一研究/论文的条目，合并为一条（保留信息最完整的来源）

2. **评分**: 为每条独立研究评分 (0-10)，评估标准：
   - 创新性和重要性
   - 与生命科学/神经科学的相关性
   - 方法学贡献

3. **摘要**: 为评分 >= {score_threshold} 的条目生成约200字的中文摘要，包含：
   - 研究问题
   - 核心发现
   - 方法亮点（如有）
   - 为什么值得关注

## 输入条目

{items_text}

## 输出格式 (JSON)

```json
[
  {{
    "original_indices": [1, 3],  // 合并了哪些条目
    "title": "保留的标题",
    "link": "保留的链接",
    "source": "来源",
    "score": 8.5,
    "summary": "200字中文摘要..."
  }},
  ...
]
```

只输出评分 >= {score_threshold} 的条目。
"""


def _call_llm(prompt: str, api_key: str | None = None) -> str:
    """Call LLM API."""
    # TODO: Implement with DeepSeek/OpenAI compatible API
    # import httpx
    # response = httpx.post(
    #     "https://api.deepseek.com/v1/chat/completions",
    #     headers={"Authorization": f"Bearer {api_key}"},
    #     json={
    #         "model": "deepseek-chat",
    #         "messages": [{"role": "user", "content": prompt}]
    #     }
    # )
    # return response.json()["choices"][0]["message"]["content"]
    pass


def _parse_response(response: str, original_items: list[NewsItem]) -> list[NewsItem]:
    """Parse LLM response into NewsItem list."""
    # TODO: Parse JSON response and create NewsItem objects
    # import json
    # data = json.loads(response)
    # results = []
    # for item in data:
    #     results.append(NewsItem(
    #         title=item["title"],
    #         link=item["link"],
    #         source_name=item["source"],
    #         score=item["score"],
    #         summary=item["summary"],
    #         content=original_items[item["original_indices"][0] - 1].content
    #     ))
    # return results
    pass
