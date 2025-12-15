"""AI-powered paper screening and scoring."""

from .models import Paper


class Screener:
    """Evaluate papers using LLM."""

    def __init__(self, api_key: str | None = None, model: str = "deepseek"):
        self.api_key = api_key
        self.model = model

    def evaluate(self, title: str, abstract: str) -> tuple[float, str]:
        """
        Evaluate paper relevance and quality.

        Returns:
            tuple: (score 0-10, reason)
        """
        # TODO: Implement LLM API call
        pass

    def batch_evaluate(self, papers: list[Paper]) -> list[tuple[float, str]]:
        """Evaluate multiple papers."""
        # TODO: Implement batch processing
        pass

    def _build_prompt(self, title: str, abstract: str) -> str:
        """Build evaluation prompt."""
        return f"""请评估这篇学术论文的价值和相关性。

标题: {title}

摘要: {abstract}

请从以下维度打分 (0-10):
1. 创新性
2. 方法严谨性
3. 与神经科学/RNA生物学的相关性

输出格式:
分数: X
理由: 简短说明
"""
