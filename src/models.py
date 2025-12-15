"""Data models for ScholarPipe."""

from datetime import datetime
from enum import Enum
import hashlib

from pydantic import BaseModel, Field, computed_field


class ItemStatus(str, Enum):
    """Processing status."""
    NEW = "new"
    SCORED = "scored"
    SUMMARIZED = "summarized"
    DELIVERED = "delivered"
    SKIPPED = "skipped"


class NewsItem(BaseModel):
    """
    一条学术资讯 - 轻量级，只保留必要信息
    把论文当新闻对待
    """

    # 核心内容
    title: str
    content: str = ""           # 原始内容 (摘要/正文片段)
    link: str = ""              # 原文链接

    # 元数据
    authors: list[str] = []     # 作者列表
    doi: str = ""               # DOI

    # 溯源
    source_name: str = ""       # "Nature RSS", "PubMed", "科学网"
    source_url: str = ""        # 来源地址 (RSS URL 或网页)

    # AI 处理结果
    score: float | None = None  # 0-10
    summary: str = ""           # 200字中文摘要

    # 状态
    status: ItemStatus = ItemStatus.NEW

    # 时间戳
    fetched_at: datetime = Field(default_factory=datetime.now)
    published_at: datetime | None = None

    @computed_field
    @property
    def content_hash(self) -> str:
        """用于去重的内容哈希"""
        text = f"{self.title}{self.content}".lower()
        text = "".join(c for c in text if c.isalnum())
        return hashlib.md5(text.encode()).hexdigest()

    class Config:
        use_enum_values = True
