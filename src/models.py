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


class SourceType(str, Enum):
    """信息源类型."""
    JOURNAL = "journal"      # 期刊RSS (Nature, Science, Cell)
    PUBMED = "pubmed"        # PubMed搜索
    PREPRINT = "preprint"    # 预印本 (bioRxiv, medRxiv)
    WECHAT = "wechat"        # 微信公众号
    NEWS = "news"            # 新闻网站 (科学网, Phys.org)
    SOCIAL = "social"        # 社交媒体 (Twitter, Reddit)


class RelatedSource(BaseModel):
    """聚合时的关联来源."""
    source_type: SourceType = SourceType.JOURNAL
    source_name: str = ""
    title: str = ""
    link: str = ""
    content: str = ""        # 该来源的内容/解读
    image_url: str = ""      # 该来源的配图
    published_at: datetime | None = None


class NewsItem(BaseModel):
    """
    一条学术资讯 - 支持多源聚合
    核心是一个"研究"，可以有多个来源的报道
    """

    # === 核心内容 ===
    title: str
    content: str = ""           # 原始内容 (摘要/正文片段)
    link: str = ""              # 原文链接

    # === 元数据 ===
    authors: list[str] = []     # 作者列表
    doi: str = ""               # DOI

    # === 溯源 ===
    source_type: SourceType = SourceType.JOURNAL  # 来源类型
    source_name: str = ""       # "Nature", "PubMed", "BioArt"
    source_url: str = ""        # 来源地址 (RSS URL 或网页)
    journal_name: str = ""      # 期刊名称 (用于PubMed来源显示原始期刊)

    # === 多媒体 ===
    image_url: str = ""         # 主配图 (Graphical Abstract)
    image_urls: list[str] = []  # 多个配图

    # === 多源聚合 ===
    related_sources: list[RelatedSource] = []  # 关联的其他来源
    source_count: int = 1       # 被多少源报道 (热度指标)
    content_cn: str = ""        # 中文解读 (来自公众号等)

    # === AI 处理结果 ===
    score: float | None = None  # 0-10 (LLM评分)
    summary: str = ""           # 200字中文摘要
    semantic_score: float | None = None  # 语义相似度得分 (0-1)
    is_vip: bool = False        # 是否命中VIP关键词
    vip_keywords: list[str] = []  # 命中的VIP关键词列表

    # === Enrichment标记 ===
    is_enriched: bool = False   # 是否已通过API补充完整摘要
    original_content_len: int = 0  # 原始内容长度

    # === 状态 ===
    status: ItemStatus = ItemStatus.NEW

    # === 时间戳 ===
    fetched_at: datetime = Field(default_factory=datetime.now)
    published_at: datetime | None = None

    @computed_field
    @property
    def content_hash(self) -> str:
        """用于去重的内容哈希"""
        text = f"{self.title}{self.content}".lower()
        text = "".join(c for c in text if c.isalnum())
        return hashlib.md5(text.encode()).hexdigest()

    @computed_field
    @property
    def has_chinese_content(self) -> bool:
        """是否有中文解读."""
        return bool(self.content_cn)

    @computed_field
    @property
    def has_image(self) -> bool:
        """是否有配图."""
        return bool(self.image_url or self.image_urls)

    @computed_field
    @property
    def is_multi_source(self) -> bool:
        """是否被多源报道."""
        return self.source_count > 1 or len(self.related_sources) > 0

    class Config:
        use_enum_values = True
