"""文章索引模块 - 维护 article_id → 存储位置 的映射."""

from pathlib import Path
from typing import Optional

from src.core import load_json, save_json

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
INDEX_FILE = DATA_DIR / "article_index.json"


def _load_index() -> dict:
    """加载索引"""
    return load_json(INDEX_FILE, default={})


def _save_index(index: dict):
    """保存索引"""
    save_json(index, INDEX_FILE)


def get_article_location(article_id: str) -> Optional[dict]:
    """
    查询文章存储位置

    Args:
        article_id: 文章 ID

    Returns:
        {"date": "2025-12-18", "version": 1} 或 None
    """
    index = _load_index()
    return index.get(article_id)
