"""文章索引模块 - 维护 article_id → 存储位置 的映射."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
INDEX_FILE = DATA_DIR / "article_index.json"
ARCHIVE_DIR = Path("output/archive")


def _load_index() -> dict:
    """加载索引"""
    if not INDEX_FILE.exists():
        return {}
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load article index, starting fresh: {e}")
        return {}


def _save_index(index: dict):
    """保存索引"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # 先写入临时文件，再原子替换，避免写入中断导致文件损坏
        temp_file = INDEX_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        temp_file.replace(INDEX_FILE)
    except Exception as e:
        logger.error(f"Failed to save article index: {e}")


# ─────────────────────────────────────────────────────────────
# 索引操作
# ─────────────────────────────────────────────────────────────

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


def add_article_to_index(article_id: str, date: str, version: int):
    """
    添加文章到索引

    Args:
        article_id: 文章 ID
        date: 日期 YYYY-MM-DD
        version: 版本号
    """
    index = _load_index()
    index[article_id] = {
        "date": date,
        "version": version,
    }
    _save_index(index)


def add_articles_batch(articles: list[dict], date: str, version: int):
    """
    批量添加文章到索引

    Args:
        articles: 文章列表，每个需要有 id 字段
        date: 日期
        version: 版本号
    """
    index = _load_index()
    for article in articles:
        article_id = article.get("id")
        if article_id:
            index[article_id] = {
                "date": date,
                "version": version,
            }
    _save_index(index)


def rebuild_index_from_archive() -> int:
    """
    从归档目录重建完整索引

    Returns:
        索引的文章数量
    """
    import hashlib

    def _generate_id(item: dict) -> str:
        """生成文章 ID"""
        if item.get("doi"):
            return f"doi_{item['doi'].replace('/', '_').replace('.', '_')}"
        title = (item.get("title") or "").strip().lower()
        return f"t_{hashlib.md5(title.encode()).hexdigest()[:12]}"

    index = {}

    if not ARCHIVE_DIR.exists():
        return 0

    # 遍历所有归档
    for year_dir in sorted(ARCHIVE_DIR.iterdir()):
        if not year_dir.is_dir() or year_dir.name == "index.json":
            continue

        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue

            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir():
                    continue

                date = f"{year_dir.name}-{month_dir.name}-{day_dir.name}"

                # 找到该日期的所有版本
                for json_file in sorted(day_dir.glob("daily_v*.json")):
                    # 提取版本号
                    version_str = json_file.stem.replace("daily_v", "")
                    try:
                        version = int(version_str)
                    except ValueError:
                        continue

                    # 读取文章列表
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            articles = json.load(f)

                        for article in articles:
                            article_id = _generate_id(article)
                            # 保留最新版本
                            if article_id not in index or index[article_id]["version"] < version:
                                index[article_id] = {
                                    "date": date,
                                    "version": version,
                                }
                    except Exception as e:
                        logger.error(f"Error reading {json_file}: {e}")

    _save_index(index)
    return len(index)


def get_index_stats() -> dict:
    """获取索引统计信息"""
    index = _load_index()
    return {
        "total_articles": len(index),
        "index_file": str(INDEX_FILE),
    }
