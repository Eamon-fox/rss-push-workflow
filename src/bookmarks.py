"""收藏功能 CRUD 模块."""

from pathlib import Path
from typing import Optional

from src.core import load_json, save_json, now_iso

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
BOOKMARKS_FILE = DATA_DIR / "bookmarks.json"


def _load_bookmarks() -> dict:
    """加载收藏数据"""
    return load_json(BOOKMARKS_FILE, default={})


def _save_bookmarks(bookmarks: dict):
    """保存收藏数据"""
    save_json(bookmarks, BOOKMARKS_FILE)


# ─────────────────────────────────────────────────────────────
# CRUD 操作
# ─────────────────────────────────────────────────────────────

def get_user_bookmarks(openid: str) -> list[dict]:
    """
    获取用户的收藏列表

    Args:
        openid: 用户的微信 openid

    Returns:
        收藏列表，按保存时间倒序排列，包含文章基本信息
    """
    bookmarks = _load_bookmarks()
    user_bookmarks = bookmarks.get(openid, {})

    # 转换为列表格式
    result = []
    for article_id, data in user_bookmarks.items():
        result.append({
            "article_id": article_id,
            "saved_at": data.get("saved_at", ""),
            "note": data.get("note", ""),
            # 文章基本信息
            "title": data.get("title", ""),
            "journal": data.get("journal", ""),
            "date": data.get("date", ""),
        })

    # 按保存时间倒序排列
    result.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    return result


def add_bookmark(
    openid: str,
    article_id: str,
    note: str = "",
    title: str = "",
    journal: str = "",
    date: str = "",
) -> dict:
    """
    添加收藏

    Args:
        openid: 用户的微信 openid
        article_id: 文章 ID
        note: 可选的备注
        title: 文章标题
        journal: 期刊名称
        date: 文章所属日期 (YYYY-MM-DD)

    Returns:
        收藏记录
    """
    bookmarks = _load_bookmarks()

    if openid not in bookmarks:
        bookmarks[openid] = {}

    bookmark_data = {
        "saved_at": now_iso(),
        "note": note,
        "title": title,
        "journal": journal,
        "date": date,
    }

    bookmarks[openid][article_id] = bookmark_data
    _save_bookmarks(bookmarks)

    return {
        "article_id": article_id,
        **bookmark_data,
    }


def remove_bookmark(openid: str, article_id: str) -> bool:
    """
    取消收藏

    Args:
        openid: 用户的微信 openid
        article_id: 文章 ID

    Returns:
        是否成功删除
    """
    bookmarks = _load_bookmarks()

    if openid not in bookmarks:
        return False

    if article_id not in bookmarks[openid]:
        return False

    del bookmarks[openid][article_id]
    _save_bookmarks(bookmarks)
    return True


def is_bookmarked(openid: str, article_id: str) -> bool:
    """
    检查是否已收藏

    Args:
        openid: 用户的微信 openid
        article_id: 文章 ID

    Returns:
        是否已收藏
    """
    bookmarks = _load_bookmarks()
    return article_id in bookmarks.get(openid, {})


def get_bookmark(openid: str, article_id: str) -> Optional[dict]:
    """
    获取单条收藏记录

    Args:
        openid: 用户的微信 openid
        article_id: 文章 ID

    Returns:
        收藏记录或 None
    """
    bookmarks = _load_bookmarks()
    user_bookmarks = bookmarks.get(openid, {})

    if article_id not in user_bookmarks:
        return None

    return {
        "article_id": article_id,
        **user_bookmarks[article_id],
    }


def get_bookmark_count(openid: str) -> int:
    """
    获取用户收藏数量

    Args:
        openid: 用户的微信 openid

    Returns:
        收藏数量
    """
    bookmarks = _load_bookmarks()
    return len(bookmarks.get(openid, {}))


def update_bookmark_note(openid: str, article_id: str, note: str) -> Optional[dict]:
    """
    更新收藏备注

    Args:
        openid: 用户的微信 openid
        article_id: 文章 ID
        note: 新的备注

    Returns:
        更新后的收藏记录或 None
    """
    bookmarks = _load_bookmarks()

    if openid not in bookmarks or article_id not in bookmarks[openid]:
        return None

    bookmarks[openid][article_id]["note"] = note
    bookmarks[openid][article_id]["updated_at"] = now_iso()
    _save_bookmarks(bookmarks)

    return {
        "article_id": article_id,
        **bookmarks[openid][article_id],
    }
