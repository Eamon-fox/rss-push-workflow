"""Archive module - manage permanent daily report storage."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.core import generate_article_id, load_json, save_json, today, now_iso

logger = logging.getLogger(__name__)

ARCHIVE_BASE = Path("output/archive")
OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
CANDIDATES_FILE = DATA_DIR / "candidates.json"


def get_archive_path(date: datetime | str) -> Path:
    """Get archive path: output/archive/YYYY/MM/DD/"""
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    return ARCHIVE_BASE / f"{date.year}" / f"{date.month:02d}" / f"{date.day:02d}"


def get_next_version(archive_path: Path) -> int:
    """Get next version number for the day."""
    if not archive_path.exists():
        return 1

    existing = list(archive_path.glob("daily_v*.json"))
    if not existing:
        return 1

    versions = []
    for f in existing:
        try:
            v = int(f.stem.split("_v")[1])
            versions.append(v)
        except (IndexError, ValueError):
            continue

    return max(versions) + 1 if versions else 1


def archive_daily(
    items: list,
    stats: dict[str, Any],
    date: Optional[str] = None,
) -> dict[str, Any]:
    """
    Archive daily report to permanent storage.

    Args:
        items: List of news items (NewsItem objects or dicts)
        stats: Pipeline statistics
        date: Date string YYYY-MM-DD, defaults to today

    Returns:
        Archive metadata for this version
    """
    if date is None:
        date = today()

    archive_path = get_archive_path(date)
    archive_path.mkdir(parents=True, exist_ok=True)

    version = get_next_version(archive_path)

    # Define file paths
    json_file = archive_path / f"daily_v{version}.json"
    html_file = archive_path / f"daily_v{version}.html"
    md_file = archive_path / f"daily_v{version}.md"

    # Copy current output to archive
    src_json = OUTPUT_DIR / "daily.json"
    src_html = OUTPUT_DIR / "daily.html"
    src_md = OUTPUT_DIR / "daily.md"

    if src_json.exists():
        shutil.copy2(src_json, json_file)
    if src_html.exists():
        shutil.copy2(src_html, html_file)
    if src_md.exists():
        shutil.copy2(src_md, md_file)

    # Update metadata
    metadata_file = archive_path / "metadata.json"
    metadata = _load_metadata(metadata_file)

    version_info = {
        "version": version,
        "created_at": now_iso(),
        "stats": stats,
        "article_count": len(items),
        "files": {
            "json": json_file.name,
            "html": html_file.name,
            "md": md_file.name,
        },
    }

    metadata["date"] = date
    metadata["versions"].append(version_info)
    metadata["latest_version"] = version

    _save_metadata(metadata_file, metadata)

    # Update global index
    _update_index(date, len(items), version)

    # Update article index
    _update_article_index(items, date, version)

    return version_info


def _update_article_index(items: list, date: str, version: int):
    """Update article ID index for quick lookup."""
    try:
        from src.article_index import _load_index, _save_index

        index = _load_index()

        for item in items:
            # 支持 NewsItem 对象和 dict
            if hasattr(item, "to_dict"):
                item_dict = item.to_dict()
            elif hasattr(item, "__dict__"):
                item_dict = vars(item)
            else:
                item_dict = item

            article_id = generate_article_id(item_dict)
            index[article_id] = {
                "date": date,
                "version": version,
            }

        _save_index(index)
    except Exception as e:
        # 索引更新失败不应阻止归档
        logger.warning(f"Failed to update article index: {e}")


def _load_metadata(path: Path) -> dict[str, Any]:
    """Load or create metadata."""
    return load_json(path, default={"versions": []})


def _save_metadata(path: Path, data: dict[str, Any]):
    """Save metadata."""
    save_json(data, path)


def _update_index(date: str, article_count: int, version: int):
    """Update global archive index."""
    index_file = ARCHIVE_BASE / "index.json"
    index = load_json(index_file, default={"dates": {}})

    index["dates"][date] = {
        "versions": version,
        "article_count": article_count,
        "updated_at": now_iso(),
    }
    index["last_updated"] = now_iso()

    save_json(index, index_file)


def load_archived_daily(
    date: str,
    version: Optional[int] = None,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Load archived daily report.

    Args:
        date: Date string YYYY-MM-DD
        version: Version number, None for latest

    Returns:
        (articles, metadata)
    """
    archive_path = get_archive_path(date)

    if not archive_path.exists():
        return [], {}

    metadata_file = archive_path / "metadata.json"
    metadata = _load_metadata(metadata_file)

    if version is None:
        version = metadata.get("latest_version", 1)

    json_file = archive_path / f"daily_v{version}.json"

    if not json_file.exists():
        return [], metadata

    articles = load_json(json_file, default=[])
    return articles, metadata


def list_archive_dates(
    year: Optional[int] = None,
    month: Optional[int] = None,
    limit: int = 30,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    List archived versions (flat structure).

    Args:
        year: Filter by year
        month: Filter by month
        limit: Max results
        offset: Pagination offset

    Returns:
        Flat list of version entries, sorted by date desc, version desc:
        [
          {"date": "2025-12-18", "version": 2, "time": "18:00", "article_count": 20},
          {"date": "2025-12-18", "version": 1, "time": "08:00", "article_count": 15},
          ...
        ]
    """
    index_file = ARCHIVE_BASE / "index.json"
    index = load_json(index_file, default={"dates": {}})

    if not index.get("dates"):
        return []

    entries = []
    for date_str in sorted(index.get("dates", {}).keys(), reverse=True):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        if year and dt.year != year:
            continue
        if month and dt.month != month:
            continue

        # Load metadata for this date to get version details
        archive_path = get_archive_path(date_str)
        metadata_file = archive_path / "metadata.json"

        if metadata_file.exists():
            metadata = _load_metadata(metadata_file)
            versions = metadata.get("versions", [])
            # Sort versions desc
            for v in sorted(versions, key=lambda x: x.get("version", 0), reverse=True):
                created_at = v.get("created_at", "")
                # Extract HH:MM from ISO timestamp
                time_str = ""
                if created_at:
                    try:
                        time_str = created_at[11:16]  # "2025-12-18T08:30:00" -> "08:30"
                    except (IndexError, TypeError):
                        pass
                entries.append({
                    "date": date_str,
                    "version": v.get("version", 1),
                    "time": time_str,
                    "article_count": v.get("article_count", 0),
                })
        else:
            # Fallback: no metadata, use index info
            info = index["dates"].get(date_str, {})
            entries.append({
                "date": date_str,
                "version": info.get("versions", 1),
                "time": "",
                "article_count": info.get("article_count", 0),
            })

    return entries[offset : offset + limit]


def get_archive_stats() -> dict[str, Any]:
    """Get archive statistics."""
    index_file = ARCHIVE_BASE / "index.json"
    index = load_json(index_file, default={"dates": {}})
    dates_info = index.get("dates", {})
    if not dates_info:
        return {
            "total_days": 0,
            "total_articles": 0,
            "date_range": None,
            "by_month": {},
        }

    sorted_dates = sorted(dates_info.keys())
    total_articles = sum(info.get("article_count", 0) for info in dates_info.values())

    # Group by month
    by_month: dict[str, dict] = {}
    for date_str, info in dates_info.items():
        month_key = date_str[:7]  # YYYY-MM
        if month_key not in by_month:
            by_month[month_key] = {"days": 0, "articles": 0}
        by_month[month_key]["days"] += 1
        by_month[month_key]["articles"] += info.get("article_count", 0)

    return {
        "total_days": len(dates_info),
        "total_articles": total_articles,
        "date_range": {
            "start": sorted_dates[0],
            "end": sorted_dates[-1],
        },
        "by_month": by_month,
    }


# ============================================================
# Candidate Pool Storage (for personalization)
# ============================================================

def archive_candidate_pool(items: list) -> Path:
    """
    追加文章到全局候选池。

    Args:
        items: List of NewsItem objects that passed Layer 1 filter

    Returns:
        Path to candidates file
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载现有候选池
    existing_candidates: list[dict] = load_json(CANDIDATES_FILE, default=[])
    existing_ids: set[str] = {generate_article_id(c) for c in existing_candidates}

    # 2. 追加新文章（去重）
    new_count = 0
    for item in items:
        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        elif hasattr(item, "to_dict"):
            item_dict = item.to_dict()
        elif hasattr(item, "__dict__"):
            item_dict = vars(item)
        else:
            item_dict = item

        candidate = {
            "doi": item_dict.get("doi"),
            "title": item_dict.get("title"),
            "content": item_dict.get("content"),
            "source_name": item_dict.get("source_name"),
            "journal_name": item_dict.get("journal_name"),
            "link": item_dict.get("link"),
            "published_at": item_dict.get("published_at"),
            "authors": item_dict.get("authors"),
            "default_score": item_dict.get("semantic_score"),
            "is_vip": item_dict.get("is_vip", False),
            "vip_keywords": item_dict.get("vip_keywords", []),
        }

        article_id = generate_article_id(candidate)
        if article_id not in existing_ids:
            existing_candidates.append(candidate)
            existing_ids.add(article_id)
            new_count += 1

    # 3. 保存
    save_json(existing_candidates, CANDIDATES_FILE)

    logger.info(f"Candidate pool: +{new_count}, total {len(existing_candidates)}")
    return CANDIDATES_FILE


def load_candidate_pool() -> list[dict]:
    """加载全局候选池。"""
    return load_json(CANDIDATES_FILE, default=[])


def load_historical_candidates() -> list[dict]:
    """
    加载候选池（带 article_id）。

    Returns:
        候选文章列表
    """
    candidates = load_candidate_pool()

    # 添加 article_id
    for candidate in candidates:
        candidate["id"] = generate_article_id(candidate)

    logger.info(f"Loaded {len(candidates)} candidates")
    return candidates


# ============================================================
# Personal Daily Storage (per-user personalized reports)
# ============================================================

def _get_personal_dir(date: str) -> Path:
    """Get personal daily directory for a date."""
    archive_path = get_archive_path(date)
    return archive_path / "personal"


def _sanitize_openid(openid: str) -> str:
    """Sanitize openid for safe filename."""
    return "".join(c for c in openid if c.isalnum() or c in "-_")


def save_personal_daily(
    openid: str,
    articles: list[dict],
    date: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> Path:
    """
    Save personal daily report for a user.

    Args:
        openid: User's openid
        articles: List of article dicts (already scored and sorted)
        date: Date string YYYY-MM-DD, defaults to today
        config_hash: Hash of user config (for cache invalidation)

    Returns:
        Path to saved file
    """
    if date is None:
        date = today()

    personal_dir = _get_personal_dir(date)
    personal_dir.mkdir(parents=True, exist_ok=True)

    safe_openid = _sanitize_openid(openid)
    personal_file = personal_dir / f"{safe_openid}.json"

    data = {
        "openid": openid,
        "date": date,
        "generated_at": now_iso(),
        "config_hash": config_hash,
        "article_count": len(articles),
        "articles": articles,
    }

    save_json(data, personal_file)

    # 更新文章索引
    _update_article_index(articles, date, version=1)

    logger.info(f"Saved personal daily for {safe_openid}: {len(articles)} articles -> {personal_file}")
    return personal_file


def load_personal_daily(
    openid: str,
    date: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """
    Load personal daily report for a user.

    Args:
        openid: User's openid
        date: Date string YYYY-MM-DD, defaults to today

    Returns:
        (articles, metadata) or ([], {}) if not found
    """
    if date is None:
        date = today()

    personal_dir = _get_personal_dir(date)
    safe_openid = _sanitize_openid(openid)
    personal_file = personal_dir / f"{safe_openid}.json"

    data = load_json(personal_file, default=None)
    if data is None:
        return [], {}

    articles = data.get("articles", [])
    metadata = {
        "openid": data.get("openid"),
        "date": data.get("date"),
        "generated_at": data.get("generated_at"),
        "config_hash": data.get("config_hash"),
        "article_count": data.get("article_count"),
    }
    return articles, metadata


def has_personal_daily(openid: str, date: Optional[str] = None) -> bool:
    """Check if user has a personal daily for the given date."""
    if date is None:
        date = today()

    personal_dir = _get_personal_dir(date)
    safe_openid = _sanitize_openid(openid)
    personal_file = personal_dir / f"{safe_openid}.json"

    return personal_file.exists()


def delete_personal_daily(openid: str, date: Optional[str] = None) -> bool:
    """
    Delete personal daily report (for regeneration).

    Returns:
        True if deleted, False if not found
    """
    if date is None:
        date = today()

    personal_dir = _get_personal_dir(date)
    safe_openid = _sanitize_openid(openid)
    personal_file = personal_dir / f"{safe_openid}.json"

    if personal_file.exists():
        personal_file.unlink()
        logger.info(f"Deleted personal daily for {safe_openid} on {date}")
        return True
    return False
