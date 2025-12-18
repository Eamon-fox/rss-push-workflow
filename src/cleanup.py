"""Cleanup module - manage intermediate data and log lifecycle."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Intermediate directories to clean (date-based subdirectories)
# Note: data/raw is NOT included because it serves as RSS/PubMed cache (5h TTL)
INTERMEDIATE_DIRS = [
    "data/cleaned",
    "data/deduped",
    "data/filtered",
]

# Protected files that must NOT be deleted
PROTECTED_FILES = [
    "data/embedding_cache.db",
    "data/llm_cache.db",
    "data/seen.json",
    # 用户系统相关
    "data/users.json",
    "data/bookmarks.json",
    "data/article_index.json",
]

# Log retention period
LOG_RETENTION_DAYS = 30

# Raw data retention (RSS/PubMed cache)
RAW_DATA_RETENTION_DAYS = 2


def cleanup_intermediate_data(date: str | None = None) -> dict[str, int]:
    """
    Clean up intermediate processing data.

    Args:
        date: Specific date (YYYY-MM-DD) to clean, None for all dates

    Returns:
        Dict with cleanup statistics
    """
    stats = {"dirs_removed": 0, "errors": 0}

    for dir_path in INTERMEDIATE_DIRS:
        base = Path(dir_path)
        if not base.exists():
            continue

        if date:
            # Remove specific date subdirectory
            target = base / date
            if target.exists() and target.is_dir():
                try:
                    shutil.rmtree(target)
                    stats["dirs_removed"] += 1
                    logger.info(f"Cleaned: {target}")
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Failed to clean {target}: {e}")
        else:
            # Remove all date subdirectories
            for subdir in base.iterdir():
                if subdir.is_dir():
                    try:
                        shutil.rmtree(subdir)
                        stats["dirs_removed"] += 1
                        logger.info(f"Cleaned: {subdir}")
                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Failed to clean {subdir}: {e}")

    return stats


def cleanup_old_raw_data(retention_days: int = RAW_DATA_RETENTION_DAYS) -> dict[str, int]:
    """
    Clean up old raw data (RSS/PubMed cache) beyond retention period.

    Args:
        retention_days: Keep raw data from the last N days

    Returns:
        Dict with cleanup statistics
    """
    stats = {"dirs_removed": 0, "errors": 0}
    raw_dir = Path("data/raw")

    if not raw_dir.exists():
        return stats

    cutoff = datetime.now() - timedelta(days=retention_days)

    for subdir in raw_dir.iterdir():
        if not subdir.is_dir():
            continue
        try:
            # Parse date from directory name (format: YYYY-MM-DD)
            dir_date = datetime.strptime(subdir.name, "%Y-%m-%d")
            if dir_date < cutoff:
                shutil.rmtree(subdir)
                stats["dirs_removed"] += 1
                logger.info(f"Removed old raw data: {subdir}")
        except ValueError:
            # Skip non-date-formatted directories
            continue
        except Exception as e:
            stats["errors"] += 1
            logger.error(f"Failed to remove {subdir}: {e}")

    return stats


def cleanup_old_logs(retention_days: int = LOG_RETENTION_DAYS) -> dict[str, int]:
    """
    Clean up old log files.

    Args:
        retention_days: Keep logs from the last N days

    Returns:
        Dict with cleanup statistics
    """
    stats = {"files_removed": 0, "errors": 0}
    log_dir = Path("logs")

    if not log_dir.exists():
        return stats

    cutoff = datetime.now() - timedelta(days=retention_days)

    for log_file in log_dir.glob("*.log"):
        try:
            # Parse date from filename (format: YYYY-MM-DD.log)
            date_str = log_file.stem
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                log_file.unlink()
                stats["files_removed"] += 1
                logger.info(f"Removed old log: {log_file}")
        except ValueError:
            # Skip non-date-formatted files (like cron.log)
            continue
        except Exception as e:
            stats["errors"] += 1
            logger.error(f"Failed to remove {log_file}: {e}")

    return stats


def verify_protected_files() -> bool:
    """
    Verify that protected files exist.

    Returns:
        True if all protected files exist
    """
    all_exist = True
    for filepath in PROTECTED_FILES:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Protected file missing: {filepath}")
            all_exist = False
    return all_exist


def get_intermediate_data_size() -> dict[str, int]:
    """
    Get size of intermediate data directories.

    Returns:
        Dict mapping directory path to size in bytes
    """
    sizes = {}
    for dir_path in INTERMEDIATE_DIRS:
        base = Path(dir_path)
        if not base.exists():
            sizes[dir_path] = 0
            continue

        total_size = 0
        for f in base.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        sizes[dir_path] = total_size

    return sizes
