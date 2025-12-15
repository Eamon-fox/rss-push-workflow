"""Cross-period deduplication using hash records."""

import json
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_FILE = "data/seen.json"


def load(filepath: str = DEFAULT_FILE) -> dict[str, str]:
    """Load seen records {hash: timestamp}."""
    path = Path(filepath)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(seen: dict[str, str], filepath: str = DEFAULT_FILE) -> None:
    """Save seen records."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seen, f, indent=2)


def is_seen(seen: dict, content_hash: str, window_hours: int = 72) -> bool:
    """Check if seen within time window."""
    if content_hash not in seen:
        return False
    seen_time = datetime.fromisoformat(seen[content_hash])
    cutoff = datetime.now() - timedelta(hours=window_hours)
    return seen_time > cutoff


def mark_seen(seen: dict, content_hash: str) -> None:
    """Mark as seen now."""
    seen[content_hash] = datetime.now().isoformat()


def cleanup(seen: dict, max_age_hours: int = 168) -> dict:
    """Remove records older than max_age (default 7 days)."""
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    return {
        h: t for h, t in seen.items()
        if datetime.fromisoformat(t) > cutoff
    }
