"""Seen records management."""

import json
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_FILE = "data/seen.json"
MAX_AGE_DAYS = 7


def load(filepath: str = DEFAULT_FILE) -> dict[str, str]:
    """Load seen records {fingerprint: timestamp}."""
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


def is_seen(seen: dict, fingerprint: str) -> bool:
    """Check if fingerprint exists in records."""
    return fingerprint in seen


def mark_seen(seen: dict, fingerprint: str) -> None:
    """Mark fingerprint as seen now."""
    seen[fingerprint] = datetime.now().isoformat()


def mark_batch(seen: dict, fingerprints: list[str]) -> None:
    """Mark multiple fingerprints as seen."""
    now = datetime.now().isoformat()
    for fp in fingerprints:
        seen[fp] = now


def cleanup(seen: dict, max_age_days: int = MAX_AGE_DAYS) -> dict:
    """Remove records older than max_age."""
    cutoff = datetime.now() - timedelta(days=max_age_days)
    return {
        fp: ts for fp, ts in seen.items()
        if datetime.fromisoformat(ts) > cutoff
    }
