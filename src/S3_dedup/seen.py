"""Seen records management with per-user isolation."""

from datetime import datetime, timedelta
from pathlib import Path

from src.core import load_json, save_json, now_iso

SEEN_DIR = Path("data/seen")
DEFAULT_USER = "default"
MAX_AGE_DAYS = 7


def _get_user_file(user_id: str | None = None) -> Path:
    """Get seen file path for a user."""
    user_id = user_id or DEFAULT_USER
    # Sanitize user_id to prevent path traversal
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return SEEN_DIR / f"{safe_id}.json"


def load(user_id: str | None = None) -> dict[str, str]:
    """Load seen records {fingerprint: timestamp} for a user."""
    path = _get_user_file(user_id)
    return load_json(path, default={})


def save(seen: dict[str, str], user_id: str | None = None) -> None:
    """Save seen records for a user."""
    path = _get_user_file(user_id)
    save_json(seen, path)


def is_seen(seen: dict, fingerprint: str) -> bool:
    """Check if fingerprint exists in records."""
    return fingerprint in seen


def mark_seen(seen: dict, fingerprint: str) -> None:
    """Mark fingerprint as seen now."""
    seen[fingerprint] = now_iso()


def mark_batch(seen: dict, fingerprints: list[str]) -> None:
    """Mark multiple fingerprints as seen."""
    now = now_iso()
    for fp in fingerprints:
        seen[fp] = now


def cleanup(seen: dict, max_age_days: int = MAX_AGE_DAYS) -> dict:
    """Remove records older than max_age."""
    cutoff = datetime.now() - timedelta(days=max_age_days)
    return {
        fp: ts for fp, ts in seen.items()
        if datetime.fromisoformat(ts) > cutoff
    }


def migrate_legacy_seen() -> None:
    """Migrate old data/seen.json to new per-user format."""
    legacy_file = Path("data/seen.json")
    if not legacy_file.exists():
        return

    try:
        legacy_data = load_json(legacy_file, default=None)
        if legacy_data is None:
            return

        # Save to default user
        save(legacy_data, user_id=DEFAULT_USER)

        # Rename old file as backup
        backup_file = Path("data/seen.json.bak")
        legacy_file.rename(backup_file)
        print(f"Migrated {len(legacy_data)} seen records to per-user format")
    except Exception as e:
        print(f"Migration failed: {e}")


def list_users() -> list[str]:
    """List all users with seen records."""
    if not SEEN_DIR.exists():
        return []
    return [f.stem for f in SEEN_DIR.glob("*.json")]
