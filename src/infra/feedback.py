"""Feedback collection module for user relevance feedback.

Records user feedback (relevant/irrelevant) and click events to improve scoring.
Feedback data is stored in data/feedback.json for later analysis and score adjustment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

FEEDBACK_PATH = Path("data/feedback.json")

FeedbackType = Literal["relevant", "irrelevant", "clicked"]


@dataclass
class FeedbackEntry:
    """Single feedback entry for an article."""
    fingerprint: str  # Article fingerprint (DOI or title hash)
    feedback_type: FeedbackType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Article metadata at time of feedback
    title: str = ""
    source_name: str = ""
    semantic_score: float = 0.0
    vip_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackEntry":
        return cls(
            fingerprint=data.get("fingerprint", ""),
            feedback_type=data.get("feedback_type", "clicked"),
            timestamp=data.get("timestamp", ""),
            title=data.get("title", ""),
            source_name=data.get("source_name", ""),
            semantic_score=data.get("semantic_score", 0.0),
            vip_keywords=data.get("vip_keywords", []),
        )


@dataclass
class FeedbackStore:
    """In-memory feedback store with persistence."""
    entries: List[FeedbackEntry] = field(default_factory=list)

    # Index by fingerprint for quick lookup
    _by_fingerprint: Dict[str, List[FeedbackEntry]] = field(default_factory=dict, repr=False)

    def add(self, entry: FeedbackEntry) -> None:
        """Add a feedback entry."""
        self.entries.append(entry)
        if entry.fingerprint not in self._by_fingerprint:
            self._by_fingerprint[entry.fingerprint] = []
        self._by_fingerprint[entry.fingerprint].append(entry)

    def get_feedback(self, fingerprint: str) -> List[FeedbackEntry]:
        """Get all feedback for a given fingerprint."""
        return self._by_fingerprint.get(fingerprint, [])

    def get_latest_feedback(self, fingerprint: str) -> Optional[FeedbackEntry]:
        """Get the most recent feedback for a fingerprint."""
        entries = self.get_feedback(fingerprint)
        if not entries:
            return None
        return max(entries, key=lambda e: e.timestamp)

    def count_by_type(self) -> Dict[str, int]:
        """Count feedback entries by type."""
        counts: Dict[str, int] = {"relevant": 0, "irrelevant": 0, "clicked": 0}
        for entry in self.entries:
            counts[entry.feedback_type] = counts.get(entry.feedback_type, 0) + 1
        return counts

    def save(self, path: Path = FEEDBACK_PATH) -> None:
        """Save feedback to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self.entries]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.entries)} feedback entries to {path}")

    @classmethod
    def load(cls, path: Path = FEEDBACK_PATH) -> "FeedbackStore":
        """Load feedback from JSON file."""
        store = cls()
        if not path.exists():
            return store

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                entry = FeedbackEntry.from_dict(item)
                store.add(entry)

            logger.info(f"Loaded {len(store.entries)} feedback entries from {path}")
        except Exception as e:
            logger.error(f"Error loading feedback from {path}: {e}")

        return store


# Module-level singleton (lazy loaded)
_feedback_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Get the global feedback store (lazy loaded)."""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore.load()
    return _feedback_store


def record_feedback(
    fingerprint: str,
    feedback_type: FeedbackType,
    *,
    title: str = "",
    source_name: str = "",
    semantic_score: float = 0.0,
    vip_keywords: Optional[List[str]] = None,
    auto_save: bool = True,
) -> FeedbackEntry:
    """Record a feedback entry.

    Args:
        fingerprint: Article fingerprint (DOI or title hash)
        feedback_type: "relevant", "irrelevant", or "clicked"
        title: Article title
        source_name: Source name
        semantic_score: Semantic score at time of feedback
        vip_keywords: VIP keywords matched
        auto_save: Whether to auto-save after recording

    Returns:
        The created FeedbackEntry
    """
    entry = FeedbackEntry(
        fingerprint=fingerprint,
        feedback_type=feedback_type,
        title=title,
        source_name=source_name,
        semantic_score=semantic_score,
        vip_keywords=vip_keywords or [],
    )

    store = get_feedback_store()
    store.add(entry)

    if auto_save:
        store.save()

    logger.debug(f"Recorded {feedback_type} feedback for {fingerprint[:20]}...")
    return entry


def record_relevant(
    fingerprint: str,
    **kwargs,
) -> FeedbackEntry:
    """Shortcut to record 'relevant' feedback."""
    return record_feedback(fingerprint, "relevant", **kwargs)


def record_irrelevant(
    fingerprint: str,
    **kwargs,
) -> FeedbackEntry:
    """Shortcut to record 'irrelevant' feedback."""
    return record_feedback(fingerprint, "irrelevant", **kwargs)


def record_clicked(
    fingerprint: str,
    **kwargs,
) -> FeedbackEntry:
    """Shortcut to record 'clicked' feedback."""
    return record_feedback(fingerprint, "clicked", **kwargs)


def get_feedback_stats() -> Dict[str, any]:
    """Get feedback statistics."""
    store = get_feedback_store()
    counts = store.count_by_type()

    # Calculate average scores by feedback type
    scores_by_type: Dict[str, List[float]] = {"relevant": [], "irrelevant": [], "clicked": []}
    for entry in store.entries:
        if entry.semantic_score > 0:
            scores_by_type[entry.feedback_type].append(entry.semantic_score)

    avg_scores = {}
    for fb_type, scores in scores_by_type.items():
        if scores:
            avg_scores[fb_type] = round(sum(scores) / len(scores), 4)
        else:
            avg_scores[fb_type] = None

    return {
        "total_entries": len(store.entries),
        "counts": counts,
        "avg_scores": avg_scores,
        "unique_articles": len(store._by_fingerprint),
    }


def export_for_analysis(path: Optional[Path] = None) -> Path:
    """Export feedback data for analysis.

    Creates a CSV file with feedback data for easy analysis.
    """
    import csv

    if path is None:
        path = Path("data/feedback_export.csv")

    store = get_feedback_store()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "fingerprint",
            "feedback_type",
            "semantic_score",
            "title",
            "source_name",
            "vip_keywords",
        ])

        for entry in store.entries:
            writer.writerow([
                entry.timestamp,
                entry.fingerprint,
                entry.feedback_type,
                entry.semantic_score,
                entry.title,
                entry.source_name,
                "|".join(entry.vip_keywords),
            ])

    logger.info(f"Exported {len(store.entries)} feedback entries to {path}")
    return path


def import_from_browser_export(path: Path, auto_save: bool = True) -> int:
    """Import feedback from browser-exported JSON file.

    The browser export format matches the FeedbackEntry format.
    Duplicates (same fingerprint + feedback_type + timestamp) are skipped.

    Args:
        path: Path to the JSON file exported from browser
        auto_save: Whether to save after importing

    Returns:
        Number of entries imported
    """
    if not path.exists():
        logger.error(f"Import file not found: {path}")
        return 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading import file: {e}")
        return 0

    if not isinstance(data, list):
        logger.error("Import file must contain a JSON array")
        return 0

    store = get_feedback_store()

    # Build set of existing entries for deduplication
    existing = set()
    for entry in store.entries:
        key = (entry.fingerprint, entry.feedback_type, entry.timestamp)
        existing.add(key)

    imported = 0
    for item in data:
        entry = FeedbackEntry.from_dict(item)
        key = (entry.fingerprint, entry.feedback_type, entry.timestamp)
        if key not in existing:
            store.add(entry)
            existing.add(key)
            imported += 1

    if auto_save and imported > 0:
        store.save()

    logger.info(f"Imported {imported} new feedback entries from {path}")
    return imported


def merge_feedback_files(*paths: Path, output: Optional[Path] = None) -> int:
    """Merge multiple feedback files into one.

    Args:
        paths: Paths to feedback JSON files
        output: Output path (defaults to FEEDBACK_PATH)

    Returns:
        Total number of unique entries
    """
    if output is None:
        output = FEEDBACK_PATH

    all_entries: Dict[str, FeedbackEntry] = {}

    for path in paths:
        if not path.exists():
            logger.warning(f"File not found, skipping: {path}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                entry = FeedbackEntry.from_dict(item)
                # Use unique key to deduplicate
                key = f"{entry.fingerprint}:{entry.feedback_type}:{entry.timestamp}"
                if key not in all_entries:
                    all_entries[key] = entry
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")

    # Save merged entries
    output.parent.mkdir(parents=True, exist_ok=True)
    entries_list = sorted(all_entries.values(), key=lambda e: e.timestamp)

    with open(output, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in entries_list], f, ensure_ascii=False, indent=2)

    logger.info(f"Merged {len(entries_list)} entries from {len(paths)} files to {output}")
    return len(entries_list)
