"""Deduplication logic with 72-hour window."""

import hashlib
from datetime import datetime, timedelta

from .models import Paper, DedupeResult
from .database import Database


class Deduplicator:
    """Check and handle duplicate papers."""

    def __init__(self, db: Database, window_hours: int = 72):
        self.db = db
        self.window_hours = window_hours

    def check(self, paper: Paper) -> DedupeResult:
        """Check if paper is a duplicate."""
        # TODO: Implement full check logic
        pass

    def exists(self, doi: str) -> bool:
        """Quick check: does this DOI exist?"""
        if not doi:
            return False
        return self.db.exists_by_doi(doi)

    @staticmethod
    def title_hash(title: str) -> str:
        """Generate normalized title hash."""
        # Remove punctuation, lowercase, hash
        normalized = "".join(c.lower() for c in title if c.isalnum() or c.isspace())
        normalized = " ".join(normalized.split())  # Normalize whitespace
        return hashlib.md5(normalized.encode()).hexdigest()

    @staticmethod
    def title_similarity(title1: str, title2: str) -> float:
        """Calculate title similarity (0-1)."""
        # TODO: Implement fuzzy matching
        pass
