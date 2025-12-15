"""SQLite database operations."""

import sqlite3
from datetime import datetime
from pathlib import Path

from .models import Paper, PaperStatus


class Database:
    """SQLite database for paper storage."""

    def __init__(self, db_path: str = "data/history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doi TEXT UNIQUE,
                    title TEXT NOT NULL,
                    title_hash TEXT,
                    abstract TEXT,
                    authors TEXT,
                    source TEXT,
                    url TEXT,
                    score REAL,
                    score_reason TEXT,
                    status TEXT DEFAULT 'new',
                    pdf_path TEXT,
                    summary TEXT,
                    published_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON papers(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created ON papers(created_at)
            """)

    def insert(self, paper: Paper) -> int:
        """Insert a new paper. Returns the row ID."""
        # TODO: Implement
        pass

    def exists_by_doi(self, doi: str) -> bool:
        """Check if paper with DOI exists."""
        # TODO: Implement
        pass

    def get_by_status(self, status: PaperStatus) -> list[Paper]:
        """Get papers by status."""
        # TODO: Implement
        pass

    def get_by_min_score(self, min_score: float) -> list[Paper]:
        """Get papers with score >= min_score."""
        # TODO: Implement
        pass

    def update_score(self, doi: str, score: float, reason: str = "") -> None:
        """Update paper score."""
        # TODO: Implement
        pass

    def update_status(self, doi: str, status: PaperStatus) -> None:
        """Update paper status."""
        # TODO: Implement
        pass

    def update_pdf_path(self, doi: str, pdf_path: str) -> None:
        """Update paper PDF path."""
        # TODO: Implement
        pass
