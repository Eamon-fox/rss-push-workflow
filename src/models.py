"""Core data models for ScholarPipe."""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class PaperStatus(str, Enum):
    """Paper processing status."""
    NEW = "new"
    SCREENED = "screened"
    FETCHED = "fetched"
    PROCESSED = "processed"
    SKIPPED = "skipped"


class Paper(BaseModel):
    """Core paper model that flows through the entire pipeline."""

    doi: str | None = None
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    source: str = ""
    url: str = ""

    # Processing metadata
    score: float | None = None
    score_reason: str = ""
    status: PaperStatus = PaperStatus.NEW
    pdf_path: str | None = None
    summary: str = ""

    # Timestamps
    published_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class DedupeResult(BaseModel):
    """Result of deduplication check."""

    is_duplicate: bool
    existing_doi: str | None = None
    match_type: str = ""  # "exact_doi", "title_hash", "similar_title"
    similarity: float = 0.0
