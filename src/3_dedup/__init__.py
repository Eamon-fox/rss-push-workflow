"""Step 3: Fingerprint-based deduplication."""

from .fingerprint import get_fingerprint
from .seen import load, save, mark_seen, mark_batch, cleanup
from .filter import filter_unseen, filter_duplicates_in_batch

__all__ = [
    "get_fingerprint",
    "load",
    "save",
    "mark_seen",
    "mark_batch",
    "cleanup",
    "filter_unseen",
    "filter_duplicates_in_batch",
]
