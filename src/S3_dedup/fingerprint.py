"""Fingerprint calculation for deduplication."""

import re
import hashlib

from ..models import NewsItem


def normalize_doi(doi: str) -> str:
    """
    Normalize DOI to standard format.

    Examples:
        "doi:10.1038/xxx" -> "10.1038/xxx"
        "https://doi.org/10.1038/xxx" -> "10.1038/xxx"
        "10.1038/XXX" -> "10.1038/xxx"
    """
    if not doi:
        return ""

    # Remove common prefixes
    doi = doi.strip()
    prefixes = ["doi:", "https://doi.org/", "http://doi.org/", "DOI:"]
    for prefix in prefixes:
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):]
            break

    return doi.lower().strip()


def normalize_title(title: str) -> str:
    """
    Normalize title for hashing.

    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    """
    if not title:
        return ""

    title = title.lower()
    title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
    title = re.sub(r'\s+', ' ', title)      # Collapse whitespace
    return title.strip()


def hash_title(title: str) -> str:
    """Hash normalized title to MD5."""
    normalized = normalize_title(title)
    if not normalized:
        return ""
    return hashlib.md5(normalized.encode()).hexdigest()


def get_fingerprint(item: NewsItem) -> str:
    """
    Get unique fingerprint for item.

    Priority:
        1. DOI (most reliable)
        2. Title hash (fallback)

    Returns:
        Fingerprint string, or empty string if cannot compute
    """
    # Try DOI first
    if item.doi:
        normalized = normalize_doi(item.doi)
        if normalized:
            return normalized

    # Fallback to title hash
    if item.title:
        return hash_title(item.title)

    return ""
