"""Article ID generation utilities."""

import hashlib
from typing import Any, Union


def generate_article_id(
    item: Union[dict, Any],
    doi_key: str = "doi",
    title_key: str = "title",
) -> str:
    """
    Generate article ID based on DOI or title hash.

    Priority:
    1. DOI-based ID: doi_10_1234_example
    2. Title hash ID: t_abc123def456

    Args:
        item: Article dict or object with doi/title attributes
        doi_key: Key/attribute name for DOI
        title_key: Key/attribute name for title

    Returns:
        Stable article ID string
    """
    # Extract values from dict or object
    if isinstance(item, dict):
        doi = item.get(doi_key)
        title = item.get(title_key) or ""
    else:
        doi = getattr(item, doi_key, None)
        title = getattr(item, title_key, "") or ""

    if doi:
        # Normalize DOI: replace / and . with _
        safe_doi = doi.replace("/", "_").replace(".", "_")
        return f"doi_{safe_doi}"

    # Fallback to title hash
    title_normalized = title.strip().lower()
    title_hash = hashlib.md5(title_normalized.encode()).hexdigest()[:12]
    return f"t_{title_hash}"
