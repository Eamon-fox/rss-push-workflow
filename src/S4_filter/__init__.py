"""Step 4: Hierarchical filtering (cheap rules before LLM)."""

from .bio_keywords import filter_bio, BIO_KEYWORDS
from .hybrid import (
    filter_hybrid,
    filter_layer1,
    score_with_anchors,
    get_embedder,
    compute_embeddings_batch,
    HybridFilterStats,
)

__all__ = [
    "filter_bio",
    "BIO_KEYWORDS",
    "filter_hybrid",
    "filter_layer1",
    "score_with_anchors",
    "get_embedder",
    "compute_embeddings_batch",
    "HybridFilterStats",
]
