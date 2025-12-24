"""Embedding cache using SQLite for persistent storage.

Caches text embeddings to avoid redundant model inference.
Cache key: SHA256 hash of text content (first 16 chars).
Cache priority: api > prod > dev (higher quality models preferred).
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


# Model priority patterns (higher = better quality)
# Order matters: first match wins
MODEL_PRIORITY_PATTERNS = [
    # Cloud APIs (highest quality)
    (["dashscope", "siliconflow", "openai", "cohere"], 100),
    # High-quality local GPU models
    (["bge-m3", "bge-large", "e5-large", "gte-large"], 50),
    # Lightweight local models (lowest priority)
    (["minilm", "all-minilm", "paraphrase"], 10),
]


def _get_model_priority(model_id: str) -> int:
    """Get priority for a model ID based on pattern matching."""
    model_lower = model_id.lower()
    for patterns, priority in MODEL_PRIORITY_PATTERNS:
        for pattern in patterns:
            if pattern in model_lower:
                return priority
    return 25  # Unknown model, medium priority


class Embedder(Protocol):
    """Protocol for embedding models."""
    def encode(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingCache:
    """SQLite-based embedding cache with model-aware invalidation and DOI lookup."""

    def __init__(self, db_path: str | Path, model_id: str):
        """
        Initialize cache.

        Args:
            db_path: Path to SQLite database file.
            model_id: Model identifier for cache invalidation.
        """
        self.db_path = Path(db_path)
        self.model_id = model_id
        self._db_lock = Lock()  # Thread lock for database operations
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_table()

    def _init_table(self):
        """Create cache table if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_id ON embedding_cache(model_id)
        """)
        # DOI lookup table for cross-source cache hits
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS doi_cache_map (
                doi TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _hash_text(self, text: str) -> str:
        """Compute SHA256 hash of text (first 16 chars)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI for consistent lookup."""
        if not doi:
            return ""
        doi = doi.lower().strip()
        # Remove common prefixes
        for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
        return doi

    def get_by_doi(self, doi: str, target_dim: int | None = None) -> np.ndarray | None:
        """
        Look up embedding by DOI (cross-source cache hit).

        Args:
            doi: Paper DOI.
            target_dim: Expected embedding dimension.

        Returns:
            Cached embedding or None if not found.
        """
        doi = self._normalize_doi(doi)
        if not doi:
            return None

        with self._db_lock:
            # Look up text_hash from DOI
            cursor = self.conn.execute(
                "SELECT text_hash FROM doi_cache_map WHERE doi = ?",
                (doi,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            text_hash = row[0]

            # Look up embedding by text_hash
            cursor = self.conn.execute(
                "SELECT model_id, embedding, dim FROM embedding_cache WHERE text_hash = ?",
                (text_hash,)
            )

            best_priority = -1
            best_emb = None

            for model_id, emb_blob, dim in cursor:
                if target_dim is not None and dim != target_dim:
                    continue
                priority = _get_model_priority(model_id)
                if priority > best_priority:
                    best_priority = priority
                    best_emb = np.frombuffer(emb_blob, dtype=np.float32).reshape(dim)

        return best_emb

    def get_many_by_doi(self, dois: list[str], target_dim: int | None = None) -> dict[int, np.ndarray]:
        """
        Batch lookup embeddings by DOI.

        Args:
            dois: List of DOIs.
            target_dim: Expected embedding dimension.

        Returns:
            Dict mapping original index -> embedding for cache hits.
        """
        result = {}
        for i, doi in enumerate(dois):
            emb = self.get_by_doi(doi, target_dim)
            if emb is not None:
                result[i] = emb
        return result

    def register_doi(self, doi: str, text: str):
        """
        Register DOI -> text_hash mapping for future cross-source lookups.

        Args:
            doi: Paper DOI.
            text: The text content (used to compute hash).
        """
        doi = self._normalize_doi(doi)
        if not doi:
            return

        text_hash = self._hash_text(text)
        try:
            with self._db_lock:
                self.conn.execute(
                    "INSERT OR REPLACE INTO doi_cache_map (doi, text_hash) VALUES (?, ?)",
                    (doi, text_hash)
                )
                self.conn.commit()
        except Exception:
            pass  # Ignore duplicate errors

    def register_dois_batch(self, items: list[tuple[str, str]]):
        """
        Batch register DOI -> text_hash mappings.

        Args:
            items: List of (doi, text) tuples.
        """
        rows = []
        for doi, text in items:
            doi = self._normalize_doi(doi)
            if doi:
                rows.append((doi, self._hash_text(text)))

        if rows:
            with self._db_lock:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO doi_cache_map (doi, text_hash) VALUES (?, ?)",
                    rows
                )
                self.conn.commit()

    def get_many(self, texts: list[str], target_dim: int | None = None) -> dict[int, np.ndarray]:
        """
        Batch lookup cached embeddings with priority-based model selection.

        Priority: api > prod > dev (uses highest priority available cache).
        Only returns embeddings matching target_dim if specified.

        Args:
            texts: List of texts to look up.
            target_dim: Expected embedding dimension (filters incompatible caches).

        Returns:
            Dict mapping original index -> embedding for cache hits.
        """
        if not texts:
            return {}

        hashes = [self._hash_text(t) for t in texts]
        placeholders = ",".join("?" * len(hashes))

        with self._db_lock:
            # Query all matching hashes (across all models)
            cursor = self.conn.execute(
                f"""
                SELECT text_hash, model_id, embedding, dim FROM embedding_cache
                WHERE text_hash IN ({placeholders})
                """,
                hashes,
            )

            # Group by hash, keep highest priority model for each
            hash_to_best: dict[str, tuple[int, np.ndarray]] = {}  # hash -> (priority, embedding)
            for row in cursor:
                text_hash, model_id, emb_blob, dim = row

                # Skip if dimension mismatch
                if target_dim is not None and dim != target_dim:
                    continue

                emb = np.frombuffer(emb_blob, dtype=np.float32).reshape(dim)
                priority = _get_model_priority(model_id)

                if text_hash not in hash_to_best or priority > hash_to_best[text_hash][0]:
                    hash_to_best[text_hash] = (priority, emb)

        # Map original index -> embedding
        result: dict[int, np.ndarray] = {}
        for i, h in enumerate(hashes):
            if h in hash_to_best:
                result[i] = hash_to_best[h][1]

        return result

    def set_many(self, texts: list[str], embeddings: np.ndarray):
        """
        Batch store embeddings in cache.

        Args:
            texts: List of texts.
            embeddings: Corresponding embeddings array (N x dim).
        """
        if len(texts) == 0:
            return

        dim = embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]

        rows = []
        for text, emb in zip(texts, embeddings):
            text_hash = self._hash_text(text)
            emb_blob = np.asarray(emb, dtype=np.float32).tobytes()
            rows.append((text_hash, self.model_id, emb_blob, dim))

        with self._db_lock:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (text_hash, model_id, embedding, dim)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            self.conn.commit()

    def clear(self, model_id: str | None = None):
        """Clear cache entries, optionally for specific model only."""
        with self._db_lock:
            if model_id:
                self.conn.execute("DELETE FROM embedding_cache WHERE model_id = ?", (model_id,))
            else:
                self.conn.execute("DELETE FROM embedding_cache")
            self.conn.commit()

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._db_lock:
            cursor = self.conn.execute("""
                SELECT model_id, COUNT(*) as count
                FROM embedding_cache
                GROUP BY model_id
            """)
            by_model = {row[0]: row[1] for row in cursor}

        total = sum(by_model.values())
        return {"total": total, "by_model": by_model}

    def close(self):
        """Close database connection."""
        with self._db_lock:
            self.conn.close()


class CachedEmbedder:
    """Embedder wrapper with caching support, cross-model priority lookup, and DOI-based lookup."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache, embedding_dim: int | None = None):
        """
        Wrap an embedder with caching.

        Args:
            inner: The actual embedder to use for cache misses.
            cache: EmbeddingCache instance.
            embedding_dim: Expected embedding dimension (auto-detected if None).
        """
        self.inner = inner
        self.cache = cache
        self._embedding_dim = embedding_dim
        self._stats = {"hits": 0, "misses": 0, "doi_hits": 0}

    def _detect_dim(self, sample_text: str = "dimension probe") -> int:
        """Detect embedding dimension by encoding a sample."""
        sample_emb = self.inner.encode([sample_text])
        sample_emb = np.asarray(sample_emb)
        return sample_emb.shape[1] if len(sample_emb.shape) > 1 else sample_emb.shape[0]

    def encode(self, texts: list[str], dois: list[str] | None = None) -> np.ndarray:
        """
        Encode texts with caching, DOI-based lookup, and cross-model priority lookup.

        Cache lookup order:
        1. DOI-based lookup (if DOI provided) - cross-source cache hit
        2. Text hash lookup - same text from any source
        3. Compute embedding and cache

        Args:
            texts: List of texts to encode.
            dois: Optional list of DOIs for cross-source cache lookup.

        Returns:
            Embeddings array (N x dim).
        """
        if not texts:
            return np.array([])

        n = len(texts)
        if dois is None or len(dois) != n:
            # Ensure dois has same length as texts
            dois = (list(dois) + [""] * n)[:n] if dois else [""] * n

        # Auto-detect dimension if not set
        if self._embedding_dim is None:
            self._embedding_dim = self._detect_dim()

        cached: dict[int, np.ndarray] = {}

        # 1. First try DOI-based lookup (cross-source cache hit)
        for i, doi in enumerate(dois):
            if doi:
                emb = self.cache.get_by_doi(doi, target_dim=self._embedding_dim)
                if emb is not None:
                    cached[i] = emb
                    self._stats["doi_hits"] += 1

        # 2. For remaining, try text hash lookup
        remaining_indices = [i for i in range(n) if i not in cached]
        if remaining_indices:
            remaining_texts = [texts[i] for i in remaining_indices]
            text_cached = self.cache.get_many(remaining_texts, target_dim=self._embedding_dim)
            for local_idx, emb in text_cached.items():
                orig_idx = remaining_indices[local_idx]
                cached[orig_idx] = emb

        self._stats["hits"] += len(cached)

        # 3. Find still missing indices
        missing_indices = [i for i in range(n) if i not in cached]
        self._stats["misses"] += len(missing_indices)

        # 4. If all cached, return directly
        if not missing_indices:
            logger.info(
                f"Embedding cache: {len(cached)} hits ({self._stats['doi_hits']} via DOI), "
                f"0 misses (100.0% hit rate)"
            )
            return np.stack([cached[i] for i in range(n)])

        # 5. Compute missing embeddings
        missing_texts = [texts[i] for i in missing_indices]
        new_embeddings = self.inner.encode(missing_texts)
        new_embeddings = np.asarray(new_embeddings)

        # 6. Store new embeddings in cache
        self.cache.set_many(missing_texts, new_embeddings)

        # 7. Register DOI -> text_hash mappings for future cross-source lookups
        doi_mappings = []
        for local_idx, orig_idx in enumerate(missing_indices):
            if dois[orig_idx]:
                doi_mappings.append((dois[orig_idx], texts[orig_idx]))
        if doi_mappings:
            self.cache.register_dois_batch(doi_mappings)

        # 8. Merge results in original order
        dim = self._embedding_dim
        result = np.zeros((n, dim), dtype=np.float32)

        # Fill cached
        for i, emb in cached.items():
            result[i] = emb

        # Fill new
        for idx, orig_i in enumerate(missing_indices):
            result[orig_i] = new_embeddings[idx]

        # Log cache performance
        logger.info(
            f"Embedding cache: {len(cached)} hits ({self._stats['doi_hits']} via DOI), "
            f"{len(missing_indices)} misses "
            f"({len(cached)/(len(cached)+len(missing_indices))*100:.1f}% hit rate)"
        )

        return result

    @property
    def stats(self) -> dict:
        """Get hit/miss statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {**self._stats, "total": total, "hit_rate": f"{hit_rate:.1%}"}

    def reset_stats(self):
        """Reset hit/miss counters."""
        self._stats = {"hits": 0, "misses": 0}

    def log_session_summary(self):
        """Log session summary to file."""
        stats = self.stats
        logger.info(
            f"Session summary: {stats['hits']} cache hits, {stats['misses']} API calls, "
            f"{stats['hit_rate']} hit rate"
        )
