"""Hybrid filter (Layer1 regex + Layer2 tiered scoring).

New scoring formula:
  final_score = base_score × anchor_tier_mult × coverage_mult × vip_mult

All multipliers are relative, making the system adaptive to different score distributions.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple, cast

import numpy as np

logger = logging.getLogger(__name__)

from ..models import NewsItem
from .embedding_cache import CachedEmbedder, EmbeddingCache
from .config import (
    GENERAL_BIO_KEYWORDS,
    SCORING_CONFIG,
    TIERED_ANCHORS,
    TIERED_VIP_KEYWORDS,
    # Legacy exports for backwards compatibility
    SEMANTIC_ANCHORS,
    THRESHOLD_NORMAL,
    VIP_KEYWORDS,
)

FILTER_DIR = Path("data/filtered")
EMBEDDING_CACHE_PATH = Path("data/embedding_cache.db")

# Global cache instance (initialized on first use)
_embedding_cache: EmbeddingCache | None = None


def _get_embedding_cache(model_id: str) -> EmbeddingCache:
    """Get or create embedding cache for given model."""
    global _embedding_cache
    if _embedding_cache is None or _embedding_cache.model_id != model_id:
        _embedding_cache = EmbeddingCache(EMBEDDING_CACHE_PATH, model_id)
    return _embedding_cache


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> object:
        """Return L2-normalized embeddings for given texts (numpy array or list)."""


def _load_semantic_config() -> dict:
    """Load semantic model config from settings.yaml."""
    import yaml
    config_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    return settings.get("semantic", {})


# Model presets (local models)
_MODEL_PRESETS = {
    "dev": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "trust_remote_code": False,
        "instruction": None,
    },
    "prod": {
        "model_id": "BAAI/bge-m3",
        "device": "cuda",
        "trust_remote_code": True,
        "instruction": "Represent passage for retrieval: {}",
    },
}


class _DashScopeEmbedder:
    """DashScope API-based embedder (cloud, no local GPU needed)."""

    def __init__(self, model: str = "text-embedding-v4", dimension: int = 1024, concurrency: int = 4):
        self.model = model
        self.dimension = dimension
        self.concurrency = concurrency
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY not set. Required for api preset. "
                "Get your key from https://dashscope.console.aliyun.com/"
            )

    def encode(self, texts: list[str]) -> object:
        """Batch encode texts using DashScope API."""
        import dashscope
        from http import HTTPStatus
        from concurrent.futures import ThreadPoolExecutor, as_completed

        dashscope.api_key = self._api_key

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        results = [None] * len(texts)
        batch_size = 10

        def process_batch(batch_info: tuple[int, list[tuple[int, str]]]) -> list[tuple[int, list[float]]]:
            batch_idx, items = batch_info
            indices, batch_texts = zip(*items) if items else ([], [])
            batch_texts = [t if t.strip() else " " for t in batch_texts]

            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=list(batch_texts),
                dimension=self.dimension,
            )
            if resp.status_code != HTTPStatus.OK:
                raise ValueError(f"DashScope API error (batch {batch_idx}): {resp.message}")

            embeddings = [e["embedding"] for e in resp.output["embeddings"]]
            return list(zip(indices, embeddings))

        batches: list[tuple[int, list[tuple[int, str]]]] = []
        current_batch: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            current_batch.append((i, text))
            if len(current_batch) >= batch_size:
                batches.append((len(batches), current_batch))
                current_batch = []
        if current_batch:
            batches.append((len(batches), current_batch))

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(process_batch, batch): batch[0] for batch in batches}
            for future in as_completed(futures):
                for idx, embedding in future.result():
                    results[idx] = embedding

        arr = np.array(results, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return arr / norms


def _default_embedder() -> Embedder:
    """Lazily construct the embedder based on config/settings.yaml."""
    semantic_cfg = _load_semantic_config()
    preset = os.environ.get("SEMANTIC_PRESET") or semantic_cfg.get("preset", "dev")

    # Cache configuration
    cache_enabled = semantic_cfg.get("cache_enabled", True)  # Default enabled

    if preset == "api":
        api_cfg = semantic_cfg.get("api", {})
        provider = api_cfg.get("provider", "dashscope")
        model = api_cfg.get("model", "text-embedding-v4")
        dimension = int(api_cfg.get("dimension", 1024))
        concurrency = int(api_cfg.get("concurrency", 4))

        if provider == "dashscope":
            print(f"      Semantic API: DashScope {model} (dim={dimension}, preset={preset})")
            inner = _DashScopeEmbedder(model=model, dimension=dimension, concurrency=concurrency)
            if cache_enabled:
                cache = _get_embedding_cache(f"dashscope:{model}")
                print(f"      Embedding cache: enabled (model={model})")
                return CachedEmbedder(inner, cache)
            return inner
        else:
            raise ValueError(f"Unknown API provider: {provider}.")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Missing dependency: sentence-transformers.") from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError("Missing dependency: torch.") from e

    batch_size = int(os.environ.get("HYBRID_EMBED_BATCH") or semantic_cfg.get("batch_size", 16))
    local_files_only = semantic_cfg.get("local_files_only", False)

    if preset in _MODEL_PRESETS:
        cfg = _MODEL_PRESETS[preset]
        model_id = os.environ.get("HYBRID_EMBED_MODEL") or cfg["model_id"]
        device = cfg["device"]
        trust_remote_code = cfg["trust_remote_code"]
        instruction = cfg["instruction"]
    elif preset == "custom":
        custom_cfg = semantic_cfg.get("custom", {})
        model_id = os.environ.get("HYBRID_EMBED_MODEL") or custom_cfg.get("model_id", "sentence-transformers/all-MiniLM-L6-v2")
        device = custom_cfg.get("device", "cpu")
        trust_remote_code = custom_cfg.get("trust_remote_code", False)
        instruction = custom_cfg.get("instruction")
    else:
        raise ValueError(f"Unknown semantic preset: {preset}.")

    instruction = os.environ.get("HYBRID_EMBED_INSTRUCTION", instruction)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Preset '{preset}' requires CUDA but no GPU available.")

    print(f"      Semantic model: {model_id} on {device} (preset={preset})")

    model = SentenceTransformer(
        model_id,
        device=device,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    class _STEmbedder:
        def encode(self, texts: list[str]) -> object:
            if instruction:
                formatted = [
                    instruction.format(text or "") if "{}" in instruction
                    else f"{instruction} {text or ''}".strip()
                    for text in texts
                ]
            else:
                formatted = [text or "" for text in texts]
            emb = model.encode(
                formatted,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return emb

    inner = _STEmbedder()
    if cache_enabled:
        cache = _get_embedding_cache(model_id)
        print(f"      Embedding cache: enabled (model={model_id})")
        return CachedEmbedder(inner, cache)
    return inner


# ============================================================
# VIP Keyword Matching (Tiered)
# ============================================================

def _clean_keyword_for_display(kw: str) -> str:
    """Remove regex markers like \\b for display."""
    return kw.replace(r"\b", "").strip()


def _compile_tiered_vip_patterns() -> Dict[str, List[Tuple[str, re.Pattern]]]:
    """Compile VIP patterns by tier."""
    result = {}
    for tier, patterns in [
        ("tier1", TIERED_VIP_KEYWORDS.tier1_patterns),
        ("tier2", TIERED_VIP_KEYWORDS.tier2_patterns),
        ("tier3", TIERED_VIP_KEYWORDS.tier3_patterns),
    ]:
        result[tier] = [
            (_clean_keyword_for_display(p), re.compile(p, re.IGNORECASE))
            for p in patterns
        ]
    return result


_TIERED_VIP_PATTERNS = _compile_tiered_vip_patterns()


def _find_vip_matches(text: str) -> Dict[str, List[str]]:
    """Find VIP keyword matches by tier.

    Returns: {"tier1": ["RTCB", ...], "tier2": ["IRE1", ...], ...}
    """
    matches: Dict[str, List[str]] = {"tier1": [], "tier2": [], "tier3": []}
    for tier, patterns in _TIERED_VIP_PATTERNS.items():
        for display_name, pattern in patterns:
            if pattern.search(text):
                matches[tier].append(display_name)
    return matches


def _calculate_vip_multiplier(vip_matches: Dict[str, List[str]]) -> Tuple[float, List[str]]:
    """Calculate VIP multiplier based on tiered matches.

    Returns: (multiplier, list of matched keywords for display)
    """
    all_keywords = []
    tiers_hit = []

    for tier in ["tier1", "tier2", "tier3"]:
        if vip_matches[tier]:
            all_keywords.extend(vip_matches[tier])
            tiers_hit.append(tier)

    if not tiers_hit:
        return 1.0, []

    # Get base multiplier from highest tier hit
    if "tier1" in tiers_hit:
        base_mult = TIERED_VIP_KEYWORDS.tier1_multiplier
    elif "tier2" in tiers_hit:
        base_mult = TIERED_VIP_KEYWORDS.tier2_multiplier
    else:
        base_mult = TIERED_VIP_KEYWORDS.tier3_multiplier

    # Stack bonus for multiple tiers hit
    extra_tiers = len(tiers_hit) - 1
    stack_bonus = extra_tiers * TIERED_VIP_KEYWORDS.stack_bonus
    final_mult = base_mult + stack_bonus

    # Cap at max multiplier
    final_mult = min(final_mult, TIERED_VIP_KEYWORDS.max_multiplier)

    return final_mult, all_keywords


# ============================================================
# Bio Keywords Filter (Layer 1)
# ============================================================

def _compile_union_regex(terms: list[str], *, escape: bool) -> re.Pattern[str]:
    parts = [re.escape(t) for t in terms] if escape else list(terms)
    return re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE)


_GENERAL_BIO_RE = _compile_union_regex(GENERAL_BIO_KEYWORDS, escape=True)
_VIP_RE = _compile_union_regex(VIP_KEYWORDS, escape=False)


# ============================================================
# Semantic Scoring (Tiered Anchors)
# ============================================================

_ANCHOR_CACHE_ATTR = "_semantic_anchor_cache"
_NEGATIVE_ANCHOR_CACHE_ATTR = "_negative_anchor_cache"


def _anchor_embeddings(embedder: Embedder, anchors: list[str]) -> Any:
    """Compute anchor embeddings once per embedder instance."""
    if hasattr(embedder, _ANCHOR_CACHE_ATTR):
        cached = getattr(embedder, _ANCHOR_CACHE_ATTR)
        if cached is not None:
            return cached

    embeddings = embedder.encode(anchors)
    try:
        setattr(embedder, _ANCHOR_CACHE_ATTR, embeddings)
    except Exception:
        pass
    return embeddings


def _negative_anchor_embeddings(embedder: Embedder, anchors: list[str]) -> Any:
    """Compute negative anchor embeddings once per embedder instance."""
    if hasattr(embedder, _NEGATIVE_ANCHOR_CACHE_ATTR):
        cached = getattr(embedder, _NEGATIVE_ANCHOR_CACHE_ATTR)
        if cached is not None:
            return cached

    embeddings = embedder.encode(anchors)
    try:
        setattr(embedder, _NEGATIVE_ANCHOR_CACHE_ATTR, embeddings)
    except Exception:
        pass
    return embeddings


def _calculate_negative_penalty(
    negative_sims: List[float],
    threshold: float,
    penalty: float,
) -> Tuple[float, float]:
    """Calculate negative penalty based on similarity to negative anchors.

    Args:
        negative_sims: List of cosine similarities to negative anchors
        threshold: Similarity threshold to trigger penalty
        penalty: Penalty multiplier (e.g., 0.6 means 40% reduction)

    Returns:
        (penalty_multiplier, max_negative_similarity)
    """
    if not negative_sims:
        return 1.0, 0.0

    max_neg_sim = max(negative_sims)
    if max_neg_sim >= threshold:
        return penalty, max_neg_sim
    return 1.0, max_neg_sim


def _calculate_tiered_anchor_score(
    similarities: List[float],
) -> Tuple[float, float, float]:
    """Calculate base score, tier multiplier, and coverage multiplier.

    Args:
        similarities: List of cosine similarities to each anchor (in order: tier1, tier2, tier3)

    Returns:
        (base_score, tier_multiplier, coverage_multiplier)
    """
    if not similarities:
        return 0.0, 1.0, 1.0

    # Get tier boundaries
    n_tier1 = len(TIERED_ANCHORS.tier1)
    n_tier2 = len(TIERED_ANCHORS.tier2)

    tier1_sims = similarities[:n_tier1]
    tier2_sims = similarities[n_tier1:n_tier1 + n_tier2]
    tier3_sims = similarities[n_tier1 + n_tier2:]

    # Base score: max across all anchors
    base_score = max(similarities)

    # Find which tier the best match belongs to
    best_idx = similarities.index(base_score)
    best_tier = TIERED_ANCHORS.get_tier(best_idx)
    tier_multiplier = SCORING_CONFIG.anchor_tier_multiplier.get(best_tier, 1.0)

    # Coverage multiplier: bonus for hitting multiple tiers
    threshold = SCORING_CONFIG.anchor_coverage_threshold
    coverage_bonus = 0.0

    if tier1_sims and max(tier1_sims) > threshold:
        coverage_bonus += SCORING_CONFIG.anchor_coverage_bonus.get("tier1", 0.0)
    if tier2_sims and max(tier2_sims) > threshold:
        coverage_bonus += SCORING_CONFIG.anchor_coverage_bonus.get("tier2", 0.0)
    if tier3_sims and max(tier3_sims) > threshold:
        coverage_bonus += SCORING_CONFIG.anchor_coverage_bonus.get("tier3", 0.0)

    coverage_multiplier = 1.0 + coverage_bonus

    return base_score, tier_multiplier, coverage_multiplier


# ============================================================
# Main Filter Logic
# ============================================================

def _item_text(item: NewsItem, *, max_len: int = 2000) -> str:
    title = (item.title or "").strip()
    content = (item.content or "").strip()
    text = (title + "\n" + content).strip()
    return text if len(text) <= max_len else text[:max_len]


def _cos_sim_matrix(a: object, b: object) -> list[list[float]]:
    """Cosine similarity for L2-normalized embeddings."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    if a_arr.size == 0 or b_arr.size == 0:
        return []
    sims = a_arr @ b_arr.T
    return cast(list[list[float]], sims.tolist())


@dataclass(frozen=True)
class HybridFilterStats:
    total: int
    layer1_dropped: int
    layer2_kept: int
    layer2_dropped: int
    avg_final_score: float = 0.0
    min_final_score: float = 0.0
    max_final_score: float = 0.0


def filter_hybrid(
    items: list[NewsItem],
    *,
    embedder: Embedder | None = None,
    seen_records: dict[str, str] | None = None,
    record_layer2_dropped_to_seen: bool = True,
    save: bool = True,
) -> tuple[list[NewsItem], list[NewsItem], HybridFilterStats]:
    """
    Step 4 hybrid filtering with tiered scoring.

    Scoring formula:
      final_score = base_score × anchor_tier_mult × coverage_mult × vip_mult × negative_penalty

    Negative penalty is applied when article matches negative anchors (e.g., clinical trials,
    agricultural applications, pure computational work, reviews/commentaries).

    Returns:
      (kept_items, dropped_items, stats)
    """
    if not items:
        return [], [], HybridFilterStats(0, 0, 0, 0)

    print(f"      Loading embedding model...")
    embedder = embedder or _default_embedder()
    print(f"      Model loaded.")

    layer1_pass: list[NewsItem] = []
    dropped: list[NewsItem] = []

    # Layer 1: coarse bio filter
    print(f"      Layer 1: filtering {len(items)} items by bio keywords...")
    for item in items:
        text = _item_text(item)
        if _VIP_RE.search(text) or _GENERAL_BIO_RE.search(text):
            layer1_pass.append(item)
        else:
            dropped.append(item)
    print(f"      Layer 1: {len(layer1_pass)} passed, {len(dropped)} dropped")

    # Layer 2: tiered semantic + VIP scoring
    print(f"      Layer 2: scoring {len(layer1_pass)} items...")
    kept: list[NewsItem] = []
    layer2_dropped: list[NewsItem] = []
    final_scores: list[float] = []

    # Prepare valid items
    valid_items: list[NewsItem] = []
    valid_texts: list[str] = []
    for item in layer1_pass:
        text = _item_text(item)
        if text:
            valid_items.append(item)
            valid_texts.append(text)
        else:
            layer2_dropped.append(item)

    if not valid_items:
        stats = HybridFilterStats(
            total=len(items),
            layer1_dropped=len(dropped),
            layer2_kept=0,
            layer2_dropped=len(layer2_dropped),
        )
        dropped.extend(layer2_dropped)
        return kept, dropped, stats

    # Get anchors
    anchors = TIERED_ANCHORS.all_anchors()
    if not anchors:
        print("      Warning: No semantic anchors configured!")
        dropped.extend(valid_items)
        stats = HybridFilterStats(
            total=len(items),
            layer1_dropped=len(items) - len(layer1_pass),
            layer2_kept=0,
            layer2_dropped=len(valid_items),
        )
        return kept, dropped, stats

    # Compute anchor embeddings
    print(f"      Computing anchor embeddings ({len(anchors)} anchors: {len(TIERED_ANCHORS.tier1)} T1, {len(TIERED_ANCHORS.tier2)} T2, {len(TIERED_ANCHORS.tier3)} T3)...")
    anchor_emb = _anchor_embeddings(embedder, anchors)

    # Compute negative anchor embeddings (if configured)
    negative_anchors = TIERED_ANCHORS.negative
    negative_emb = None
    if negative_anchors:
        print(f"      Computing negative anchor embeddings ({len(negative_anchors)} negative anchors)...")
        negative_emb = _negative_anchor_embeddings(embedder, negative_anchors)

    # Batch encode items
    batch_size = int(os.environ.get("HYBRID_EMBED_BATCH", "16"))
    total_items = len(valid_items)
    all_sims: list[list[float]] = []
    all_negative_sims: list[list[float]] = []

    for batch_start in range(0, total_items, batch_size):
        batch_end = min(batch_start + batch_size, total_items)
        batch_texts = valid_texts[batch_start:batch_end]

        print(f"      Encoding batch {batch_start // batch_size + 1}/{(total_items + batch_size - 1) // batch_size} ({batch_end}/{total_items})...")
        batch_emb = embedder.encode(batch_texts)
        batch_sims = _cos_sim_matrix(batch_emb, anchor_emb)
        all_sims.extend(batch_sims)

        # Compute negative similarities if configured
        if negative_emb is not None:
            batch_neg_sims = _cos_sim_matrix(batch_emb, negative_emb)
            all_negative_sims.extend(batch_neg_sims)

    print(f"      Computing final scores...")

    threshold = SCORING_CONFIG.final_threshold
    negative_penalty_count = 0  # Track how many items were penalized

    # Prepare negative similarities iterator
    neg_sims_iter = iter(all_negative_sims) if all_negative_sims else iter([])

    for item, text, sims in zip(valid_items, valid_texts, all_sims):
        # Calculate anchor-based score components
        base_score, tier_mult, coverage_mult = _calculate_tiered_anchor_score(sims)

        # Calculate VIP multiplier
        vip_matches = _find_vip_matches(text)
        vip_mult, vip_keywords = _calculate_vip_multiplier(vip_matches)

        # Calculate negative penalty
        neg_penalty = 1.0
        max_neg_sim = 0.0
        if all_negative_sims:
            neg_sims = next(neg_sims_iter, [])
            neg_penalty, max_neg_sim = _calculate_negative_penalty(
                neg_sims,
                SCORING_CONFIG.negative_threshold,
                SCORING_CONFIG.negative_penalty,
            )
            if neg_penalty < 1.0:
                negative_penalty_count += 1

        # Final score
        final_score = base_score * tier_mult * coverage_mult * vip_mult * neg_penalty
        final_score = min(1.0, final_score)  # Cap at 1.0

        # Store in item
        item.semantic_score = round(final_score, 4)
        item.is_vip = bool(vip_keywords)
        item.vip_keywords = vip_keywords

        # Filter by threshold
        if final_score >= threshold:
            kept.append(item)
            final_scores.append(final_score)
        else:
            layer2_dropped.append(item)

    dropped.extend(layer2_dropped)

    # Print summary
    if final_scores:
        avg_score = sum(final_scores) / len(final_scores)
        min_score = min(final_scores)
        max_score = max(final_scores)
        print(f"      Layer 2: {len(kept)} kept, {len(layer2_dropped)} dropped (threshold={threshold})")
        print(f"      Final scores: avg={avg_score:.3f}, min={min_score:.3f}, max={max_score:.3f}")
        if negative_penalty_count > 0:
            print(f"      Negative anchors: {negative_penalty_count} items penalized")
    else:
        avg_score = min_score = max_score = 0.0
        print(f"      Layer 2: 0 kept, {len(layer2_dropped)} dropped")

    # Print cache statistics if available
    if isinstance(embedder, CachedEmbedder):
        cache_stats = embedder.stats
        print(f"      Cache: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']})")

    # Record dropped to seen
    if record_layer2_dropped_to_seen and seen_records is not None and layer2_dropped:
        try:
            fingerprint_mod = importlib.import_module("src.S3_dedup.fingerprint")
            seen_mod = importlib.import_module("src.S3_dedup.seen")
            get_fingerprint = getattr(fingerprint_mod, "get_fingerprint")
            mark_batch = getattr(seen_mod, "mark_batch")
            fps = [get_fingerprint(it) for it in layer2_dropped]
            fps = [fp for fp in fps if fp]
            if fps:
                mark_batch(seen_records, fps)
        except Exception:
            pass

    if save and kept:
        _save_filtered(kept)

    stats = HybridFilterStats(
        total=len(items),
        layer1_dropped=len(items) - len(layer1_pass),
        layer2_kept=len(kept),
        layer2_dropped=len(layer2_dropped),
        avg_final_score=round(avg_score, 4),
        min_final_score=round(min_score, 4),
        max_final_score=round(max_score, 4),
    )

    # Log filter results and cache stats
    logger.info(
        f"Hybrid filter: {stats.total} total -> {stats.layer2_kept} kept "
        f"(L1 dropped {stats.layer1_dropped}, L2 dropped {stats.layer2_dropped})"
    )
    if isinstance(embedder, CachedEmbedder):
        embedder.log_session_summary()

    return kept, dropped, stats


def _save_filtered(items: list[NewsItem]) -> None:
    """Save filtered results to data/filtered/{date}/all.json, sorted by score."""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = FILTER_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    sorted_items = _sort_by_score(items)

    filepath = dir_path / "all.json"
    data = [item.model_dump() for item in sorted_items]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved filtered to {filepath}")


def _sort_by_score(items: list[NewsItem]) -> list[NewsItem]:
    """Return items sorted by semantic_score descending."""
    if not items:
        return []

    return sorted(
        items,
        key=lambda it: it.semantic_score if it.semantic_score is not None else float("-inf"),
        reverse=True,
    )


# Legacy function for backwards compatibility
def _find_vip_keywords(text: str) -> list[str]:
    """Find all VIP keywords that match in the text (legacy)."""
    matches = _find_vip_matches(text)
    return matches["tier1"] + matches["tier2"] + matches["tier3"]


def _aggregate_semantic_score(similarities: list[float]) -> float:
    """Legacy scoring function."""
    base, tier_mult, cov_mult = _calculate_tiered_anchor_score(similarities)
    return min(1.0, base * tier_mult * cov_mult)


def _sort_by_semantic(items: list[NewsItem]) -> list[NewsItem]:
    """Legacy alias."""
    return _sort_by_score(items)
