"""Hybrid filter (Layer1 regex + Layer2 VIP + semantic vector fallback)."""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from ..models import NewsItem
from .config import (
    GENERAL_BIO_KEYWORDS,
    SEMANTIC_ANCHORS,
    THRESHOLD_NORMAL,
    VIP_KEYWORDS,
)

FILTER_DIR = Path("data/filtered")


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> object:
        """Return L2-normalized embeddings for given texts (numpy array or list)."""


def _default_embedder() -> Embedder:
    """
    Lazily construct the GPU-accelerated embedder (BGE-M3 by default).
    Requires CUDA; raises if no compatible GPU is available.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from e

    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: torch. Install a GPU build "
            "(e.g. pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121)"
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Hybrid filtering requires a CUDA-enabled GPU. "
            "Install the matching CUDA build of torch and ensure GPU drivers are available."
        )

    model_id = os.environ.get("HYBRID_EMBED_MODEL", "BAAI/bge-m3")
    batch_size = int(os.environ.get("HYBRID_EMBED_BATCH", "16"))
    device = "cuda"

    model = SentenceTransformer(
        model_id,
        device=device,
        trust_remote_code=True,  # BGE-M3 exposes custom encode logic via remote code
        local_files_only=True,  # Don't auto-download; fail if model not present
    )

    instruction = os.environ.get(
        "HYBRID_EMBED_INSTRUCTION",
        "Represent passage for retrieval: {}",
    )

    class _STEmbedder:
        def encode(self, texts: list[str]) -> object:
            formatted = [
                instruction.format(text or "")
                if "{}" in instruction
                else f"{instruction} {text or ''}".strip()
                for text in texts
            ]
            emb = model.encode(
                formatted,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return emb

    return _STEmbedder()


def _compile_union_regex(terms: list[str], *, escape: bool) -> re.Pattern[str]:
    parts = [re.escape(t) for t in terms] if escape else list(terms)
    return re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE)


_GENERAL_BIO_RE = _compile_union_regex(GENERAL_BIO_KEYWORDS, escape=True)
_VIP_RE = _compile_union_regex(VIP_KEYWORDS, escape=False)
_ANCHOR_CACHE_ATTR = "_semantic_anchor_cache"

# Compile individual VIP patterns for matching
# Store (display_name, pattern) - strip regex markers for display
def _clean_keyword_for_display(kw: str) -> str:
    """Remove regex markers like \\b for display."""
    return kw.replace(r"\b", "").strip()

_VIP_PATTERNS = [
    (_clean_keyword_for_display(kw), re.compile(kw, re.IGNORECASE))
    for kw in VIP_KEYWORDS
]


def _find_vip_keywords(text: str) -> list[str]:
    """Find all VIP keywords that match in the text."""
    matched = []
    for display_name, pattern in _VIP_PATTERNS:
        if pattern.search(text):
            matched.append(display_name)
    return matched


def _item_text(item: NewsItem, *, max_len: int = 2000) -> str:
    title = (item.title or "").strip()
    content = (item.content or "").strip()
    text = (title + "\n" + content).strip()
    return text if len(text) <= max_len else text[:max_len]


def _cos_sim_matrix(a: object, b: object) -> list[list[float]]:
    """
    Cosine similarity for L2-normalized embeddings:
      sim(a_i, b_j) = dot(a_i, b_j)
    """
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    if a_arr.size == 0 or b_arr.size == 0:
        return []
    # Embeddings are already L2-normalized -> cosine similarity == dot product.
    sims = a_arr @ b_arr.T
    return cast(list[list[float]], sims.tolist())


@dataclass(frozen=True)
class HybridFilterStats:
    total: int
    layer1_dropped: int
    layer2_vip_kept: int
    layer2_semantic_kept: int
    layer2_semantic_dropped: int


def filter_hybrid(
    items: list[NewsItem],
    *,
    embedder: Embedder | None = None,
    seen_records: dict[str, str] | None = None,
    record_layer2_dropped_to_seen: bool = True,
    save: bool = True,
) -> tuple[list[NewsItem], list[NewsItem], HybridFilterStats]:
    """
    Step 4 hybrid filtering.

    Returns:
      (kept_items, dropped_items, stats)
    """
    if not items:
        return [], [], HybridFilterStats(0, 0, 0, 0, 0)

    print(f"      Loading embedding model...")
    embedder = embedder or _default_embedder()
    print(f"      Model loaded.")

    layer1_pass: list[NewsItem] = []
    dropped: list[NewsItem] = []

    # Layer 1: coarse bio filter
    print(f"      Layer 1: filtering {len(items)} items by bio keywords...")
    for item in items:
        text = _item_text(item)
        # VIP terms should never be lost at Layer 1.
        if _VIP_RE.search(text) or _GENERAL_BIO_RE.search(text):
            layer1_pass.append(item)
        else:
            dropped.append(item)
    print(f"      Layer 1: {len(layer1_pass)} passed, {len(dropped)} dropped")

    # Layer 2: VIP + semantic scoring (all items get scored)
    print(f"      Layer 2: semantic scoring {len(layer1_pass)} items...")
    kept: list[NewsItem] = []
    vip_kept = 0
    semantic_kept = 0
    semantic_dropped: list[NewsItem] = []

    # Identify VIP items and their matched keywords
    vip_matches: list[list[str]] = []  # List of matched VIP keywords per item
    valid_items: list[NewsItem] = []
    for item in layer1_pass:
        text = _item_text(item)
        if not text:
            semantic_dropped.append(item)
            continue
        matched_keywords = _find_vip_keywords(text)
        vip_matches.append(matched_keywords)
        valid_items.append(item)

    # Compute semantic scores for ALL valid items (including VIP)
    if valid_items:
        anchors = [a.strip() for a in SEMANTIC_ANCHORS if a and a.strip()]
        if not anchors:
            # No anchors configured - VIP pass, others drop
            for item, matched_kws in zip(valid_items, vip_matches):
                if matched_kws:
                    item.is_vip = True
                    item.vip_keywords = matched_kws
                    kept.append(item)
                    vip_kept += 1
                else:
                    semantic_dropped.append(item)
        else:
            print(f"      Computing anchor embeddings ({len(anchors)} anchors)...")
            anchor_emb = _anchor_embeddings(embedder, anchors)

            # 分批编码，边算边出结果，避免长时间卡住
            batch_size = int(os.environ.get("HYBRID_EMBED_BATCH", "16"))
            total_items = len(valid_items)
            sims: list[list[float]] = []

            for batch_start in range(0, total_items, batch_size):
                batch_end = min(batch_start + batch_size, total_items)
                batch_items = valid_items[batch_start:batch_end]
                batch_texts = [_item_text(it) for it in batch_items]

                print(f"      Encoding batch {batch_start // batch_size + 1}/{(total_items + batch_size - 1) // batch_size} ({batch_end}/{total_items})...")
                batch_emb = embedder.encode(batch_texts)
                batch_sims = _cos_sim_matrix(batch_emb, anchor_emb)
                sims.extend(batch_sims)

            print(f"      Scoring {total_items} items...")

            for item, row, matched_kws in zip(valid_items, sims, vip_matches):
                score = _aggregate_semantic_score(row)
                item.semantic_score = round(score, 4)
                item.is_vip = bool(matched_kws)
                item.vip_keywords = matched_kws

                if matched_kws:
                    # VIP: always keep, score is for reference
                    kept.append(item)
                    vip_kept += 1
                elif score > THRESHOLD_NORMAL:
                    # Non-VIP but high semantic score: keep
                    kept.append(item)
                    semantic_kept += 1
                else:
                    # Non-VIP and low semantic score: drop
                    semantic_dropped.append(item)

    dropped.extend(semantic_dropped)
    print(f"      Layer 2: {len(kept)} kept (VIP={vip_kept}, semantic={semantic_kept}), {len(semantic_dropped)} dropped")

    if record_layer2_dropped_to_seen and seen_records is not None and semantic_dropped:
        try:
            fingerprint_mod = importlib.import_module("src.3_dedup.fingerprint")
            seen_mod = importlib.import_module("src.3_dedup.seen")
            get_fingerprint = getattr(fingerprint_mod, "get_fingerprint")
            mark_batch = getattr(seen_mod, "mark_batch")
            fps = [get_fingerprint(it) for it in semantic_dropped]
            fps = [fp for fp in fps if fp]
            if fps:
                mark_batch(seen_records, fps)
        except Exception:
            # Best-effort: hybrid filtering should still work even if seen module changes.
            pass

    if save and kept:
        _save_filtered(kept)

    stats = HybridFilterStats(
        total=len(items),
        layer1_dropped=len(items) - len(layer1_pass),
        layer2_vip_kept=vip_kept,
        layer2_semantic_kept=semantic_kept,
        layer2_semantic_dropped=len(semantic_dropped),
    )
    return kept, dropped, stats


def _save_filtered(items: list[NewsItem]) -> None:
    """Save filtered results to data/filtered/{date}/all.json, sorted by semantic score."""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = FILTER_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    sorted_items = _sort_by_semantic(items)

    filepath = dir_path / "all.json"
    data = [item.model_dump() for item in sorted_items]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved filtered to {filepath}")


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
        # Some embedder implementations (e.g., with __slots__) may forbid setting attributes.
        pass
    return embeddings


def _aggregate_semantic_score(similarities: list[float]) -> float:
    """
    Aggregate cosine similarities so multiple Anchor hits add a mild boost.

    Singles keep their raw max score; additional anchors above the normal
    threshold contribute diminishing bonuses capped to avoid overpowering
    strong single-hit items.
    """
    if not similarities:
        return 0.0

    ordered = sorted(similarities, reverse=True)
    primary = ordered[0]

    if len(ordered) == 1:
        return primary

    extras = [score for score in ordered[1:] if score >= THRESHOLD_NORMAL]
    bonus = 0.0
    weight = 0.12  # diminishing contribution for each extra anchor
    for score in extras[:3]:
        bonus += max(0.0, score - THRESHOLD_NORMAL) * weight
        weight *= 0.5

    # Cap total boost so single-hit items still rank fairly.
    combined = primary + min(0.18, bonus)
    return min(1.0, combined)


def _sort_by_semantic(items: list[NewsItem]) -> list[NewsItem]:
    """Return items sorted by semantic_score descending; keep original order if no scores."""
    if not items:
        return []

    has_semantic = any(getattr(it, "semantic_score", None) is not None for it in items)
    if not has_semantic:
        return items

    return sorted(
        items,
        key=lambda it: it.semantic_score if it.semantic_score is not None else float("-inf"),
        reverse=True,
    )
