"""Hybrid filter (Layer1 regex + Layer2 VIP + semantic vector fallback)."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from ..models import NewsItem
from .config import (
    BLACKLIST_TITLE_KEYWORDS,
    GENERAL_BIO_KEYWORDS,
    SEMANTIC_ANCHORS,
    THRESHOLD_NORMAL,
    THRESHOLD_TOP_JOURNAL,
    TOP_JOURNALS,
    VIP_KEYWORDS,
)

FILTER_DIR = Path("data/filtered")


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> object:
        """Return L2-normalized embeddings for given texts (numpy array or list)."""


def _default_embedder() -> Embedder:
    """
    Lazily import and construct the fastest local model by default.
    Model download (~80MB) happens on first run via sentence-transformers cache.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: sentence-transformers. "
            "Install with: pip install sentence-transformers torch"
        ) from e

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    class _STEmbedder:
        def encode(self, texts: list[str]) -> object:
            emb = model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return emb

    return _STEmbedder()


def _compile_union_regex(terms: list[str], *, escape: bool) -> re.Pattern[str]:
    parts = [re.escape(t) for t in terms] if escape else list(terms)
    return re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE)


_BLACKLIST_RE = _compile_union_regex(list(BLACKLIST_TITLE_KEYWORDS), escape=True)
_GENERAL_BIO_RE = _compile_union_regex(GENERAL_BIO_KEYWORDS, escape=True)
_VIP_RE = _compile_union_regex(VIP_KEYWORDS, escape=False)


def _is_top_journal(source_name: str) -> bool:
    s = (source_name or "").strip()
    if s in TOP_JOURNALS:
        return True
    # Common variants: "Science AOP", "Nature Neuroscience"
    return any(s.startswith(j) for j in TOP_JOURNALS)


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

    embedder = embedder or _default_embedder()

    layer1_pass: list[NewsItem] = []
    dropped: list[NewsItem] = []

    # Layer 1: blacklist + coarse bio filter (top journals bypass coarse filter)
    for item in items:
        title = item.title or ""
        if _BLACKLIST_RE.search(title):
            dropped.append(item)
            continue

        if _is_top_journal(item.source_name):
            layer1_pass.append(item)
            continue

        text = _item_text(item)
        # VIP terms should never be lost at Layer 1.
        if _VIP_RE.search(text) or _GENERAL_BIO_RE.search(text):
            layer1_pass.append(item)
        else:
            dropped.append(item)

    # Layer 2: VIP + semantic fallback
    kept: list[NewsItem] = []
    vip_kept = 0
    semantic_kept = 0
    semantic_dropped: list[NewsItem] = []
    to_vector: list[NewsItem] = []

    for item in layer1_pass:
        text = _item_text(item)
        if not text:
            semantic_dropped.append(item)
            continue
        if _VIP_RE.search(text):
            kept.append(item)
            vip_kept += 1
            continue
        to_vector.append(item)

    if to_vector:
        anchors = [a.strip() for a in SEMANTIC_ANCHORS if a and a.strip()]
        if not anchors:
            semantic_dropped.extend(to_vector)
        else:
            anchor_emb = embedder.encode(anchors)
            item_texts = [_item_text(it) for it in to_vector]
            item_emb = embedder.encode(item_texts)
            sims = _cos_sim_matrix(item_emb, anchor_emb)

            for item, row in zip(to_vector, sims):
                score = max(row) if row else 0.0
                threshold = THRESHOLD_TOP_JOURNAL if _is_top_journal(item.source_name) else THRESHOLD_NORMAL
                if score > threshold:
                    kept.append(item)
                    semantic_kept += 1
                else:
                    semantic_dropped.append(item)

    dropped.extend(semantic_dropped)

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
    """Save filtered results to data/filtered/{date}/all.json"""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = FILTER_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    filepath = dir_path / "all.json"
    data = [item.model_dump() for item in items]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved filtered to {filepath}")
