"""Step 4 config - load from config/filter.yaml and config/semantic_anchors.yaml.

All filtering parameters are externalized to config files:
- config/filter.yaml: thresholds, VIP keywords (tiered), bio keywords, scoring params
- config/semantic_anchors.yaml: semantic anchor texts (tiered)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Dict, List

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
FILTER_CONFIG_PATH = CONFIG_DIR / "filter.yaml"
ANCHORS_CONFIG_PATH = CONFIG_DIR / "semantic_anchors.yaml"
# Fallback to JSON if YAML doesn't exist
ANCHORS_JSON_PATH = CONFIG_DIR / "semantic_anchors.json"


@dataclass
class TieredAnchors:
    """Tiered semantic anchors (positive and negative)."""
    tier1: List[str] = field(default_factory=list)
    tier2: List[str] = field(default_factory=list)
    tier3: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)  # 负向锚点

    def all_anchors(self) -> List[str]:
        """Return all positive anchors as flat list."""
        return self.tier1 + self.tier2 + self.tier3

    def get_tier(self, anchor_index: int) -> str:
        """Get tier name for anchor at given index in all_anchors()."""
        if anchor_index < len(self.tier1):
            return "tier1"
        elif anchor_index < len(self.tier1) + len(self.tier2):
            return "tier2"
        else:
            return "tier3"


@dataclass
class TieredVipKeywords:
    """Tiered VIP keywords with multipliers."""
    tier1_patterns: List[str] = field(default_factory=list)
    tier1_multiplier: float = 1.50
    tier2_patterns: List[str] = field(default_factory=list)
    tier2_multiplier: float = 1.30
    tier3_patterns: List[str] = field(default_factory=list)
    tier3_multiplier: float = 1.15
    max_multiplier: float = 1.80
    stack_bonus: float = 0.10


@dataclass
class ScoringConfig:
    """Scoring configuration."""
    final_threshold: float = 0.50

    # Anchor tier multipliers
    anchor_tier_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "tier1": 1.20,
        "tier2": 1.10,
        "tier3": 1.00,
    })

    # Anchor coverage bonus
    anchor_coverage_threshold: float = 0.40
    anchor_coverage_bonus: Dict[str, float] = field(default_factory=lambda: {
        "tier1": 0.12,
        "tier2": 0.06,
        "tier3": 0.03,
    })

    # Negative anchor settings
    negative_threshold: float = 0.45  # 负向匹配阈值
    negative_penalty: float = 0.60    # 惩罚乘数 (0.6 = 降低40%)


def _load_filter_config() -> dict:
    """Load filter configuration from YAML."""
    if not FILTER_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Filter config not found: {FILTER_CONFIG_PATH}")
    with open(FILTER_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_tiered_anchors() -> TieredAnchors:
    """Load tiered semantic anchors from YAML."""
    # Try YAML first
    if ANCHORS_CONFIG_PATH.exists():
        with open(ANCHORS_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        def clean_anchors(anchors: list) -> list:
            return [str(a).strip() for a in (anchors or []) if str(a).strip()]

        return TieredAnchors(
            tier1=clean_anchors(data.get("tier1", [])),
            tier2=clean_anchors(data.get("tier2", [])),
            tier3=clean_anchors(data.get("tier3", [])),
            negative=clean_anchors(data.get("negative", [])),
        )

    # Fallback to JSON (all in tier1 for backwards compatibility)
    if ANCHORS_JSON_PATH.exists():
        import json
        with open(ANCHORS_JSON_PATH, "r", encoding="utf-8") as f:
            anchors = json.load(f)
        if isinstance(anchors, list):
            return TieredAnchors(
                tier1=[str(a).strip() for a in anchors if str(a).strip()]
            )

    return TieredAnchors()


def _load_tiered_vip_keywords(config: dict) -> TieredVipKeywords:
    """Load tiered VIP keywords from config."""
    vip_config = config.get("vip_keywords", {})

    # Handle new tiered format
    if isinstance(vip_config, dict) and "tier1" in vip_config:
        tier1 = vip_config.get("tier1", {})
        tier2 = vip_config.get("tier2", {})
        tier3 = vip_config.get("tier3", {})

        return TieredVipKeywords(
            tier1_patterns=tier1.get("patterns", []),
            tier1_multiplier=tier1.get("multiplier", 1.50),
            tier2_patterns=tier2.get("patterns", []),
            tier2_multiplier=tier2.get("multiplier", 1.30),
            tier3_patterns=tier3.get("patterns", []),
            tier3_multiplier=tier3.get("multiplier", 1.15),
            max_multiplier=config.get("vip_max_multiplier", 1.80),
            stack_bonus=config.get("vip_stack_bonus", 0.10),
        )

    # Handle old flat list format (backwards compatibility - all in tier1)
    if isinstance(vip_config, list):
        return TieredVipKeywords(
            tier1_patterns=vip_config,
            tier1_multiplier=1.50,
        )

    return TieredVipKeywords()


def _load_scoring_config(config: dict) -> ScoringConfig:
    """Load scoring configuration."""
    anchor_tier_mult = config.get("anchor_tier_multiplier", {})
    anchor_coverage = config.get("anchor_coverage", {})
    negative_anchor = config.get("negative_anchor", {})

    return ScoringConfig(
        final_threshold=config.get("final_threshold", 0.50),
        anchor_tier_multiplier={
            "tier1": anchor_tier_mult.get("tier1", 1.20),
            "tier2": anchor_tier_mult.get("tier2", 1.10),
            "tier3": anchor_tier_mult.get("tier3", 1.00),
        },
        anchor_coverage_threshold=anchor_coverage.get("threshold", 0.40),
        anchor_coverage_bonus={
            "tier1": anchor_coverage.get("bonus", {}).get("tier1", 0.12),
            "tier2": anchor_coverage.get("bonus", {}).get("tier2", 0.06),
            "tier3": anchor_coverage.get("bonus", {}).get("tier3", 0.03),
        },
        negative_threshold=negative_anchor.get("threshold", 0.45),
        negative_penalty=negative_anchor.get("penalty", 0.60),
    )


# Load config at module import
_config = _load_filter_config()

# Exported config values
TIERED_ANCHORS: TieredAnchors = _load_tiered_anchors()
TIERED_VIP_KEYWORDS: TieredVipKeywords = _load_tiered_vip_keywords(_config)
SCORING_CONFIG: ScoringConfig = _load_scoring_config(_config)
GENERAL_BIO_KEYWORDS: List[str] = _config.get("bio_keywords", [])

# Legacy exports for backwards compatibility
SEMANTIC_ANCHORS: List[str] = TIERED_ANCHORS.all_anchors()
VIP_KEYWORDS: List[str] = (
    TIERED_VIP_KEYWORDS.tier1_patterns +
    TIERED_VIP_KEYWORDS.tier2_patterns +
    TIERED_VIP_KEYWORDS.tier3_patterns
)
THRESHOLD_NORMAL: float = SCORING_CONFIG.final_threshold
