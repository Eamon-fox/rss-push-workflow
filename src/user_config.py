"""User configuration module for personalization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.core import load_json, save_json

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("data/user_configs")
MAX_ANCHOR_LENGTH = 3000  # 单条 anchor 最大字符数
MAX_USER_ANCHORS = 50  # 用户自定义 anchor 最大数量（不含系统默认）


class VIPKeywords(BaseModel):
    """VIP keywords configuration by tier."""

    tier1: dict = Field(
        default_factory=lambda: {"multiplier": 1.50, "patterns": []},
        description="Tier 1 VIP keywords (highest priority)",
    )
    tier2: dict = Field(
        default_factory=lambda: {"multiplier": 1.30, "patterns": []},
        description="Tier 2 VIP keywords",
    )
    tier3: dict = Field(
        default_factory=lambda: {"multiplier": 1.15, "patterns": []},
        description="Tier 3 VIP keywords",
    )


class SemanticAnchors(BaseModel):
    """Semantic anchors for personalized scoring (tiered)."""

    tier1: list[str] = Field(
        default_factory=list,
        description="Tier 1 anchors - core research interests (weight: 0.50)",
    )
    tier2: list[str] = Field(
        default_factory=list,
        description="Tier 2 anchors - closely related topics (weight: 0.35)",
    )
    tier3: list[str] = Field(
        default_factory=list,
        description="Tier 3 anchors - extended interests (weight: 0.15)",
    )
    negative: list[str] = Field(
        default_factory=list,
        description="Negative anchor texts (penalize matching articles)",
    )

    def all_positive(self) -> list[str]:
        """Return all positive anchors as flat list (tier1 + tier2 + tier3)."""
        return self.tier1 + self.tier2 + self.tier3


class ScoringParams(BaseModel):
    """Scoring parameters for personalization (tiered scoring)."""

    # Tier weights (should sum to 1.0)
    tier_weights: dict = Field(
        default_factory=lambda: {"tier1": 0.50, "tier2": 0.35, "tier3": 0.15},
        description="Weight for each tier in weighted scoring",
    )

    # Tier thresholds (minimum similarity to count)
    tier_thresholds: dict = Field(
        default_factory=lambda: {"tier1": 0.30, "tier2": 0.35, "tier3": 0.40},
        description="Minimum similarity threshold for each tier",
    )

    # Aggregation method
    aggregation: str = Field(
        default="max",
        description="Aggregation method: 'max', 'top_k', or 'mean'",
    )

    # Coverage bonus
    coverage_threshold: float = Field(
        default=0.40,
        description="Similarity threshold to count as 'covered' for bonus",
    )
    coverage_bonus: dict = Field(
        default_factory=lambda: {"tier1": 0.10, "tier2": 0.06, "tier3": 0.03},
        description="Bonus multiplier for each tier when covered",
    )

    # VIP settings
    vip_max_multiplier: float = Field(
        default=1.80,
        description="Maximum VIP multiplier",
    )

    # Negative anchor settings
    negative_threshold: float = Field(
        default=0.38,
        description="Similarity threshold to trigger negative penalty",
    )
    negative_penalty: float = Field(
        default=0.60,
        description="Penalty multiplier when negative threshold is exceeded",
    )


class UserConfig(BaseModel):
    """Complete user configuration for personalization."""

    openid: str = Field(..., description="User's WeChat OpenID")
    vip_keywords: VIPKeywords = Field(default_factory=VIPKeywords)
    semantic_anchors: SemanticAnchors = Field(default_factory=SemanticAnchors)
    scoring_params: ScoringParams = Field(default_factory=ScoringParams)

    # Cache for anchor embeddings (not persisted)
    _anchor_embeddings_cache: Optional[dict] = None


def _get_config_path(openid: str) -> Path:
    """Get config file path for a user."""
    # Sanitize openid for use as filename
    safe_id = openid.replace("/", "_").replace("\\", "_")
    return CONFIG_DIR / f"{safe_id}.json"


def load_user_config(openid: str) -> Optional[UserConfig]:
    """
    Load user configuration from file.

    Handles migration from old format (positive/negative) to new format (tier1/tier2/tier3/negative).

    Args:
        openid: User's OpenID

    Returns:
        UserConfig if exists, None otherwise
    """
    config_path = _get_config_path(openid)

    if not config_path.exists():
        return None

    try:
        data = load_json(config_path, default=None)
        if data is None:
            return None

        # Migrate old format: positive -> tier2
        anchors_data = data.get("semantic_anchors", {})
        if "positive" in anchors_data and "tier1" not in anchors_data:
            logger.info(f"Migrating old anchor format for {openid}")
            old_positive = anchors_data.pop("positive", [])
            anchors_data["tier1"] = []
            anchors_data["tier2"] = old_positive  # Old positive -> tier2 (medium priority)
            anchors_data["tier3"] = []
            data["semantic_anchors"] = anchors_data

        return UserConfig(**data)
    except Exception as e:
        logger.error(f"Failed to load user config for {openid}: {e}")
        return None


def _sanitize_anchors(anchors: list[str], max_count: int = MAX_USER_ANCHORS) -> tuple[list[str], dict]:
    """
    Sanitize anchors: dedupe, truncate length, limit count.

    Returns:
        (sanitized_list, hints_dict)
    """
    hints = {
        "truncated": 0,  # 被截断的数量
        "deduped": 0,    # 去重删除的数量
        "dropped": 0,    # 超出数量限制被丢弃的数量
    }

    # 1. 去重 (保持顺序，保留首次出现)
    seen = set()
    deduped = []
    for anchor in anchors:
        anchor_stripped = anchor.strip()
        if anchor_stripped and anchor_stripped not in seen:
            seen.add(anchor_stripped)
            deduped.append(anchor_stripped)
        elif anchor_stripped in seen:
            hints["deduped"] += 1

    # 2. 截断过长的
    result = []
    for anchor in deduped:
        if len(anchor) > MAX_ANCHOR_LENGTH:
            anchor = anchor[:MAX_ANCHOR_LENGTH]
            hints["truncated"] += 1
            logger.warning(f"Truncated anchor to {MAX_ANCHOR_LENGTH} chars")
        result.append(anchor)

    # 3. 限制数量 (保留最新的，即列表末尾的)
    if len(result) > max_count:
        hints["dropped"] = len(result) - max_count
        result = result[-max_count:]  # 保留最新的
        logger.warning(f"Dropped {hints['dropped']} anchors due to limit")

    return result, hints


def save_user_config(config: UserConfig) -> dict:
    """
    Save user configuration to file.

    Args:
        config: UserConfig to save

    Returns:
        hints dict with sanitization info for frontend
    """
    all_hints = {"tier1": {}, "tier2": {}, "tier3": {}, "negative": {}}

    # Sanitize anchors before saving (each tier separately)
    config.semantic_anchors.tier1, all_hints["tier1"] = _sanitize_anchors(
        config.semantic_anchors.tier1
    )
    config.semantic_anchors.tier2, all_hints["tier2"] = _sanitize_anchors(
        config.semantic_anchors.tier2
    )
    config.semantic_anchors.tier3, all_hints["tier3"] = _sanitize_anchors(
        config.semantic_anchors.tier3
    )
    config.semantic_anchors.negative, all_hints["negative"] = _sanitize_anchors(
        config.semantic_anchors.negative
    )

    config_path = _get_config_path(config.openid)
    save_json(config.model_dump(), config_path)
    logger.info(f"Saved user config for {config.openid}")

    return all_hints


def delete_user_config(openid: str) -> bool:
    """
    Delete user configuration.

    Args:
        openid: User's OpenID

    Returns:
        True if deleted, False if not found
    """
    config_path = _get_config_path(openid)

    if not config_path.exists():
        return False

    try:
        config_path.unlink()
        logger.info(f"Deleted user config for {openid}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete user config for {openid}: {e}")
        return False


def get_or_create_config(openid: str) -> UserConfig:
    """
    Get user config, creating with system defaults if not exists.

    Args:
        openid: User's OpenID

    Returns:
        UserConfig (existing or newly created with system defaults)
    """
    config = load_user_config(openid)

    if config is not None:
        return config

    # Create config with system defaults
    system_config = _load_system_config()
    anchors = system_config["semantic_anchors"]
    config = UserConfig(
        openid=openid,
        vip_keywords=VIPKeywords(
            tier1=system_config["vip_keywords"].get("tier1", {}),
            tier2=system_config["vip_keywords"].get("tier2", {}),
            tier3=system_config["vip_keywords"].get("tier3", {}),
        ),
        semantic_anchors=SemanticAnchors(
            tier1=anchors.get("tier1", []),
            tier2=anchors.get("tier2", []),
            tier3=anchors.get("tier3", []),
            negative=anchors.get("negative", []),
        ),
        scoring_params=ScoringParams(**system_config.get("scoring_params", {})),
    )
    save_user_config(config)
    logger.info(f"Created new user config with system defaults for {openid}")
    return config


def update_vip_keywords(openid: str, vip_keywords: VIPKeywords) -> UserConfig:
    """
    Update user's VIP keywords.

    Args:
        openid: User's OpenID
        vip_keywords: New VIP keywords configuration

    Returns:
        Updated UserConfig
    """
    config = get_or_create_config(openid)
    config.vip_keywords = vip_keywords
    save_user_config(config)
    return config


def update_semantic_anchors(openid: str, anchors: SemanticAnchors) -> UserConfig:
    """
    Update user's semantic anchors.

    Args:
        openid: User's OpenID
        anchors: New semantic anchors configuration

    Returns:
        Updated UserConfig
    """
    config = get_or_create_config(openid)
    config.semantic_anchors = anchors
    save_user_config(config)
    return config


def update_scoring_params(openid: str, params: ScoringParams) -> UserConfig:
    """
    Update user's scoring parameters.

    Args:
        openid: User's OpenID
        params: New scoring parameters

    Returns:
        Updated UserConfig
    """
    config = get_or_create_config(openid)
    config.scoring_params = params
    save_user_config(config)
    return config


def list_user_configs() -> list[str]:
    """
    List all users with configurations.

    Returns:
        List of OpenIDs
    """
    if not CONFIG_DIR.exists():
        return []

    openids = []
    for config_file in CONFIG_DIR.glob("*.json"):
        openids.append(config_file.stem)

    return openids


def _load_system_config() -> dict:
    """
    Load system-level config from YAML files.

    Returns:
        Dict with system defaults for vip_keywords, semantic_anchors, and scoring_params
    """
    import yaml

    result = {
        "vip_keywords": {"tier1": {}, "tier2": {}, "tier3": {}},
        "semantic_anchors": {"tier1": [], "tier2": [], "tier3": [], "negative": []},
        "scoring_params": {},
    }

    # Load from filter.yaml
    filter_path = Path("config/filter.yaml")
    if filter_path.exists():
        try:
            with open(filter_path, "r", encoding="utf-8") as f:
                filter_config = yaml.safe_load(f)

            # VIP keywords
            vip = filter_config.get("vip_keywords", {})
            for tier in ["tier1", "tier2", "tier3"]:
                if tier in vip:
                    result["vip_keywords"][tier] = {
                        "multiplier": vip[tier].get("multiplier", 1.0),
                        "patterns": vip[tier].get("patterns", []),
                    }

            # Scoring params from filter.yaml
            tiered_scoring = filter_config.get("tiered_scoring", {})
            if tiered_scoring:
                result["scoring_params"]["tier_weights"] = tiered_scoring.get("tier_weights", {})
                result["scoring_params"]["tier_thresholds"] = tiered_scoring.get("tier_thresholds", {})
                result["scoring_params"]["aggregation"] = tiered_scoring.get("aggregation", "max")

            anchor_coverage = filter_config.get("anchor_coverage", {})
            if anchor_coverage:
                result["scoring_params"]["coverage_threshold"] = anchor_coverage.get("threshold", 0.40)
                result["scoring_params"]["coverage_bonus"] = anchor_coverage.get("bonus", {})

            negative_anchor = filter_config.get("negative_anchor", {})
            if negative_anchor:
                result["scoring_params"]["negative_threshold"] = negative_anchor.get("threshold", 0.38)
                result["scoring_params"]["negative_penalty"] = negative_anchor.get("penalty", 0.60)

            result["scoring_params"]["vip_max_multiplier"] = filter_config.get("vip_max_multiplier", 1.80)

        except Exception as e:
            logger.warning(f"Failed to load filter.yaml: {e}")

    # Load semantic anchors from semantic_anchors.yaml (tiered format)
    anchors_path = Path("config/semantic_anchors.yaml")
    if anchors_path.exists():
        try:
            with open(anchors_path, "r", encoding="utf-8") as f:
                anchors_config = yaml.safe_load(f)
            # Keep tiered structure
            result["semantic_anchors"]["tier1"] = anchors_config.get("tier1", [])
            result["semantic_anchors"]["tier2"] = anchors_config.get("tier2", [])
            result["semantic_anchors"]["tier3"] = anchors_config.get("tier3", [])
            result["semantic_anchors"]["negative"] = anchors_config.get("negative", [])
        except Exception as e:
            logger.warning(f"Failed to load semantic_anchors.yaml: {e}")

    return result


def get_default_config() -> dict:
    """
    Get default configuration values (from system YAML files).

    Returns:
        Dict with system default values
    """
    system_config = _load_system_config()
    # Merge with ScoringParams defaults
    default_scoring = ScoringParams()
    merged_scoring = default_scoring.model_dump()
    merged_scoring.update(system_config.get("scoring_params", {}))

    return {
        "vip_keywords": system_config["vip_keywords"],
        "semantic_anchors": system_config["semantic_anchors"],
        "scoring_params": merged_scoring,
    }


def get_default_user_config() -> UserConfig:
    """
    Get a UserConfig instance with default values.

    Returns:
        UserConfig with system defaults (openid="__default__")
    """
    defaults = get_default_config()
    return UserConfig(
        openid="__default__",
        vip_keywords=VIPKeywords(**{
            k: v for k, v in defaults["vip_keywords"].items()
        }),
        semantic_anchors=SemanticAnchors(**defaults["semantic_anchors"]),
        scoring_params=ScoringParams(**defaults["scoring_params"]),
    )
