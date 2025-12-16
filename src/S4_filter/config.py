"""Step 4 config - load from config/filter.yaml and config/semantic_anchors.json.

All filtering parameters are externalized to config files:
- config/filter.yaml: thresholds, VIP keywords, bio keywords
- config/semantic_anchors.json: semantic anchor texts
"""

from __future__ import annotations

import json
from pathlib import Path
import os

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
FILTER_CONFIG_PATH = CONFIG_DIR / "filter.yaml"
ANCHORS_CONFIG_PATH = CONFIG_DIR / "semantic_anchors.json"


def _load_filter_config() -> dict:
    """Load filter configuration from YAML."""
    if not FILTER_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Filter config not found: {FILTER_CONFIG_PATH}")
    with open(FILTER_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_anchors() -> list[str]:
    """Load semantic anchors from JSON."""
    if not ANCHORS_CONFIG_PATH.exists():
        return []
    with open(ANCHORS_CONFIG_PATH, "r", encoding="utf-8") as f:
        anchors = json.load(f)
    if not isinstance(anchors, list):
        raise ValueError("semantic_anchors.json must be a list of strings")
    return [str(a).strip() for a in anchors if str(a).strip()]


# Load config at module import
_config = _load_filter_config()

# Exported config values
_env_threshold = os.environ.get("HYBRID_SEMANTIC_THRESHOLD")
try:
    THRESHOLD_NORMAL: float = float(_env_threshold) if _env_threshold else _config.get("semantic_threshold", 0.35)
except ValueError:
    THRESHOLD_NORMAL = _config.get("semantic_threshold", 0.35)
VIP_KEYWORDS: list[str] = _config.get("vip_keywords", [])
GENERAL_BIO_KEYWORDS: list[str] = _config.get("bio_keywords", [])
SEMANTIC_ANCHORS: list[str] = _load_anchors()
