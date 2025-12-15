"""Bio keyword filter - requires at least one biology-related term."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import NewsItem

FILTER_DIR = Path("data/filtered")

# Word stems for morphological matching (e.g., "protein" matches "proteins")
BIO_KEYWORDS = [
    # High specificity - molecular biology
    "rna", "dna", "protein", "peptide", "nucleotide",
    "chromatin", "histone", "epigenet", "proteom",
    "kinase", "phosphatas", "phosphoryl", "proteas", "ligase", "hydrolase", "polymeras",

    # High specificity - cell types
    "neuron", "glia", "astrocyt", "microglia",
    "neutrophil", "macrophag", "lymphocyt", "fibroblast",
    "oocyt", "sperm",

    # High specificity - organelles
    "mitochond", "ribosom", "lysosom", "chloroplast",

    # High specificity - neurotransmitters
    "dopamin", "serotonin", "glutamat",

    # High specificity - disease
    "cancer", "tumor", "carcinom", "leukem", "lymphom",
    "alzheim", "parkinson", "diabet",

    # High specificity - immunology
    "antibod", "antigen", "cytokin", "antibiotic", "vaccin",

    # High specificity - pathogens
    "virus", "viral", "bacteri", "pathogen", "fungal", "fungi",
    "toxoplasm", "parasit",

    # High specificity - techniques
    "cryo", "crystallograph", "crispr", "sequenc",

    # Medium specificity - genetics
    "gene", "genom", "genetic", "mutat", "transcript",

    # Medium specificity - neuroscience
    "brain", "neural", "cortex", "hippocam", "synap", "cognitiv",

    # Medium specificity - cell/tissue
    "cell", "tissue",

    # Medium specificity - immunity
    "immun", "inflamm", "infect",

    # Medium specificity - development/physiology
    "embryo", "ovar", "reproduct", "fertiliz",
    "mammal", "vertebrat", "hormon", "aging", "ageing", "lifespan",

    # Medium specificity - microbiology
    "microb", "microorganism", "mosquit",

    # Medium specificity - plant biology
    "chlorophyll", "photosyn", "pollinat", "plant", "xylem", "phloem",
    "seed", "dormancy", "germination",

    # Medium specificity - biochemistry
    "biosynthes", "metabol", "receptor", "ligand", "enzym",

    # Medium specificity - physiology
    "regulat", "signal", "pathway",

    # Compound terms (high specificity)
    "stem cell", "t cell", "b cell", "cell line", "cell type", "cell death",
]


def has_bio_keyword(title: str, content: str) -> tuple[bool, list[str]]:
    """
    Check if title or content contains any bio keyword.

    Returns:
        Tuple of (has_keyword, matched_keywords)
    """
    text = f"{title} {content}".lower()
    matched = []
    for kw in BIO_KEYWORDS:
        if kw in text:
            matched.append(kw)
    return len(matched) > 0, matched


def filter_bio(
    items: "list[NewsItem]",
    save: bool = True,
) -> tuple["list[NewsItem]", "list[NewsItem]"]:
    """
    Filter items by bio keyword presence.

    Args:
        items: Items to filter
        save: Whether to save results to disk

    Returns:
        Tuple of (passed_items, filtered_items)
    """
    passed = []
    filtered = []

    for item in items:
        has_kw, _ = has_bio_keyword(item.title, item.content)
        if has_kw:
            passed.append(item)
        else:
            filtered.append(item)

    if save and passed:
        _save_filtered(passed)

    return passed, filtered


def _save_filtered(items: "list[NewsItem]") -> None:
    """Save filtered results to data/filtered/{date}/all.json"""
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = FILTER_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    filepath = dir_path / "all.json"
    data = [item.model_dump() for item in items]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved filtered to {filepath}")
