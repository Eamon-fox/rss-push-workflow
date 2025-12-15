"""Step 4 config - hybrid filtering keywords and anchors.

Keep these lists small and high-signal:
- VIP_KEYWORDS: regex patterns for "must keep" terms.
- SEMANTIC_ANCHORS: natural-language anchors (titles/abstracts) for semantic recall.
- GENERAL_BIO_KEYWORDS: coarse biology keyword/stem list for cheap filtering.
"""

# 1) VIP 本命词 (Regex Pattern): 绝对不漏掉的核心词
VIP_KEYWORDS: list[str] = [
    r"\bRTCB\b",
    r"tRNA ligase",
    r"\bIRE1\b",
    r"\bXBP1\b",
    r"\bUPR\b",
]

# 2) 语义锚点 (Natural Language): 用于捞回相关但未提关键词的文章
# 建议直接复制经典论文的 Title + Abstract
SEMANTIC_ANCHORS: list[str] = [
    "Mechanism of RTCB-mediated RNA ligation in neurons under stress conditions.",
    "The IRE1-XBP1 signaling pathway and unfolded protein response (UPR) during ER stress.",
    "Neurodegeneration caused by defects in tRNA metabolism and RNA ligation pathways.",
]

# 3) 生物学通用词 (Keyword/Stems): 用于过滤物理/天文/地质/招聘等文章
# NOTE: These are intentionally stem-like to catch morphology (e.g., epigenet -> epigenetics).
GENERAL_BIO_KEYWORDS: list[str] = [
    # High specificity - molecular biology
    "rna",
    "dna",
    "protein",
    "peptide",
    "nucleotide",
    "chromatin",
    "histone",
    "epigenet",
    "proteom",
    "kinase",
    "phosphatas",
    "phosphoryl",
    "proteas",
    "ligase",
    "hydrolase",
    "polymeras",
    # High specificity - cell types
    "neuron",
    "glia",
    "astrocyt",
    "microglia",
    "neutrophil",
    "macrophag",
    "lymphocyt",
    "fibroblast",
    "oocyt",
    "sperm",
    # High specificity - organelles
    "mitochond",
    "ribosom",
    "lysosom",
    "chloroplast",
    # High specificity - neurotransmitters
    "dopamin",
    "serotonin",
    "glutamat",
    # High specificity - disease
    "cancer",
    "tumor",
    "carcinom",
    "leukem",
    "lymphom",
    "alzheim",
    "parkinson",
    "diabet",
    # High specificity - immunology
    "antibod",
    "antigen",
    "cytokin",
    "antibiotic",
    "vaccin",
    # High specificity - pathogens
    "virus",
    "viral",
    "bacteri",
    "pathogen",
    "fungal",
    "fungi",
    "toxoplasm",
    "parasit",
    # High specificity - techniques
    "cryo",
    "crystallograph",
    "crispr",
    "sequenc",
    # Medium specificity - genetics
    "gene",
    "genom",
    "genetic",
    "mutat",
    "transcript",
    # Medium specificity - neuroscience
    "brain",
    "neural",
    "cortex",
    "hippocam",
    "synap",
    "cognitiv",
    # Medium specificity - cell/tissue
    "cell",
    "tissue",
    # Medium specificity - immunity
    "immun",
    "inflamm",
    "infect",
    # Medium specificity - development/physiology
    "embryo",
    "ovar",
    "reproduct",
    "fertiliz",
    "mammal",
    "vertebrat",
    "hormon",
    "aging",
    "ageing",
    "lifespan",
    # Medium specificity - microbiology
    "microb",
    "microorganism",
    "mosquit",
    # Medium specificity - plant biology
    "chlorophyll",
    "photosyn",
    "pollinat",
    "plant",
    "xylem",
    "phloem",
    "seed",
    "dormancy",
    "germination",
    # Medium specificity - biochemistry
    "biosynthes",
    "metabol",
    "receptor",
    "ligand",
    "enzym",
    # Medium specificity - physiology
    "regulat",
    "signal",
    "pathway",
    # Compound terms (high specificity)
    "stem cell",
    "t cell",
    "b cell",
    "cell line",
    "cell type",
    "cell death",
]

TOP_JOURNALS: set[str] = {"Cell", "Nature", "Science"}
BLACKLIST_TITLE_KEYWORDS: tuple[str, ...] = ("Correction", "Retraction", "Erratum")

# Layer 2 dynamic thresholds
THRESHOLD_NORMAL: float = 0.35
THRESHOLD_TOP_JOURNAL: float = 0.15

