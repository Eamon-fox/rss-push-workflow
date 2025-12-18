"""RSS Parsers - one per source."""

from .nature import NatureParser
from .cell import CellParser
from .science import ScienceParser
from .biorxiv import BiorxivParser, MedrxivParser
from .wechat import WechatParser, NewsParser
from .wiley import WileyParser
from .acs import ACSParser

# 来源名 -> 解析器映射
PARSERS = {
    # ========== Nature 系列 ==========
    "Nature": NatureParser,
    "Nature Neuroscience": NatureParser,
    "Nature Genetics": NatureParser,
    "Nature Cell Biology": NatureParser,
    "Nature Methods": NatureParser,
    "Nature Structural & Molecular Biology": NatureParser,
    "Nature Chemical Biology": NatureParser,
    "Nature Communications": NatureParser,
    "Nature Biotechnology": NatureParser,
    "Nature Medicine": NatureParser,
    "Nature Reviews Molecular Cell Biology": NatureParser,
    "Nature Reviews Genetics": NatureParser,
    "Nature Protocols": NatureParser,
    "Communications Biology": NatureParser,
    "Molecular Biology of the Cell": NatureParser,  # 格式类似Nature

    # ========== Cell 系列 ==========
    "Cell": CellParser,
    "Molecular Cell": CellParser,
    "Cell Reports": CellParser,
    "Neuron": CellParser,
    "Cell Stem Cell": CellParser,
    "Cancer Cell": CellParser,
    "Immunity": CellParser,
    "Developmental Cell": CellParser,
    "Cell Metabolism": CellParser,
    "Cell Chemical Biology": CellParser,
    "Current Biology": CellParser,
    "Structure": CellParser,
    "American Journal of Human Genetics": CellParser,
    "Cell Reports Methods": CellParser,
    "Trends in Cell Biology": CellParser,
    "Trends in Biochemical Sciences": CellParser,

    # ========== Science 系列 ==========
    "Science": ScienceParser,
    "Science AOP": ScienceParser,
    "Science Advances": ScienceParser,
    "Science Signaling": ScienceParser,

    # ========== Wiley (FEBS等) ==========
    "FEBS Journal": WileyParser,
    "FEBS Letters": WileyParser,

    # ========== ACS ==========
    "Biochemistry (ACS)": ACSParser,

    # ========== 预印本 ==========
    "bioRxiv": BiorxivParser,
    "bioRxiv Neuroscience": BiorxivParser,
    "bioRxiv Cell Biology": BiorxivParser,
    "bioRxiv Molecular Biology": BiorxivParser,
    "bioRxiv Biochemistry": BiorxivParser,
    "bioRxiv Genetics": BiorxivParser,
    "bioRxiv Genomics": BiorxivParser,
    "bioRxiv Immunology": BiorxivParser,
    "bioRxiv Cancer Biology": BiorxivParser,
    "bioRxiv Developmental Biology": BiorxivParser,
    "bioRxiv Systems Biology": BiorxivParser,
    "bioRxiv Bioinformatics": BiorxivParser,
    "medRxiv": MedrxivParser,

    # ========== 公众号 ==========
    "BioArt": WechatParser,
    "科研圈": WechatParser,
    "生物探索": WechatParser,
    "知识分子": WechatParser,
    "生物谷": WechatParser,

    # ========== 新闻 ==========
    "科学网": NewsParser,
    "果壳": NewsParser,
    "Phys.org": NewsParser,
    "Phys.org Biology": NewsParser,
    "Science Daily": NewsParser,
    "EurekAlert Life Sciences": NewsParser,
    "Quanta Magazine Biology": NewsParser,
}


def get_parser(source_name: str):
    """Get parser class for source."""
    return PARSERS.get(source_name)
