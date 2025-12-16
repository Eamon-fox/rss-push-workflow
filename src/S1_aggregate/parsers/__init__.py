"""RSS Parsers - one per source."""

from .nature import NatureParser
from .cell import CellParser
from .science import ScienceParser
from .biorxiv import BiorxivParser, MedrxivParser
from .wechat import WechatParser, NewsParser

# 来源名 -> 解析器映射
PARSERS = {
    # 期刊
    "Nature": NatureParser,
    "Nature Neuroscience": NatureParser,
    "Nature Genetics": NatureParser,
    "Nature Cell Biology": NatureParser,
    "Nature Communications": NatureParser,
    "Cell": CellParser,
    "Science": ScienceParser,
    "Science AOP": ScienceParser,
    # 预印本
    "bioRxiv": BiorxivParser,
    "bioRxiv Neuroscience": BiorxivParser,
    "bioRxiv Cell Biology": BiorxivParser,
    "bioRxiv Molecular Biology": BiorxivParser,
    "medRxiv": MedrxivParser,
    # 公众号
    "BioArt": WechatParser,
    "科研圈": WechatParser,
    "生物探索": WechatParser,
    "知识分子": WechatParser,
    "生物谷": WechatParser,
    # 新闻
    "科学网": NewsParser,
    "果壳": NewsParser,
    "Phys.org": NewsParser,
    "Science Daily": NewsParser,
}


def get_parser(source_name: str):
    """Get parser class for source."""
    return PARSERS.get(source_name)
