"""RSS Parsers - one per source."""

from .nature import NatureParser
from .cell import CellParser
from .science import ScienceParser

# 来源名 -> 解析器映射
PARSERS = {
    "Nature": NatureParser,
    "Nature Neuroscience": NatureParser,
    "Cell": CellParser,
    "Science": ScienceParser,
    "Science AOP": ScienceParser,
}


def get_parser(source_name: str):
    """Get parser class for source."""
    return PARSERS.get(source_name)
