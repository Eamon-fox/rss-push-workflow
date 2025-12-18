"""Step 6: Output delivery."""

from .console import to_console, to_json, to_markdown
from .html import to_html, to_html_file

__all__ = ["to_console", "to_json", "to_markdown", "to_html", "to_html_file"]
