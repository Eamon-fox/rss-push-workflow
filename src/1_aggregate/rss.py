"""RSS feed fetcher - universal for all RSS sources."""

import httpx
import feedparser
from ..models import NewsItem

# 伪装浏览器，避免被屏蔽
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def fetch(source: dict) -> list[NewsItem]:
    """
    Fetch from RSS source.

    Args:
        source: {"name": "...", "url": "..."}

    Returns:
        List of NewsItem
    """
    url = source["url"]
    name = source["name"]

    # 用 httpx 带 headers 请求，避免 403
    try:
        resp = httpx.get(url, headers=HEADERS, follow_redirects=True, timeout=30)
        if resp.status_code != 200:
            print(f"  [{name}] HTTP {resp.status_code}")
            return []
        content = resp.text
    except Exception as e:
        print(f"  [{name}] Error: {e}")
        return []

    feed = feedparser.parse(content)

    items = []
    for entry in feed.entries:
        items.append(NewsItem(
            title=entry.get("title", ""),
            content=entry.get("summary", entry.get("description", "")),
            link=entry.get("link", ""),
            source_name=name,
            source_url=url,
        ))

    return items
