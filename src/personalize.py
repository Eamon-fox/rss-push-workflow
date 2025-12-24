"""Personalized ranking engine for user-specific article scoring.

Simplified version that delegates to hybrid.score_items() for unified scoring logic.
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import NewsItem
from .user_config import UserConfig, get_or_create_config

logger = logging.getLogger(__name__)


def rerank_for_user(
    openid: str,
    articles: list[dict],
    limit: int = 20,
) -> list[dict]:
    """
    Rerank articles for a user using their personalized configuration.

    Uses the unified score_items() function from hybrid.py.

    Args:
        openid: User's OpenID
        articles: List of candidate articles (dicts)
        limit: Maximum number of articles to return

    Returns:
        Top N articles sorted by personalized score
    """
    if not articles:
        return []

    config = get_or_create_config(openid)

    # Check if user has custom configuration
    has_custom_anchors = bool(config.semantic_anchors.all_positive())
    has_custom_vip = any([
        config.vip_keywords.tier1.get("patterns"),
        config.vip_keywords.tier2.get("patterns"),
        config.vip_keywords.tier3.get("patterns"),
    ])

    if not has_custom_anchors and not has_custom_vip:
        # No custom config, use default scores
        logger.info(f"User {openid} has no custom config, using default scores")
        sorted_articles = sorted(
            articles,
            key=lambda a: a.get("semantic_score") or a.get("default_score", 0.0),
            reverse=True,
        )
        return sorted_articles[:limit]

    # Convert dicts to NewsItem objects
    items = []
    for article in articles:
        try:
            item = NewsItem(
                doi=article.get("doi"),
                title=article.get("title", ""),
                content=article.get("content", ""),
                source=article.get("source", article.get("source_name", "")),
                url=article.get("url", article.get("link", "")),
                journal_name=article.get("journal_name"),
                authors=article.get("authors", []),
                published_at=article.get("published_at"),
                image_url=article.get("image_url"),
                image_urls=article.get("image_urls", []),
            )
            # Store original dict for later
            item._original_dict = article
            items.append(item)
        except Exception as e:
            logger.warning(f"Failed to convert article to NewsItem: {e}")
            continue

    if not items:
        return []

    # Use unified scoring function
    from .S4_filter.hybrid import score_items

    scored = score_items(
        items,
        anchors=config.semantic_anchors,
        scoring_params=config.scoring_params,
        vip_keywords=config.vip_keywords,
    )

    # Convert back to dicts and add personalized_score
    results = []
    for item, score in scored[:limit]:
        if hasattr(item, '_original_dict'):
            article = item._original_dict.copy()
        else:
            article = item.model_dump()
        article["personalized_score"] = round(score, 4)
        article["semantic_score"] = round(score, 4)
        article["is_vip"] = item.is_vip
        article["vip_keywords"] = item.vip_keywords
        results.append(article)

    logger.info(
        f"Reranked {len(articles)} articles for user {openid}, "
        f"returning top {len(results)}"
    )

    return results


# Legacy class for backwards compatibility
class PersonalizedRanker:
    """
    Legacy class for backwards compatibility.

    Use rerank_for_user() directly for simpler usage.
    """

    def __init__(self, user_config: UserConfig):
        self.config = user_config

    def rerank(self, articles: list[dict], limit: int = 20) -> list[dict]:
        """Delegate to rerank_for_user()."""
        return rerank_for_user(self.config.openid, articles, limit)
