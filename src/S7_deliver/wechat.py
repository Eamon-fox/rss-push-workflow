"""微信公众号推送模块.

将日报推送到微信公众号。
"""

import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# 微信 API 端点
WX_API_BASE = "https://api.weixin.qq.com/cgi-bin"

# 从环境变量获取配置
def _get_config():
    return {
        "app_id": os.getenv("WX_APP_ID", ""),
        "app_secret": os.getenv("WX_APP_SECRET", ""),
    }


class WechatPublisher:
    """微信公众号发布器."""

    def __init__(self, app_id: str = "", app_secret: str = ""):
        config = _get_config()
        self.app_id = app_id or config["app_id"]
        self.app_secret = app_secret or config["app_secret"]
        self._access_token: Optional[str] = None
        self._token_expires: float = 0

    def _get_access_token(self) -> str:
        """获取 access_token (带缓存)."""
        import time

        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        url = f"{WX_API_BASE}/token"
        params = {
            "grant_type": "client_credential",
            "appid": self.app_id,
            "secret": self.app_secret,
        }

        resp = httpx.get(url, params=params, timeout=30)
        data = resp.json()

        if "access_token" not in data:
            raise RuntimeError(f"获取 access_token 失败: {data}")

        self._access_token = data["access_token"]
        # 提前 5 分钟过期
        self._token_expires = time.time() + data.get("expires_in", 7200) - 300

        logger.info("获取 access_token 成功")
        return self._access_token

    def upload_thumb(self, image_path: str) -> str:
        """上传封面图片，返回 media_id."""
        token = self._get_access_token()
        url = f"{WX_API_BASE}/media/uploadimg?access_token={token}"

        with open(image_path, "rb") as f:
            files = {"media": f}
            resp = httpx.post(url, files=files, timeout=60)

        data = resp.json()
        if "url" not in data:
            raise RuntimeError(f"上传图片失败: {data}")

        return data["url"]

    def create_draft(self, articles: list[dict]) -> str:
        """
        创建草稿箱图文消息.

        Args:
            articles: 图文列表，每个元素包含:
                - title: 标题
                - content: HTML 内容
                - author: 作者 (可选)
                - digest: 摘要 (可选)
                - thumb_media_id: 封面图 media_id (可选)
                - content_source_url: 原文链接 (可选)

        Returns:
            media_id
        """
        token = self._get_access_token()
        url = f"{WX_API_BASE}/draft/add?access_token={token}"

        # 构造请求体
        wx_articles = []
        for a in articles:
            wx_articles.append({
                "title": a.get("title", ""),
                "author": a.get("author", "ScholarPipe"),
                "digest": a.get("digest", "")[:120] if a.get("digest") else "",
                "content": a.get("content", ""),
                "content_source_url": a.get("content_source_url", ""),
                # 封面图 (可选，没有则使用默认)
                # "thumb_media_id": a.get("thumb_media_id", ""),
            })

        body = {"articles": wx_articles}
        resp = httpx.post(url, json=body, timeout=60)
        data = resp.json()

        if "media_id" not in data:
            raise RuntimeError(f"创建草稿失败: {data}")

        logger.info(f"创建草稿成功: {data['media_id']}")
        return data["media_id"]

    def publish_draft(self, media_id: str) -> str:
        """
        发布草稿 (提交审核后自动发布).

        Returns:
            publish_id
        """
        token = self._get_access_token()
        url = f"{WX_API_BASE}/freepublish/submit?access_token={token}"

        body = {"media_id": media_id}
        resp = httpx.post(url, json=body, timeout=60)
        data = resp.json()

        if data.get("errcode", 0) != 0:
            raise RuntimeError(f"发布失败: {data}")

        logger.info(f"发布成功: {data.get('publish_id')}")
        return data.get("publish_id", "")


def generate_article_html(articles: list[dict], date: str = "") -> str:
    """
    生成公众号文章 HTML.

    Args:
        articles: 文章列表 (从 daily.json 加载)
        date: 日期字符串

    Returns:
        HTML 内容
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    # 样式
    style = """
    <style>
        .article-item {
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .article-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            line-height: 1.4;
        }
        .article-title a {
            color: #333;
            text-decoration: none;
        }
        .article-meta {
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
        }
        .article-summary {
            font-size: 14px;
            color: #555;
            line-height: 1.7;
        }
        .vip-tag {
            display: inline-block;
            background: #ff9800;
            color: white;
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 3px;
            margin-left: 6px;
        }
        .score-tag {
            display: inline-block;
            background: #4caf50;
            color: white;
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .header {
            text-align: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 2px solid #2196f3;
        }
        .header h1 {
            font-size: 20px;
            color: #2196f3;
            margin-bottom: 8px;
        }
        .header p {
            font-size: 13px;
            color: #888;
        }
    </style>
    """

    # 头部
    html_parts = [
        style,
        f"""
        <section class="header">
            <h1>ScholarPipe 学术日报</h1>
            <p>{date} | 共 {len(articles)} 篇精选</p>
        </section>
        """
    ]

    # 文章列表
    for i, a in enumerate(articles[:15], 1):  # 最多15篇
        title = a.get("title", "")
        summary = a.get("summary", "") or a.get("content", "")[:200]
        source = a.get("source_name", "")
        score = a.get("semantic_score") or 0
        is_vip = a.get("is_vip", False)
        vip_keywords = a.get("vip_keywords", [])
        link = a.get("link", "")
        doi = a.get("doi", "")

        # 清理 summary 中的 HTML
        summary = re.sub(r'<[^>]+>', '', summary)
        if len(summary) > 250:
            summary = summary[:250] + "..."

        # VIP 标签
        vip_html = ""
        if is_vip and vip_keywords:
            vip_html = f'<span class="vip-tag">{vip_keywords[0]}</span>'
        elif is_vip:
            vip_html = '<span class="vip-tag">VIP</span>'

        # 分数标签
        score_pct = int(score * 100)
        score_html = f'<span class="score-tag">{score_pct}%</span>'

        html_parts.append(f"""
        <section class="article-item">
            <p class="article-title">
                {i}. <a href="{link}">{title}</a>
                {vip_html}
            </p>
            <p class="article-meta">
                {score_html} {source}
                {f' | DOI: {doi}' if doi else ''}
            </p>
            <p class="article-summary">{summary}</p>
        </section>
        """)

    # 底部
    html_parts.append("""
        <section style="text-align: center; color: #888; font-size: 12px; margin-top: 24px;">
            <p>由 ScholarPipe AI 自动生成</p>
            <p>点击标题查看原文</p>
        </section>
    """)

    return "\n".join(html_parts)


def publish_daily(
    articles: list[dict] = None,
    json_path: str = "output/daily.json",
    dry_run: bool = False,
) -> dict:
    """
    发布日报到公众号.

    Args:
        articles: 文章列表，如果不提供则从 json_path 加载
        json_path: JSON 文件路径
        dry_run: 仅生成 HTML，不实际发布

    Returns:
        {"success": bool, "media_id": str, "html": str}
    """
    # 加载文章
    if articles is None:
        path = Path(json_path)
        if not path.exists():
            return {"success": False, "error": f"文件不存在: {json_path}"}

        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)

    if not articles:
        return {"success": False, "error": "没有文章"}

    # 生成 HTML
    date = datetime.now().strftime("%Y-%m-%d")
    html_content = generate_article_html(articles, date)

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "html": html_content,
            "article_count": len(articles),
        }

    # 发布
    try:
        publisher = WechatPublisher()

        # 创建草稿
        draft_articles = [{
            "title": f"ScholarPipe 学术日报 | {date}",
            "content": html_content,
            "author": "ScholarPipe",
            "digest": f"今日精选 {len(articles)} 篇学术论文",
        }]

        media_id = publisher.create_draft(draft_articles)

        # 发布
        publish_id = publisher.publish_draft(media_id)

        return {
            "success": True,
            "media_id": media_id,
            "publish_id": publish_id,
            "article_count": len(articles),
        }

    except Exception as e:
        logger.error(f"发布失败: {e}")
        return {"success": False, "error": str(e)}


# CLI
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("=== DRY RUN 模式 ===\n")

    result = publish_daily(dry_run=dry_run)

    if result.get("success"):
        if dry_run:
            print(f"文章数: {result['article_count']}")
            print("\n=== HTML 预览 ===\n")
            print(result["html"][:2000])
            print("...")
        else:
            print(f"发布成功!")
            print(f"Media ID: {result.get('media_id')}")
            print(f"Publish ID: {result.get('publish_id')}")
    else:
        print(f"失败: {result.get('error')}")
        sys.exit(1)
