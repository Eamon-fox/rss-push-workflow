"""WeChat public account (公众号) and news site parser."""

import re
from .base import BaseParser
from ...models import NewsItem, SourceType


class WechatParser(BaseParser):
    """
    Parser for WeChat public account RSS feeds.

    支持的RSS源:
    - Feeddd: https://feeddd.org/
    - WeRSS: https://werss.app/
    - RSSHub: https://docs.rsshub.app/

    公众号RSS通常包含:
    - title: 文章标题
    - link: 原文链接 (mp.weixin.qq.com)
    - description: 正文内容 (可能包含图片HTML)
    - pubDate: 发布时间
    """

    source_type = SourceType.WECHAT

    def parse_entry(self, entry: dict) -> NewsItem:
        title = entry.get("title", "")
        link = entry.get("link", "")
        content = entry.get("summary", entry.get("description", ""))

        # 从HTML内容中提取纯文本和图片
        content_text, image_url = self._extract_content_and_image(content)

        # 尝试从内容中提取DOI
        doi = self._extract_doi_from_content(content_text)

        return NewsItem(
            title=title,
            content=content_text,
            content_cn=content_text,  # 公众号内容就是中文解读
            link=link,
            doi=doi,
            image_url=image_url,
            source_type=self.source_type,
            source_name=self.source_name,
            source_url=self.source_url,
        )

    def _extract_content_and_image(self, html_content: str) -> tuple[str, str]:
        """从HTML中提取纯文本和第一张图片."""
        if not html_content:
            return "", ""

        # 提取第一张图片
        image_url = ""
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_content, re.I)
        if img_match:
            image_url = img_match.group(1)
            # 处理微信图片URL (data-src可能是真实URL)
            data_src_match = re.search(r'data-src=["\']([^"\']+)["\']', html_content, re.I)
            if data_src_match:
                image_url = data_src_match.group(1)

        # 去除HTML标签
        text = re.sub(r'<[^>]+>', ' ', html_content)
        # 清理空白
        text = re.sub(r'\s+', ' ', text).strip()

        return text, image_url

    def _extract_doi_from_content(self, text: str) -> str:
        """尝试从文本中提取DOI."""
        # DOI格式: 10.xxxx/xxxxx
        doi_match = re.search(r'(10\.\d{4,}/[^\s,;)\]]+)', text)
        if doi_match:
            doi = doi_match.group(1)
            # 清理末尾标点
            doi = doi.rstrip('.')
            return doi
        return ""


class NewsParser(BaseParser):
    """
    Parser for general science news sites.

    支持:
    - 科学网 (sciencenet.cn)
    - 果壳 (guokr.com)
    - Phys.org
    - Science Daily
    """

    source_type = SourceType.NEWS

    def parse_entry(self, entry: dict) -> NewsItem:
        title = entry.get("title", "")
        link = entry.get("link", "")
        content = entry.get("summary", entry.get("description", ""))

        # 清理HTML
        content_text = re.sub(r'<[^>]+>', ' ', content)
        content_text = re.sub(r'\s+', ' ', content_text).strip()

        # 尝试提取DOI
        doi = ""
        doi_match = re.search(r'(10\.\d{4,}/[^\s,;)\]]+)', content_text)
        if doi_match:
            doi = doi_match.group(1).rstrip('.')

        # 判断是否中文内容
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', content_text))

        return NewsItem(
            title=title,
            content=content_text,
            content_cn=content_text if is_chinese else "",
            link=link,
            doi=doi,
            source_type=self.source_type,
            source_name=self.source_name,
            source_url=self.source_url,
        )
