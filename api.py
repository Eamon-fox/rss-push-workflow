"""ScholarPipe MVP API - 小程序后端接口."""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="ScholarPipe API",
    description="学术日报 MVP 接口",
    version="0.1.0",
)

# 允许跨域 (小程序/网页调用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据目录
OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")


# ─────────────────────────────────────────────────────────────
# 响应模型
# ─────────────────────────────────────────────────────────────

class ArticleItem(BaseModel):
    """文章简要信息 (列表用)"""
    id: str
    title: str
    summary: str
    source: str
    score: float
    is_vip: bool
    vip_keywords: list[str]
    link: str
    doi: str
    authors: list[str]
    published_at: Optional[str]
    image_url: str


class DailyResponse(BaseModel):
    """日报响应"""
    date: str
    total: int
    articles: list[ArticleItem]


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def _generate_id(item: dict) -> str:
    """生成文章ID (基于DOI或标题哈希)"""
    if item.get("doi"):
        return f"doi_{item['doi'].replace('/', '_').replace('.', '_')}"
    title = (item.get("title") or "").strip().lower()
    return f"t_{hashlib.md5(title.encode()).hexdigest()[:12]}"


def _load_daily(date: Optional[str] = None) -> tuple[str, list[dict]]:
    """
    加载日报数据

    Args:
        date: 日期字符串 YYYY-MM-DD，None 表示今天

    Returns:
        (date_str, articles)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # 先尝试 output/daily.json (最新)
    daily_file = OUTPUT_DIR / "daily.json"

    # 也可以尝试按日期的目录结构
    dated_file = DATA_DIR / "filtered" / date / "filtered.json"

    target_file = None
    if daily_file.exists():
        target_file = daily_file
    elif dated_file.exists():
        target_file = dated_file

    if target_file is None:
        return date, []

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            articles = json.load(f)
        return date, articles
    except Exception as e:
        print(f"Error loading {target_file}: {e}")
        return date, []


def _transform_article(item: dict) -> ArticleItem:
    """转换为 API 响应格式"""
    # 处理发布时间
    pub_at = item.get("published_at")
    if pub_at:
        # 可能是 datetime 字符串，取前10位作为日期
        pub_at = str(pub_at)[:10] if pub_at else None

    return ArticleItem(
        id=_generate_id(item),
        title=item.get("title", ""),
        summary=item.get("summary", ""),
        source=item.get("source_name", ""),
        score=item.get("semantic_score") or 0.0,
        is_vip=item.get("is_vip", False),
        vip_keywords=item.get("vip_keywords", []),
        link=item.get("link", ""),
        doi=item.get("doi", ""),
        authors=item.get("authors", [])[:5],  # 最多5个作者
        published_at=pub_at,
        image_url=item.get("image_url", ""),
    )


# ─────────────────────────────────────────────────────────────
# API 端点
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "ScholarPipe API",
        "version": "0.1.0",
        "status": "ok",
    }


@app.get("/api/daily", response_model=DailyResponse)
async def get_daily(date: Optional[str] = None):
    """
    获取日报

    Args:
        date: 可选，格式 YYYY-MM-DD，默认今天
    """
    date_str, articles = _load_daily(date)

    # 转换格式
    items = [_transform_article(a) for a in articles]

    # 按分数排序 (已经排好序的，但确保一下)
    items.sort(key=lambda x: x.score, reverse=True)

    return DailyResponse(
        date=date_str,
        total=len(items),
        articles=items,
    )


@app.get("/api/articles/{article_id}")
async def get_article(article_id: str):
    """
    获取单篇文章详情
    """
    _, articles = _load_daily()

    for item in articles:
        if _generate_id(item) == article_id:
            return {
                "id": article_id,
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source_name", ""),
                "journal": item.get("journal_name", ""),
                "score": item.get("semantic_score") or 0.0,
                "is_vip": item.get("is_vip", False),
                "vip_keywords": item.get("vip_keywords", []),
                "link": item.get("link", ""),
                "doi": item.get("doi", ""),
                "authors": item.get("authors", []),
                "published_at": str(item.get("published_at", ""))[:10] if item.get("published_at") else None,
                "image_url": item.get("image_url", ""),
                "image_urls": item.get("image_urls", []),
            }

    raise HTTPException(status_code=404, detail="Article not found")


# ─────────────────────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
