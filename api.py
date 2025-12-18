"""ScholarPipe MVP API - 小程序后端接口."""

import json
import hashlib
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.archive import load_archived_daily, list_archive_dates, get_archive_stats
from src.auth import wx_login, verify_token, get_user_info
from src.bookmarks import (
    get_user_bookmarks,
    add_bookmark,
    remove_bookmark,
    is_bookmarked,
    get_bookmark_count,
)
from src.article_index import get_article_location


# ─────────────────────────────────────────────────────────────
# 流水线运行状态
# ─────────────────────────────────────────────────────────────

class PipelineStatus:
    """流水线运行状态追踪"""
    def __init__(self):
        self.running = False
        self.last_run: Optional[str] = None
        self.last_status: Optional[str] = None  # success, failed, running
        self.last_error: Optional[str] = None
        self.last_duration: Optional[float] = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            self.running = True
            self.last_run = datetime.now().isoformat()
            self.last_status = "running"
            self.last_error = None

    def finish(self, success: bool, error: str = None, duration: float = None):
        with self._lock:
            self.running = False
            self.last_status = "success" if success else "failed"
            self.last_error = error
            self.last_duration = duration

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "last_run": self.last_run,
                "last_status": self.last_status,
                "last_error": self.last_error,
                "last_duration_seconds": self.last_duration,
            }


pipeline_status = PipelineStatus()

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
    journal: str
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


class WxLoginRequest(BaseModel):
    """微信登录请求"""
    code: str


class WxLoginResponse(BaseModel):
    """微信登录响应"""
    token: str
    openid: str
    is_new_user: bool


class BookmarkRequest(BaseModel):
    """收藏请求"""
    note: str = ""
    title: str = ""
    journal: str = ""
    date: str = ""


class BookmarkItem(BaseModel):
    """收藏项"""
    article_id: str
    saved_at: str
    note: str
    title: str
    journal: str
    date: str


class BookmarkListResponse(BaseModel):
    """收藏列表响应"""
    total: int
    bookmarks: list[BookmarkItem]


# ─────────────────────────────────────────────────────────────
# 认证依赖
# ─────────────────────────────────────────────────────────────

async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> str:
    """
    认证依赖：从 Authorization header 获取当前用户

    Returns:
        openid

    Raises:
        HTTPException 401 如果未认证或 token 无效
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证信息")

    # 支持 "Bearer xxx" 和 "xxx" 两种格式
    token = authorization
    if authorization.startswith("Bearer "):
        token = authorization[7:]

    openid = verify_token(token)
    if not openid:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")

    return openid


async def get_optional_user(
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """
    可选认证依赖：尝试获取当前用户，失败返回 None
    """
    if not authorization:
        return None

    token = authorization
    if authorization.startswith("Bearer "):
        token = authorization[7:]

    return verify_token(token)


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def _generate_id(item: dict) -> str:
    """生成文章ID (基于DOI或标题哈希)"""
    if item.get("doi"):
        return f"doi_{item['doi'].replace('/', '_').replace('.', '_')}"
    title = (item.get("title") or "").strip().lower()
    return f"t_{hashlib.md5(title.encode()).hexdigest()[:12]}"


def _load_daily(
    date: Optional[str] = None,
    version: Optional[int] = None,
) -> tuple[str, list[dict]]:
    """
    加载日报数据

    Args:
        date: 日期字符串 YYYY-MM-DD，None 表示今天
        version: 版本号，None 表示最新版本

    Returns:
        (date_str, articles)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # 优先从归档加载
    articles, metadata = load_archived_daily(date, version)
    if articles:
        return date, articles

    # 回退: 尝试 output/daily.json (当天首次运行前的兼容)
    daily_file = OUTPUT_DIR / "daily.json"
    if daily_file.exists():
        try:
            with open(daily_file, "r", encoding="utf-8") as f:
                return date, json.load(f)
        except Exception as e:
            print(f"Error loading {daily_file}: {e}")

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
        journal=item.get("journal_name", ""),
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
async def get_daily(
    date: Optional[str] = None,
    version: Optional[int] = Query(None, description="版本号，默认最新"),
):
    """
    获取日报

    Args:
        date: 可选，格式 YYYY-MM-DD，默认今天
        version: 可选，版本号，默认最新版本
    """
    date_str, articles = _load_daily(date, version)

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
async def get_article(
    article_id: str,
    date: Optional[str] = Query(None, description="日期 YYYY-MM-DD，不传则自动查索引"),
    version: Optional[int] = Query(None, description="版本号，默认最新"),
):
    """
    获取单篇文章详情

    Args:
        article_id: 文章 ID
        date: 可选，不传则自动从索引查找
        version: 可选，指定版本
    """
    # 如果未传 date，从索引查找
    if date is None:
        location = get_article_location(article_id)
        if location:
            date = location["date"]
            if version is None:
                version = location["version"]

    _, articles = _load_daily(date, version)

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


@app.get("/api/archive/dates")
async def get_archive_dates_endpoint(
    year: Optional[int] = Query(None, description="筛选年份"),
    month: Optional[int] = Query(None, ge=1, le=12, description="筛选月份"),
    limit: int = Query(30, ge=1, le=100, description="返回条数"),
    offset: int = Query(0, ge=0, description="分页偏移"),
):
    """
    获取历史日期列表

    Returns:
        包含日期、版本数、文章数的列表
    """
    dates = list_archive_dates(year, month, limit, offset)
    return {
        "total": len(dates),
        "dates": dates,
    }


@app.get("/api/archive/stats")
async def get_archive_stats_endpoint():
    """
    获取归档统计信息

    Returns:
        归档总览统计
    """
    return get_archive_stats()


# ─────────────────────────────────────────────────────────────
# 微信登录
# ─────────────────────────────────────────────────────────────

@app.post("/api/auth/wx-login", response_model=WxLoginResponse)
async def wx_login_endpoint(request: WxLoginRequest):
    """
    微信小程序登录

    Args:
        request: 包含小程序 wx.login() 返回的 code

    Returns:
        JWT token 和用户信息
    """
    try:
        result = await wx_login(request.code)
        return WxLoginResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")


@app.get("/api/auth/me")
async def get_me(openid: str = Depends(get_current_user)):
    """
    获取当前用户信息

    Returns:
        用户信息
    """
    user_info = get_user_info(openid)
    if not user_info:
        raise HTTPException(status_code=404, detail="用户不存在")

    return {
        "openid": openid,
        **user_info,
        "bookmark_count": get_bookmark_count(openid),
    }


# ─────────────────────────────────────────────────────────────
# 收藏功能
# ─────────────────────────────────────────────────────────────

@app.get("/api/bookmarks", response_model=BookmarkListResponse)
async def get_bookmarks_endpoint(openid: str = Depends(get_current_user)):
    """
    获取用户收藏列表

    Returns:
        收藏列表
    """
    bookmarks = get_user_bookmarks(openid)
    return BookmarkListResponse(
        total=len(bookmarks),
        bookmarks=[BookmarkItem(**b) for b in bookmarks],
    )


@app.post("/api/bookmarks/{article_id}")
async def add_bookmark_endpoint(
    article_id: str,
    request: BookmarkRequest = None,
    openid: str = Depends(get_current_user),
):
    """
    添加收藏

    Args:
        article_id: 文章 ID
        request: 包含备注和文章基本信息

    Returns:
        收藏记录
    """
    note = request.note if request else ""
    title = request.title if request else ""
    journal = request.journal if request else ""
    date = request.date if request else ""

    # 检查是否已收藏
    if is_bookmarked(openid, article_id):
        raise HTTPException(status_code=409, detail="已收藏该文章")

    bookmark = add_bookmark(
        openid=openid,
        article_id=article_id,
        note=note,
        title=title,
        journal=journal,
        date=date,
    )
    return {
        "status": "success",
        "message": "收藏成功",
        "bookmark": bookmark,
    }


@app.delete("/api/bookmarks/{article_id}")
async def remove_bookmark_endpoint(
    article_id: str,
    openid: str = Depends(get_current_user),
):
    """
    取消收藏

    Args:
        article_id: 文章 ID

    Returns:
        操作结果
    """
    success = remove_bookmark(openid, article_id)
    if not success:
        raise HTTPException(status_code=404, detail="未找到该收藏")

    return {
        "status": "success",
        "message": "已取消收藏",
    }


@app.get("/api/bookmarks/{article_id}/status")
async def get_bookmark_status(
    article_id: str,
    openid: str = Depends(get_current_user),
):
    """
    检查文章是否已收藏

    Args:
        article_id: 文章 ID

    Returns:
        收藏状态
    """
    return {
        "article_id": article_id,
        "is_bookmarked": is_bookmarked(openid, article_id),
    }


# ─────────────────────────────────────────────────────────────
# 手动触发
# ─────────────────────────────────────────────────────────────

def _run_pipeline():
    """后台运行流水线"""
    import time
    start_time = time.time()

    try:
        pipeline_status.start()

        # 运行 main.py
        result = subprocess.run(
            [".venv/bin/python", "main.py"],
            cwd="/opt/rss-push-workflow",
            capture_output=True,
            text=True,
            timeout=1800,  # 30 分钟超时
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            pipeline_status.finish(success=True, duration=duration)
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            pipeline_status.finish(success=False, error=error_msg, duration=duration)

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        pipeline_status.finish(success=False, error="Pipeline timeout (30min)", duration=duration)
    except Exception as e:
        duration = time.time() - start_time
        pipeline_status.finish(success=False, error=str(e), duration=duration)


@app.post("/api/trigger")
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """
    手动触发流水线运行

    Returns:
        触发状态
    """
    if pipeline_status.running:
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running"
        )

    background_tasks.add_task(_run_pipeline)

    return {
        "status": "triggered",
        "message": "Pipeline started in background",
        "started_at": datetime.now().isoformat(),
    }


@app.get("/api/trigger/status")
async def get_pipeline_status():
    """
    获取流水线运行状态

    Returns:
        当前状态和上次运行信息，包含实时进度
    """
    result = pipeline_status.to_dict()

    # 如果正在运行，读取进度文件
    if result["running"]:
        progress_file = DATA_DIR / "pipeline_progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    result["progress"] = json.load(f)
            except Exception:
                result["progress"] = None
        else:
            result["progress"] = None
    else:
        result["progress"] = None

    return result


# ─────────────────────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
