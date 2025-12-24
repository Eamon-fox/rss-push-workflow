"""ScholarPipe MVP API - 小程序后端接口."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Header, BackgroundTasks

from src.core import generate_article_id, today
from src.tasks import get_task_store

# ─────────────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────────────

def setup_api_logging():
    """Configure logging for API process."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "api.log"

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    return log_file

# 初始化日志
_log_file = setup_api_logging()
logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.archive import (
    load_archived_daily,
    list_archive_dates,
    get_archive_stats,
    load_historical_candidates,
    save_personal_daily,
    load_personal_daily,
    has_personal_daily,
    delete_personal_daily,
)
from src.S3_dedup import save as save_seen, mark_batch as mark_seen_batch
from src.auth import wx_login, verify_token, get_user_info
from src.S3_dedup import load as load_seen, filter_unseen, get_fingerprint
from src.models import NewsItem
from src.bookmarks import (
    get_user_bookmarks,
    add_bookmark,
    remove_bookmark,
    is_bookmarked,
    get_bookmark_count,
)
from src.article_index import get_article_location
from src.user_config import (
    UserConfig,
    VIPKeywords,
    SemanticAnchors,
    ScoringParams,
    load_user_config,
    save_user_config,
    delete_user_config,
    get_or_create_config,
    get_default_config,
    MAX_ANCHOR_LENGTH,
    MAX_USER_ANCHORS,
)
from src.personalize import rerank_for_user


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


# 全局任务存储
task_store = get_task_store()


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


class VIPKeywordsRequest(BaseModel):
    """VIP 关键词更新请求"""
    tier1: Optional[dict] = None
    tier2: Optional[dict] = None
    tier3: Optional[dict] = None


class SemanticAnchorsRequest(BaseModel):
    """语义锚点更新请求 (tiered)"""
    tier1: Optional[list[str]] = None  # 核心方向 (权重 0.50)
    tier2: Optional[list[str]] = None  # 密切相关 (权重 0.35)
    tier3: Optional[list[str]] = None  # 扩展兴趣 (权重 0.15)
    negative: Optional[list[str]] = None  # 负向锚点


class UserConfigResponse(BaseModel):
    """用户配置响应"""
    openid: str
    vip_keywords: dict
    semantic_anchors: dict
    scoring_params: dict


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
    可选认证依赖：尝试获取用户，但不强制要求

    Returns:
        openid 或 None
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
        date = today()

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


def _find_article_anywhere(
    article_id: str,
    date: Optional[str] = None,
    version: Optional[int] = None,
    openid: Optional[str] = None,
) -> Optional[dict]:
    """
    在日报中查找文章

    查找顺序:
    1. 系统日报 (system daily)
    2. 用户个人日报 (personal daily, if openid provided)

    Args:
        article_id: 文章 ID
        date: 日期，默认今天
        version: 版本号
        openid: 用户 openid (用于查找个人日报)

    Returns:
        文章字典，或 None 如果未找到
    """
    if date is None:
        date = today()

    # 1. 系统日报
    _, articles = _load_daily(date, version)
    for item in articles:
        if generate_article_id(item) == article_id:
            return item

    # 2. 用户个人日报
    if openid:
        personal_articles, _ = load_personal_daily(openid, date)
        for item in personal_articles:
            if generate_article_id(item) == article_id:
                return item

    return None


def _transform_article(item: dict) -> ArticleItem:
    """转换为 API 响应格式"""
    # 处理发布时间
    pub_at = item.get("published_at")
    if pub_at:
        # 可能是 datetime 字符串，取前10位作为日期
        pub_at = str(pub_at)[:10] if pub_at else None

    return ArticleItem(
        id=generate_article_id(item),
        title=item.get("title", ""),
        summary=item.get("summary", ""),
        source=item.get("source_name", ""),
        journal=item.get("journal_name", ""),
        score=item.get("semantic_score") or item.get("default_score") or item.get("personalized_score") or 0.0,
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

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "service": "ScholarPipe API",
        "version": "0.5.0",
        "status": "ok",
    }


@app.get("/api/daily", response_model=DailyResponse)
async def get_daily(
    date: Optional[str] = None,
    version: Optional[int] = Query(None, description="版本号，默认最新"),
):
    """
    获取系统默认日报（公共）

    个性化日报请使用 /api/my/daily

    Args:
        date: 可选，格式 YYYY-MM-DD，默认今天
        version: 可选，版本号，默认最新版本
    """
    if date is None:
        date = today()

    _, articles = _load_daily(date, version)

    # 转换格式
    items = [_transform_article(a) for a in articles]

    # 按分数排序
    items.sort(key=lambda x: x.score, reverse=True)

    return DailyResponse(
        date=date,
        total=len(items),
        articles=items,
    )


@app.get("/api/articles/{article_id}")
async def get_article(
    article_id: str,
    date: Optional[str] = Query(None, description="日期 YYYY-MM-DD，不传则自动查索引"),
    version: Optional[int] = Query(None, description="版本号，默认最新"),
    openid: Optional[str] = Depends(get_optional_user),
):
    """
    获取单篇文章详情

    查找顺序：系统日报 → 个人日报 → 候选池

    Args:
        article_id: 文章 ID
        date: 可选，不传则自动从索引查找
        version: 可选，指定版本
    """
    # 如果未传 date，先从索引查找
    if date is None:
        location = get_article_location(article_id)
        if location:
            date = location["date"]
            if version is None:
                version = location["version"]

    # 使用通用查找函数（搜索系统日报、个人日报、候选池）
    item = _find_article_anywhere(article_id, date, version, openid)

    if item:
        return {
            "id": article_id,
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source_name", ""),
            "journal": item.get("journal_name", ""),
            "score": item.get("semantic_score") or item.get("default_score") or item.get("personalized_score") or 0.0,
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

    查找顺序：系统日报 → 个人日报 → 候选池

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

    # 如果前端没传文章信息，尝试自动获取
    if not title or not journal:
        # 先尝试从索引查找日期
        article_date = date
        version = None
        if not article_date:
            location = get_article_location(article_id)
            if location:
                article_date = location["date"]
                version = location["version"]

        # 使用通用查找函数（搜索系统日报、个人日报、候选池）
        item = _find_article_anywhere(article_id, article_date, version, openid)
        if item:
            if not title:
                title = item.get("title", "")
            if not journal:
                journal = item.get("journal_name", "")
            if not date:
                date = article_date or today()

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
# 用户配置 (个性化)
# ─────────────────────────────────────────────────────────────

@app.get("/api/config", response_model=UserConfigResponse)
async def get_user_config_endpoint(openid: str = Depends(get_current_user)):
    """
    获取当前用户配置

    Returns:
        用户配置（如不存在则返回默认值）
    """
    config = get_or_create_config(openid)
    return UserConfigResponse(
        openid=config.openid,
        vip_keywords=config.vip_keywords.model_dump(),
        semantic_anchors=config.semantic_anchors.model_dump(),
        scoring_params=config.scoring_params.model_dump(),
    )


@app.put("/api/config")
async def update_user_config_endpoint(
    vip_keywords: Optional[VIPKeywordsRequest] = None,
    semantic_anchors: Optional[SemanticAnchorsRequest] = None,
    openid: str = Depends(get_current_user),
):
    """
    全量更新用户配置

    Args:
        vip_keywords: VIP 关键词配置
        semantic_anchors: 语义锚点配置

    Returns:
        更新后的配置
    """
    config = get_or_create_config(openid)

    if vip_keywords:
        if vip_keywords.tier1 is not None:
            config.vip_keywords.tier1 = vip_keywords.tier1
        if vip_keywords.tier2 is not None:
            config.vip_keywords.tier2 = vip_keywords.tier2
        if vip_keywords.tier3 is not None:
            config.vip_keywords.tier3 = vip_keywords.tier3

    if semantic_anchors:
        if semantic_anchors.tier1 is not None:
            config.semantic_anchors.tier1 = semantic_anchors.tier1
        if semantic_anchors.tier2 is not None:
            config.semantic_anchors.tier2 = semantic_anchors.tier2
        if semantic_anchors.tier3 is not None:
            config.semantic_anchors.tier3 = semantic_anchors.tier3
        if semantic_anchors.negative is not None:
            config.semantic_anchors.negative = semantic_anchors.negative

    hints = save_user_config(config)

    return {
        "status": "success",
        "message": "配置已更新",
        "config": UserConfigResponse(
            openid=config.openid,
            vip_keywords=config.vip_keywords.model_dump(),
            semantic_anchors=config.semantic_anchors.model_dump(),
            scoring_params=config.scoring_params.model_dump(),
        ),
        "hints": hints,
    }


@app.patch("/api/config/vip-keywords")
async def update_vip_keywords_endpoint(
    request: VIPKeywordsRequest,
    openid: str = Depends(get_current_user),
):
    """
    更新 VIP 关键词配置

    Args:
        request: VIP 关键词配置

    Returns:
        更新后的配置
    """
    config = get_or_create_config(openid)

    if request.tier1 is not None:
        config.vip_keywords.tier1 = request.tier1
    if request.tier2 is not None:
        config.vip_keywords.tier2 = request.tier2
    if request.tier3 is not None:
        config.vip_keywords.tier3 = request.tier3

    save_user_config(config)

    return {
        "status": "success",
        "vip_keywords": config.vip_keywords.model_dump(),
    }


@app.patch("/api/config/semantic-anchors")
async def update_semantic_anchors_endpoint(
    request: SemanticAnchorsRequest,
    openid: str = Depends(get_current_user),
):
    """
    更新语义锚点配置

    Args:
        request: 语义锚点配置

    Returns:
        更新后的配置，包含 hints（去重/截断/丢弃信息）
    """
    config = get_or_create_config(openid)

    if request.tier1 is not None:
        config.semantic_anchors.tier1 = request.tier1
    if request.tier2 is not None:
        config.semantic_anchors.tier2 = request.tier2
    if request.tier3 is not None:
        config.semantic_anchors.tier3 = request.tier3
    if request.negative is not None:
        config.semantic_anchors.negative = request.negative

    hints = save_user_config(config)

    return {
        "status": "success",
        "semantic_anchors": config.semantic_anchors.model_dump(),
        "hints": hints,  # 告知前端发生了什么清理操作
    }


@app.post("/api/config/reset")
async def reset_user_config_endpoint(openid: str = Depends(get_current_user)):
    """
    重置用户配置为默认值

    Returns:
        重置后的配置
    """
    # 先删除旧配置，再用系统默认值创建新配置
    delete_user_config(openid)
    config = get_or_create_config(openid)

    return {
        "status": "success",
        "message": "配置已重置为默认值",
        "config": UserConfigResponse(
            openid=config.openid,
            vip_keywords=config.vip_keywords.model_dump(),
            semantic_anchors=config.semantic_anchors.model_dump(),
            scoring_params=config.scoring_params.model_dump(),
        ),
    }


@app.get("/api/config/defaults")
async def get_default_config_endpoint():
    """
    获取系统默认配置（参考用）

    Returns:
        默认配置值和限制参数
    """
    defaults = get_default_config()
    defaults["limits"] = {
        "max_anchor_length": MAX_ANCHOR_LENGTH,
        "max_user_anchors": MAX_USER_ANCHORS,
    }
    return defaults


# ─────────────────────────────────────────────────────────────
# 我的日报 (/api/my/*)
# ─────────────────────────────────────────────────────────────

class MyDailyResponse(BaseModel):
    """我的日报响应"""
    date: str
    generated_at: Optional[str]
    total: int
    articles: list[ArticleItem]
    is_cached: bool  # 是否从缓存返回


class TaskProgress(BaseModel):
    """任务进度"""
    step: int              # 当前步骤编号 (1-5)
    total_steps: int       # 总步骤数
    step_name: str         # 当前步骤名称
    detail: str            # 详细描述
    current: int           # 当前处理项
    total: int             # 总项数
    percent: int           # 总体进度百分比 0-100


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: str  # pending, running, done, failed
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    error: Optional[str]
    progress: TaskProgress  # 进度信息


class GenerateTaskResponse(BaseModel):
    """生成任务创建响应"""
    task_id: str
    message: str


def _generate_personal_daily_sync(openid: str, date: str, task_id: str = None) -> list[dict]:
    """
    同步生成个人日报（内部函数）

    Args:
        openid: 用户 openid
        date: 日期
        task_id: 任务 ID，用于更新进度

    Returns:
        生成的文章列表 (原始格式)
    """
    def update_progress(step: int, step_name: str, detail: str = "", current: int = 0, total: int = 0):
        if task_id:
            task_store.update_progress(task_id, step, step_name, detail, current, total)

    # Step 1: 加载候选池
    update_progress(1, "加载候选池", "正在从历史归档加载文章...")
    candidates = load_historical_candidates()
    if not candidates:
        return []
    update_progress(1, "加载候选池", f"已加载 {len(candidates)} 篇候选文章")

    # Step 2: 过滤已读
    update_progress(2, "过滤已读", "正在过滤已读文章...")
    seen_records = load_seen(user_id=openid)
    original_count = len(candidates)
    if seen_records:
        candidate_items = [NewsItem(**c) for c in candidates]
        filtered_items, _ = filter_unseen(candidate_items, seen_records)
        candidates = [item.model_dump() for item in filtered_items]
    update_progress(2, "过滤已读", f"过滤后剩余 {len(candidates)} 篇（排除 {original_count - len(candidates)} 篇已读）")

    if not candidates:
        return []

    # Step 3: 个性化排序
    update_progress(3, "个性化排序", "正在根据用户偏好排序...")
    articles = rerank_for_user(openid, candidates, limit=20)
    update_progress(3, "个性化排序", f"已选出 {len(articles)} 篇推荐文章")

    # Step 4: 加载用户配置
    update_progress(4, "加载配置", "正在加载用户个性化配置...")

    # Step 5: 生成摘要
    from src.S6_llm.process import LLMCache, summarize_single

    cache = LLMCache()
    result_articles = []
    total_articles = len(articles)

    for idx, article in enumerate(articles):
        update_progress(5, "生成摘要", f"正在处理: {article.get('title', '')[:30]}...", idx + 1, total_articles)

        if not article.get("summary"):
            item = NewsItem(
                doi=article.get("doi"),
                title=article.get("title", ""),
                content=article.get("content", ""),
                source=article.get("source", ""),
                url=article.get("url", ""),
            )
            cached = cache.get(item)

            if cached:
                article["summary"] = cached
            else:
                try:
                    result_item = summarize_single(item, use_cache=True)
                    if result_item.summary:
                        article["summary"] = result_item.summary
                except Exception:
                    content = article.get("content", "")
                    article["summary"] = content[:200] + "..." if len(content) > 200 else content

        result_articles.append(article)

    # Step 6: 保存到持久化存储
    update_progress(5, "保存结果", "正在保存日报...", total_articles, total_articles)
    save_personal_daily(openid, result_articles, date)

    return result_articles


def _run_generate_task(task_id: str, openid: str, date: str):
    """
    后台运行生成任务（在线程中执行）
    """
    try:
        task_store.set_running(task_id)
        articles = _generate_personal_daily_sync(openid, date, task_id)
        task_store.set_done(task_id, {
            "article_count": len(articles),
            "date": date,
        })
    except Exception as e:
        task_store.set_failed(task_id, str(e))


@app.get("/api/my/daily")
async def get_my_daily(
    date: Optional[str] = Query(None, description="日期 YYYY-MM-DD，默认今天"),
    openid: str = Depends(get_current_user),
):
    """
    获取我的个性化日报

    - 如果已生成过 → 直接返回缓存
    - 如果未生成 → 返回 404，需调用 /api/my/daily/regenerate 生成

    Args:
        date: 日期，默认今天

    Returns:
        个性化日报，或 404 如果未生成
    """
    if date is None:
        date = today()

    # 尝试加载已保存的个人日报
    articles, metadata = load_personal_daily(openid, date)

    if articles:
        # 缓存命中
        items = [_transform_article(a) for a in articles]
        return MyDailyResponse(
            date=date,
            generated_at=metadata.get("generated_at"),
            total=len(items),
            articles=items,
            is_cached=True,
        )

    # 缓存未命中，检查是否有正在进行的任务
    task = task_store.get_user_task(openid)
    if task and task["status"] in ("pending", "running"):
        raise HTTPException(
            status_code=202,
            detail={
                "message": "日报正在生成中",
                "task_id": task["task_id"],
                "status": task["status"],
            },
        )

    # 没有缓存也没有任务
    raise HTTPException(
        status_code=404,
        detail="今日日报尚未生成，请调用 POST /api/my/daily/regenerate",
    )


@app.post("/api/my/daily/regenerate", response_model=GenerateTaskResponse)
async def regenerate_my_daily(
    background_tasks: BackgroundTasks,
    openid: str = Depends(get_current_user),
):
    """
    异步生成/重新生成今日个性化日报

    用于：
    - 首次生成日报
    - 用户修改配置后想重新生成
    - 用户想刷新推荐

    Returns:
        task_id，用于轮询状态
    """
    date = today()

    # 检查是否有正在进行的任务
    existing_task = task_store.get_user_task(openid)
    if existing_task and existing_task["status"] in ("pending", "running"):
        return GenerateTaskResponse(
            task_id=existing_task["task_id"],
            message="已有任务在进行中",
        )

    # 删除旧的日报
    delete_personal_daily(openid, date)

    # 创建新任务
    task_id = task_store.create(openid)

    # 在后台线程中执行生成
    background_tasks.add_task(_run_generate_task, task_id, openid, date)

    return GenerateTaskResponse(
        task_id=task_id,
        message="任务已创建，请轮询 /api/my/daily/task/{task_id} 获取状态",
    )


@app.get("/api/my/daily/task/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    openid: str = Depends(get_current_user),
):
    """
    查询任务状态

    状态说明：
    - pending: 等待执行
    - running: 正在执行
    - done: 完成（日报已生成，可调用 GET /api/my/daily 获取）
    - failed: 失败

    Args:
        task_id: 任务 ID

    Returns:
        任务状态
    """
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 验证任务归属
    if task["openid"] != openid:
        raise HTTPException(status_code=403, detail="无权访问此任务")

    return TaskResponse(
        task_id=task["task_id"],
        status=task["status"],
        created_at=task["created_at"],
        started_at=task.get("started_at"),
        finished_at=task.get("finished_at"),
        error=task.get("error"),
        progress=TaskProgress(**task.get("progress", {})),
    )


@app.get("/api/my/daily/task", response_model=TaskResponse)
async def get_my_latest_task(
    openid: str = Depends(get_current_user),
):
    """
    获取当前用户最新的任务状态

    Returns:
        最新任务状态，或 404 如果没有任务
    """
    task = task_store.get_user_task(openid)
    if not task:
        raise HTTPException(status_code=404, detail="没有进行中的任务")

    return TaskResponse(
        task_id=task["task_id"],
        status=task["status"],
        created_at=task["created_at"],
        started_at=task.get("started_at"),
        finished_at=task.get("finished_at"),
        error=task.get("error"),
        progress=TaskProgress(**task.get("progress", {})),
    )


@app.post("/api/my/read/{article_id}")
async def mark_article_read(
    article_id: str,
    date: Optional[str] = Query(None, description="文章所在日期，用于加速查找"),
    openid: str = Depends(get_current_user),
):
    """
    标记文章为已读

    下次生成日报时，该文章不会再出现。

    Args:
        article_id: 文章 ID
        date: 可选，文章所在日期（加速查找）

    Returns:
        操作结果
    """
    # 查找文章以获取原始 DOI/title，计算正确的 fingerprint
    item = _find_article_anywhere(article_id, date=date, openid=openid)

    if item:
        # 使用原始数据计算正确的 fingerprint
        news_item = NewsItem(
            doi=item.get("doi"),
            title=item.get("title", ""),
            content=item.get("content", ""),
            source=item.get("source_name", ""),
            url=item.get("link", ""),
        )
        fingerprint = get_fingerprint(news_item)
    else:
        # 未找到文章，尝试从 article_id 还原 fingerprint
        # doi_xxx 格式无法完美还原，使用 article_id 本身作为标识
        fingerprint = article_id

    if not fingerprint:
        raise HTTPException(status_code=400, detail="无法计算文章标识")

    # 加载用户的 seen 记录
    seen_records = load_seen(user_id=openid)

    # 标记为已读
    mark_seen_batch(seen_records, [fingerprint])

    # 保存
    save_seen(seen_records, user_id=openid)

    return {
        "status": "success",
        "message": "已标记为已读",
        "article_id": article_id,
        "fingerprint": fingerprint,
    }


@app.delete("/api/my/seen")
async def clear_my_seen(
    openid: str = Depends(get_current_user),
):
    """
    清空我的已读记录

    清空后，之前标记为已读的文章会重新出现在日报推荐中。

    Returns:
        操作结果
    """
    seen_records = load_seen(user_id=openid)
    count = len(seen_records)

    # 保存空记录
    save_seen({}, user_id=openid)

    return {
        "status": "success",
        "message": f"已清空 {count} 条已读记录",
        "cleared_count": count,
    }


@app.get("/api/my/history")
async def get_my_daily_history(
    limit: int = Query(30, ge=1, le=100, description="返回条数"),
    openid: str = Depends(get_current_user),
):
    """
    获取我的历史日报日期列表

    Returns:
        有个人日报的日期列表
    """
    from src.archive import ARCHIVE_BASE, _sanitize_openid

    safe_openid = _sanitize_openid(openid)
    dates = []

    if not ARCHIVE_BASE.exists():
        return {"total": 0, "dates": []}

    # 扫描归档目录
    for year_dir in sorted(ARCHIVE_BASE.iterdir(), reverse=True):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for month_dir in sorted(year_dir.iterdir(), reverse=True):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue
            for day_dir in sorted(month_dir.iterdir(), reverse=True):
                if not day_dir.is_dir() or not day_dir.name.isdigit():
                    continue

                personal_file = day_dir / "personal" / f"{safe_openid}.json"
                if personal_file.exists():
                    date_str = f"{year_dir.name}-{month_dir.name}-{day_dir.name}"

                    # 读取元数据
                    try:
                        with open(personal_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        dates.append({
                            "date": date_str,
                            "generated_at": data.get("generated_at"),
                            "article_count": data.get("article_count", 0),
                        })
                    except Exception:
                        dates.append({
                            "date": date_str,
                            "generated_at": None,
                            "article_count": 0,
                        })

                    if len(dates) >= limit:
                        return {"total": len(dates), "dates": dates}

    return {"total": len(dates), "dates": dates}


# ─────────────────────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
