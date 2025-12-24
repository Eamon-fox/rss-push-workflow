# ScholarPipe API 文档

**Base URL**: `http://localhost:8080`

---

## 接口列表

| 方法 | 路径 | 描述 | 认证 |
|------|------|------|------|
| GET | `/api/health` | 健康检查 | 无 |
| GET | `/api/daily` | 系统日报（公共） | 无 |
| GET | `/api/articles/{id}` | 文章详情 | 可选 |
| GET | `/api/archive/dates` | 历史日期列表 | 无 |
| GET | `/api/archive/stats` | 归档统计 | 无 |
| POST | `/api/auth/wx-login` | 微信登录 | 无 |
| GET | `/api/auth/me` | 当前用户信息 | 需要 |
| GET | `/api/my/daily` | 我的个性化日报 | 需要 |
| POST | `/api/my/daily/regenerate` | 生成/重新生成日报 | 需要 |
| GET | `/api/my/daily/task/{task_id}` | 查询生成任务状态 | 需要 |
| GET | `/api/my/daily/task` | 我的最新任务 | 需要 |
| POST | `/api/my/read/{id}` | 标记已读 | 需要 |
| DELETE | `/api/my/seen` | 清空已读记录 | 需要 |
| GET | `/api/my/history` | 我的历史日报 | 需要 |
| GET | `/api/bookmarks` | 收藏列表 | 需要 |
| POST | `/api/bookmarks/{id}` | 添加收藏 | 需要 |
| DELETE | `/api/bookmarks/{id}` | 取消收藏 | 需要 |
| GET | `/api/bookmarks/{id}/status` | 收藏状态 | 需要 |
| GET | `/api/config` | 用户配置 | 需要 |
| PUT | `/api/config` | 更新配置 | 需要 |
| PATCH | `/api/config/vip-keywords` | 更新VIP关键词 | 需要 |
| PATCH | `/api/config/semantic-anchors` | 更新语义锚点 | 需要 |
| POST | `/api/config/reset` | 重置配置 | 需要 |
| GET | `/api/config/defaults` | 默认配置 | 无 |

---

## 认证

需要认证的接口在 Header 中传入：
```
Authorization: Bearer {token}
```

---

## 公共接口

### GET /api/health

**Response**
```json
{"service": "ScholarPipe API", "version": "0.5.0", "status": "ok"}
```

---

### GET /api/daily

系统默认日报（公共），个性化日报请用 `/api/my/daily`

**Query**
| 参数 | 类型 | 描述 |
|------|------|------|
| date | string | 日期 YYYY-MM-DD，默认今天 |
| version | int | 版本号，默认最新 |

**Response**
```json
{
  "date": "2025-12-20",
  "total": 20,
  "articles": [
    {
      "id": "doi_10_1038_s41586-024-xxxxx",
      "title": "Article title...",
      "summary": "AI生成的中文摘要...",
      "source": "Nature",
      "journal": "Nature",
      "score": 0.85,
      "is_vip": true,
      "vip_keywords": ["CRISPR"],
      "link": "https://...",
      "doi": "10.1038/s41586-024-xxxxx",
      "authors": ["Zhang Y", "Li X"],
      "published_at": "2025-12-20",
      "image_url": "https://..."
    }
  ]
}
```

---

### GET /api/articles/{article_id}

**Query**
| 参数 | 类型 | 描述 |
|------|------|------|
| date | string | 可选，不传自动查索引 |
| version | int | 可选 |

**Response**
```json
{
  "id": "doi_10_1038_s41586-024-xxxxx",
  "title": "...",
  "content": "原文内容...",
  "summary": "...",
  "source": "Nature",
  "journal": "Nature",
  "score": 0.85,
  "is_vip": true,
  "vip_keywords": ["CRISPR"],
  "link": "https://...",
  "doi": "10.1038/...",
  "authors": ["Zhang Y", "Li X", "Wang M"],
  "published_at": "2025-12-20",
  "image_url": "https://...",
  "image_urls": ["https://..."]
}
```

**Error**: `404 Article not found`

---

### GET /api/archive/dates

**Query**
| 参数 | 类型 | 描述 |
|------|------|------|
| year | int | 筛选年份 |
| month | int | 筛选月份 1-12 |
| limit | int | 返回条数，默认30 |
| offset | int | 分页偏移 |

**Response**
```json
{
  "total": 3,
  "dates": [
    {"date": "2025-12-20", "version": 2, "time": "18:30", "article_count": 12},
    {"date": "2025-12-20", "version": 1, "time": "08:00", "article_count": 20}
  ]
}
```

---

### GET /api/archive/stats

**Response**
```json
{
  "total_days": 45,
  "total_articles": 892,
  "date_range": {"start": "2025-11-01", "end": "2025-12-20"},
  "by_month": {
    "2025-12": {"days": 20, "articles": 400}
  }
}
```

---

## 认证接口

### POST /api/auth/wx-login

**Request**
```json
{"code": "033aXX000xxx"}
```

**Response**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "openid": "oXXXX...",
  "is_new_user": false
}
```

---

### GET /api/auth/me

**Response**
```json
{
  "openid": "oXXXX...",
  "created_at": "2025-12-18T10:30:00",
  "last_login": "2025-12-20T20:00:00",
  "login_count": 5,
  "bookmark_count": 12
}
```

---

## 我的日报 /api/my/*

### GET /api/my/daily

获取我的个性化日报。未生成返回404，生成中返回202。

**Query**
| 参数 | 类型 | 描述 |
|------|------|------|
| date | string | 日期，默认今天 |

**Response 200** - 已生成
```json
{
  "date": "2025-12-20",
  "generated_at": "2025-12-20T08:30:00",
  "total": 20,
  "articles": [...],
  "is_cached": true
}
```

**Response 202** - 生成中
```json
{
  "detail": {
    "message": "日报正在生成中",
    "task_id": "abc12345",
    "status": "running"
  }
}
```

**Response 404** - 未生成
```json
{"detail": "今日日报尚未生成，请调用 POST /api/my/daily/regenerate"}
```

---

### POST /api/my/daily/regenerate

异步生成/重新生成今日日报。返回 task_id，需轮询查状态。

**Response**
```json
{
  "task_id": "abc12345",
  "message": "任务已创建，请轮询 /api/my/daily/task/{task_id} 获取状态"
}
```

---

### GET /api/my/daily/task/{task_id}

查询任务状态，包含实时进度信息。

**Response**
```json
{
  "task_id": "abc12345",
  "status": "running",
  "created_at": "2025-12-20T08:30:00",
  "started_at": "2025-12-20T08:30:01",
  "finished_at": null,
  "error": null,
  "progress": {
    "step": 5,
    "total_steps": 5,
    "step_name": "生成摘要",
    "detail": "正在处理: CRISPR gene editing...",
    "current": 8,
    "total": 20,
    "percent": 64
  }
}
```

**status 取值**: `pending` | `running` | `done` | `failed`

**progress 字段说明**:
| 字段 | 类型 | 说明 |
|------|------|------|
| step | int | 当前步骤编号 (1-5) |
| total_steps | int | 总步骤数 (5) |
| step_name | str | 当前步骤名称 |
| detail | str | 详细描述 |
| current | int | 当前处理项（如第几篇文章）|
| total | int | 总项数（如共多少篇）|
| percent | int | 总体进度百分比 0-100 |

**步骤说明**:
| 步骤 | 名称 | 进度占比 |
|------|------|----------|
| 1 | 加载候选池 | 10% |
| 2 | 过滤已读 | 10% |
| 3 | 个性化排序 | 10% |
| 4 | 加载配置 | 10% |
| 5 | 生成摘要 | 60% |

完成后调用 `GET /api/my/daily` 获取结果。

---

### GET /api/my/daily/task

获取我的最新任务状态（响应格式同上，包含 progress 字段）。

**Error**: `404 没有进行中的任务`

---

### POST /api/my/read/{article_id}

标记文章已读，下次生成日报时不再出现。

**Query**
| 参数 | 类型 | 描述 |
|------|------|------|
| date | string | 可选，加速查找 |

**Response**
```json
{
  "status": "success",
  "message": "已标记为已读",
  "article_id": "doi_10_1038_xxx",
  "fingerprint": "10.1038/xxx"
}
```

---

### DELETE /api/my/seen

清空我的已读记录。清空后，之前标记为已读的文章会重新出现在日报推荐中。

**Response**
```json
{
  "status": "success",
  "message": "已清空 42 条已读记录",
  "cleared_count": 42
}
```

---

### GET /api/my/history

我的历史日报日期列表。

**Query**
| 参数 | 类型 | 描述 |
|------|------|------|
| limit | int | 返回条数，默认30 |

**Response**
```json
{
  "total": 5,
  "dates": [
    {"date": "2025-12-20", "generated_at": "2025-12-20T08:30:00", "article_count": 20},
    {"date": "2025-12-19", "generated_at": "2025-12-19T09:00:00", "article_count": 18}
  ]
}
```

---

## 收藏 /api/bookmarks

### GET /api/bookmarks

**Response**
```json
{
  "total": 2,
  "bookmarks": [
    {
      "article_id": "doi_10_1038_xxx",
      "saved_at": "2025-12-20T15:30:00",
      "note": "重要",
      "title": "Article title",
      "journal": "Nature",
      "date": "2025-12-20"
    }
  ]
}
```

---

### POST /api/bookmarks/{article_id}

**Request**
```json
{
  "note": "可选备注",
  "title": "文章标题",
  "journal": "Nature",
  "date": "2025-12-20"
}
```

title/journal/date 不传会自动查找填充。

**Response**
```json
{
  "status": "success",
  "message": "收藏成功",
  "bookmark": {...}
}
```

**Error**: `409 已收藏该文章`

---

### DELETE /api/bookmarks/{article_id}

**Response**
```json
{"status": "success", "message": "已取消收藏"}
```

**Error**: `404 未找到该收藏`

---

### GET /api/bookmarks/{article_id}/status

**Response**
```json
{"article_id": "doi_10_1038_xxx", "is_bookmarked": true}
```

---

## 用户配置 /api/config

### GET /api/config

**Response**
```json
{
  "openid": "oXXXX...",
  "vip_keywords": {
    "tier1": {"multiplier": 1.50, "patterns": ["\\bCRISPR\\b"]},
    "tier2": {"multiplier": 1.30, "patterns": []},
    "tier3": {"multiplier": 1.15, "patterns": []}
  },
  "semantic_anchors": {
    "tier1": ["tRNA splicing mechanisms", "RTCB ligase"],
    "tier2": ["enhancer RNA regulation", "transposable elements"],
    "tier3": ["RNA processing", "gene expression"],
    "negative": ["clinical trials", "agricultural applications"]
  },
  "scoring_params": {
    "tier_weights": {"tier1": 0.50, "tier2": 0.35, "tier3": 0.15},
    "tier_thresholds": {"tier1": 0.30, "tier2": 0.35, "tier3": 0.40},
    "aggregation": "max",
    "coverage_threshold": 0.40,
    "coverage_bonus": {"tier1": 0.10, "tier2": 0.06, "tier3": 0.03},
    "vip_max_multiplier": 1.80,
    "negative_threshold": 0.38,
    "negative_penalty": 0.60
  }
}
```

---

### PUT /api/config

全量更新配置。

**Request**
```json
{
  "vip_keywords": {"tier1": {...}},
  "semantic_anchors": {
    "tier1": ["核心研究方向"],
    "tier2": ["相关领域"],
    "tier3": ["扩展兴趣"],
    "negative": ["不感兴趣的主题"]
  }
}
```

**semantic_anchors 字段说明**:
| 字段 | 权重 | 阈值 | 描述 |
|------|------|------|------|
| tier1 | 0.50 | 0.30 | 核心研究方向，权重最高 |
| tier2 | 0.35 | 0.35 | 密切相关领域 |
| tier3 | 0.15 | 0.40 | 扩展兴趣，阈值最高 |
| negative | - | 0.38 | 负向锚点，匹配时降低分数 |

---

### PATCH /api/config/vip-keywords

**Request**
```json
{
  "tier1": {"multiplier": 1.50, "patterns": ["\\bCRISPR\\b"]}
}
```

---

### PATCH /api/config/semantic-anchors

增量更新语义锚点，只更新传入的字段。

**Request**
```json
{
  "tier1": ["tRNA splicing", "RTCB ligase"],
  "tier2": ["enhancer RNA"],
  "tier3": null,
  "negative": ["clinical trials"]
}
```

传 `null` 表示不更新该字段，传 `[]` 表示清空。

**Response**
```json
{
  "status": "success",
  "semantic_anchors": {
    "tier1": ["tRNA splicing", "RTCB ligase"],
    "tier2": ["enhancer RNA"],
    "tier3": [],
    "negative": ["clinical trials"]
  },
  "hints": {
    "tier1": {"truncated": 0, "deduped": 0, "dropped": 0},
    "tier2": {"truncated": 0, "deduped": 0, "dropped": 0},
    "tier3": {"truncated": 0, "deduped": 0, "dropped": 0},
    "negative": {"truncated": 0, "deduped": 0, "dropped": 0}
  }
}
```

**hints 字段说明**:
| 字段 | 描述 |
|------|------|
| truncated | 被截断的锚点数量 (超过3000字符) |
| deduped | 去重删除的数量 |
| dropped | 超出数量限制被丢弃的数量 (每层最多50条) |

---

### POST /api/config/reset

重置为默认配置。

**Response**
```json
{"status": "success", "message": "配置已重置为默认值", "config": {...}}
```

---

### GET /api/config/defaults

获取系统默认配置（无需登录）。

**Response**
```json
{
  "vip_keywords": {
    "tier1": {"multiplier": 1.50, "patterns": []},
    "tier2": {"multiplier": 1.30, "patterns": []},
    "tier3": {"multiplier": 1.15, "patterns": []}
  },
  "semantic_anchors": {
    "tier1": ["tRNA splicing mechanisms"],
    "tier2": ["enhancer RNA regulation"],
    "tier3": ["RNA processing"],
    "negative": ["clinical trials"]
  },
  "scoring_params": {
    "tier_weights": {"tier1": 0.50, "tier2": 0.35, "tier3": 0.15},
    "tier_thresholds": {"tier1": 0.30, "tier2": 0.35, "tier3": 0.40},
    "aggregation": "max",
    "coverage_threshold": 0.40,
    "coverage_bonus": {"tier1": 0.10, "tier2": 0.06, "tier3": 0.03},
    "vip_max_multiplier": 1.80,
    "negative_threshold": 0.38,
    "negative_penalty": 0.60
  },
  "limits": {
    "max_anchor_length": 3000,
    "max_user_anchors": 50
  }
}
```

**scoring_params 字段说明**:
| 字段 | 描述 |
|------|------|
| tier_weights | 各层权重，合计为1.0 |
| tier_thresholds | 各层最低相似度阈值 |
| aggregation | 聚合方式: "max" / "top_k" / "mean" |
| coverage_threshold | 命中阈值，超过此值计入 coverage bonus |
| coverage_bonus | 各层命中加成 |
| vip_max_multiplier | VIP关键词最大乘数上限 |
| negative_threshold | 负向锚点触发阈值 |
| negative_penalty | 负向惩罚乘数 |

---

## 前端调用示例

```javascript
// 1. 登录
const { code } = await wx.login();
const { token } = await post('/api/auth/wx-login', { code });

// 2. 生成日报
const { task_id } = await post('/api/my/daily/regenerate', {}, token);

// 3. 轮询状态（带进度显示）
const poll = async (onProgress) => {
  const task = await get(`/api/my/daily/task/${task_id}`, token);

  // 回调进度信息
  if (onProgress) {
    onProgress(task.progress);
    // task.progress = {
    //   step: 5,
    //   step_name: "生成摘要",
    //   detail: "正在处理: CRISPR...",
    //   current: 8,
    //   total: 20,
    //   percent: 64
    // }
  }

  if (task.status === 'done') {
    return get('/api/my/daily', token);
  } else if (task.status === 'failed') {
    throw new Error(task.error);
  }
  await sleep(1000);
  return poll(onProgress);
};

const daily = await poll((progress) => {
  console.log(`${progress.step_name}: ${progress.percent}%`);
  console.log(progress.detail);
});

// 4. 标记已读
await post(`/api/my/read/${article.id}`, {}, token);

// 5. 收藏
await post(`/api/bookmarks/${article.id}`, {
  title: article.title,
  journal: article.journal
}, token);
```

---

## 错误码

| 状态码 | 描述 |
|--------|------|
| 200 | 成功 |
| 202 | 处理中（轮询） |
| 400 | 参数错误 |
| 401 | 未认证 |
| 403 | 无权限 |
| 404 | 不存在 |
| 409 | 冲突 |
| 500 | 服务器错误 |
