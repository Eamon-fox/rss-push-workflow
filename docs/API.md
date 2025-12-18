# ScholarPipe API 接口文档

**版本**: v0.1.0
**Base URL**: `http://localhost:8080`
**协议**: HTTP/HTTPS

---

## 概述

ScholarPipe API 提供学术日报数据的访问接口，支持获取每日精选文章列表和单篇文章详情。

### 启动服务

```bash
python api.py
# 服务监听: http://0.0.0.0:8080
```

### 通用说明

- 所有接口返回 JSON 格式
- 支持 CORS 跨域访问
- 时间格式: `YYYY-MM-DD`

---

## 接口列表

| 方法 | 路径 | 描述 | 认证 |
|------|------|------|------|
| GET | `/` | 健康检查 | 无 |
| GET | `/api/daily` | 获取日报文章列表 | 无 |
| GET | `/api/articles/{article_id}` | 获取单篇文章详情 | 无 |
| GET | `/api/archive/dates` | 获取历史日期列表 | 无 |
| GET | `/api/archive/stats` | 获取归档统计信息 | 无 |
| POST | `/api/trigger` | 手动触发流水线运行 | 无 |
| GET | `/api/trigger/status` | 获取流水线运行状态 | 无 |
| POST | `/api/auth/wx-login` | 微信小程序登录 | 无 |
| GET | `/api/auth/me` | 获取当前用户信息 | 需要 |
| GET | `/api/bookmarks` | 获取收藏列表 | 需要 |
| POST | `/api/bookmarks/{article_id}` | 添加收藏 | 需要 |
| DELETE | `/api/bookmarks/{article_id}` | 取消收藏 | 需要 |
| GET | `/api/bookmarks/{article_id}/status` | 检查收藏状态 | 需要 |

---

## 1. 健康检查

检查 API 服务是否正常运行。

### 请求

```
GET /
```

### 响应

```json
{
  "service": "ScholarPipe API",
  "version": "0.1.0",
  "status": "ok"
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| service | string | 服务名称 |
| version | string | API 版本号 |
| status | string | 服务状态，正常为 `ok` |

---

## 2. 获取日报

获取指定日期的学术日报文章列表，按相关性分数降序排列。

### 请求

```
GET /api/daily
GET /api/daily?date=2025-12-18
GET /api/daily?date=2025-12-18&version=1
```

### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| date | string | 否 | 日期，格式 `YYYY-MM-DD`，默认为当天 |
| version | int | 否 | 版本号，默认为最新版本 (每天可能有多个版本) |

### 响应

```json
{
  "date": "2024-12-18",
  "total": 42,
  "articles": [
    {
      "id": "doi_10_1038_s41586-024-xxxxx",
      "title": "Single-cell analysis reveals...",
      "summary": "这项研究利用单细胞测序技术...",
      "source": "Nature",
      "score": 0.85,
      "is_vip": true,
      "vip_keywords": ["single-cell", "RNA-seq"],
      "link": "https://www.nature.com/articles/...",
      "doi": "10.1038/s41586-024-xxxxx",
      "authors": ["Zhang Y", "Li X", "Wang M"],
      "published_at": "2024-12-18",
      "image_url": "https://media.nature.com/..."
    }
  ]
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| date | string | 日报日期 |
| total | int | 文章总数 |
| articles | array | 文章列表 |

### Article 对象

| 字段 | 类型 | 描述 |
|------|------|------|
| id | string | 文章唯一标识 (基于 DOI 或标题哈希) |
| title | string | 文章标题 |
| summary | string | AI 生成的中文摘要 |
| source | string | 来源名称 (Nature, Science, PubMed 等) |
| journal | string | 期刊名称 |
| score | float | 语义相关性分数 (0-1) |
| is_vip | bool | 是否命中 VIP 关键词 |
| vip_keywords | array | 命中的 VIP 关键词列表 |
| link | string | 原文链接 |
| doi | string | DOI 标识符 |
| authors | array | 作者列表 (最多5人) |
| published_at | string | 发布日期 |
| image_url | string | 文章配图 URL |

### 示例

```bash
# 获取今日日报
curl http://localhost:8080/api/daily

# 获取指定日期日报
curl http://localhost:8080/api/daily?date=2025-12-17

# 获取指定日期的第一个版本
curl "http://localhost:8080/api/daily?date=2025-12-17&version=1"
```

---

## 3. 获取文章详情

根据文章 ID 获取单篇文章的完整信息。**无需传入日期参数**，后端自动通过索引查找文章位置。

### 请求

```
GET /api/articles/{article_id}
```

### 路径参数

| 参数 | 类型 | 描述 |
|------|------|------|
| article_id | string | 文章唯一标识 |

### 查询参数 (可选)

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| date | string | 否 | 日期，不传则自动从索引查找 |
| version | int | 否 | 版本号，默认为最新版本 |

### 响应

```json
{
  "id": "doi_10_1038_s41586-024-xxxxx",
  "title": "Single-cell analysis reveals...",
  "content": "Abstract: Recent advances in...",
  "summary": "这项研究利用单细胞测序技术...",
  "source": "Nature",
  "journal": "Nature",
  "score": 0.85,
  "is_vip": true,
  "vip_keywords": ["single-cell", "RNA-seq"],
  "link": "https://www.nature.com/articles/...",
  "doi": "10.1038/s41586-024-xxxxx",
  "authors": ["Zhang Y", "Li X", "Wang M", "Chen L", "Liu H"],
  "published_at": "2024-12-18",
  "image_url": "https://media.nature.com/...",
  "image_urls": [
    "https://media.nature.com/fig1.jpg",
    "https://media.nature.com/fig2.jpg"
  ]
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| id | string | 文章唯一标识 |
| title | string | 文章标题 |
| content | string | 原始内容/摘要 |
| summary | string | AI 生成的中文摘要 |
| source | string | 来源名称 |
| journal | string | 期刊名称 |
| score | float | 语义相关性分数 (0-1) |
| is_vip | bool | 是否命中 VIP 关键词 |
| vip_keywords | array | 命中的 VIP 关键词列表 |
| link | string | 原文链接 |
| doi | string | DOI 标识符 |
| authors | array | 完整作者列表 |
| published_at | string | 发布日期 |
| image_url | string | 主配图 URL |
| image_urls | array | 所有配图 URL 列表 |

### 错误响应

**404 Not Found**

```json
{
  "detail": "Article not found"
}
```

### 示例

```bash
# 获取文章详情 (自动查找，推荐)
curl http://localhost:8080/api/articles/doi_10_1038_s41586-024-xxxxx

# 指定日期查找 (可选)
curl "http://localhost:8080/api/articles/doi_10_1038_s41586-024-xxxxx?date=2025-12-17"
```

---

## 4. 获取历史日期列表

获取所有已归档的日报日期列表。

### 请求

```
GET /api/archive/dates
GET /api/archive/dates?year=2025
GET /api/archive/dates?year=2025&month=12
GET /api/archive/dates?limit=10&offset=0
```

### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| year | int | 否 | 筛选年份 |
| month | int | 否 | 筛选月份 (1-12) |
| limit | int | 否 | 返回条数，默认 30，最大 100 |
| offset | int | 否 | 分页偏移，默认 0 |

### 响应

返回扁平结构，每个版本作为单独条目，按日期降序、版本降序排列：

```json
{
  "total": 3,
  "dates": [
    {
      "date": "2025-12-18",
      "version": 2,
      "time": "18:30",
      "article_count": 12
    },
    {
      "date": "2025-12-18",
      "version": 1,
      "time": "08:00",
      "article_count": 20
    },
    {
      "date": "2025-12-17",
      "version": 1,
      "time": "09:15",
      "article_count": 18
    }
  ]
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| total | int | 返回的条目数量 |
| dates | array | 版本信息列表 (扁平结构) |
| dates[].date | string | 日期 (YYYY-MM-DD) |
| dates[].version | int | 版本号 |
| dates[].time | string | 生成时间 (HH:MM) |
| dates[].article_count | int | 文章数量 |

### 示例

```bash
# 获取所有历史日期
curl http://localhost:8080/api/archive/dates

# 获取 2025 年 12 月的日期
curl "http://localhost:8080/api/archive/dates?year=2025&month=12"

# 分页获取
curl "http://localhost:8080/api/archive/dates?limit=10&offset=10"
```

---

## 5. 获取归档统计

获取归档的整体统计信息。

### 请求

```
GET /api/archive/stats
```

### 响应

```json
{
  "total_days": 45,
  "total_articles": 892,
  "date_range": {
    "start": "2025-11-01",
    "end": "2025-12-18"
  },
  "by_month": {
    "2025-12": {
      "days": 18,
      "articles": 360
    },
    "2025-11": {
      "days": 27,
      "articles": 532
    }
  }
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| total_days | int | 总天数 |
| total_articles | int | 总文章数 |
| date_range | object | 日期范围 |
| date_range.start | string | 最早日期 |
| date_range.end | string | 最新日期 |
| by_month | object | 按月统计 |
| by_month.{month}.days | int | 该月天数 |
| by_month.{month}.articles | int | 该月文章数 |

### 示例

```bash
curl http://localhost:8080/api/archive/stats
```

---

## 6. 手动触发流水线

手动触发流水线运行，用于即时生成日报。流水线在后台异步运行。

### 请求

```
POST /api/trigger
```

### 响应

**成功 (200)**

```json
{
  "status": "triggered",
  "message": "Pipeline started in background",
  "started_at": "2025-12-18T20:30:00.123456"
}
```

**冲突 (409) - 流水线正在运行**

```json
{
  "detail": "Pipeline is already running"
}
```

### 示例

```bash
# 触发流水线
curl -X POST http://localhost:8080/api/trigger

# 使用 httpie
http POST localhost:8080/api/trigger
```

---

## 7. 获取流水线状态

查询流水线当前运行状态和上次运行结果。运行时包含实时进度信息。

### 请求

```
GET /api/trigger/status
```

### 响应 (未运行时)

```json
{
  "running": false,
  "last_run": "2025-12-18T20:30:00.123456",
  "last_status": "success",
  "last_error": null,
  "last_duration_seconds": 125.5,
  "progress": null
}
```

### 响应 (运行中)

```json
{
  "running": true,
  "last_run": "2025-12-18T21:00:00.123456",
  "last_status": "running",
  "last_error": null,
  "last_duration_seconds": null,
  "progress": {
    "current_step": 5,
    "total_steps": 7,
    "step_name": "Filter",
    "step_desc": "语义过滤",
    "status": "running",
    "updated_at": "2025-12-18T21:01:30.123456",
    "input_count": 85,
    "output_count": 0
  }
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| running | bool | 是否正在运行 |
| last_run | string | 上次运行开始时间 (ISO 格式) |
| last_status | string | 上次运行状态: `running`, `success`, `failed` |
| last_error | string | 错误信息 (失败时) |
| last_duration_seconds | float | 上次运行耗时 (秒) |
| progress | object | 实时进度 (运行时有值，否则为 null) |

### Progress 对象

| 字段 | 类型 | 描述 |
|------|------|------|
| current_step | int | 当前步骤 (1-7) |
| total_steps | int | 总步骤数 (7) |
| step_name | string | 步骤名称 |
| step_desc | string | 步骤中文描述 |
| status | string | 步骤状态: `running`, `completed` |
| updated_at | string | 更新时间 (ISO 格式) |
| input_count | int | 输入数量 |
| output_count | int | 输出数量 |

### 步骤说明

| Step | Name | 描述 |
|------|------|------|
| 1 | Aggregate | 抓取 RSS/PubMed |
| 2 | Clean | HTML 清洗 |
| 3 | Dedup | 去重 |
| 4 | Enrich | 补充摘要 |
| 5 | Filter | 语义过滤 |
| 6 | LLM | 生成摘要 |
| 7 | Deliver | 输出结果 |

### 前端使用示例

```javascript
// 轮询进度
async function pollProgress() {
  const res = await fetch('/api/trigger/status');
  const data = await res.json();

  if (data.running && data.progress) {
    const pct = (data.progress.current_step / data.progress.total_steps) * 100;
    console.log(`${data.progress.step_desc} (${pct.toFixed(0)}%)`);
  }

  if (data.running) {
    setTimeout(pollProgress, 2000); // 每 2 秒轮询
  }
}
```

### 示例

```bash
# 查询状态
curl http://localhost:8080/api/trigger/status

# 轮询等待完成
while true; do
  status=$(curl -s http://localhost:8080/api/trigger/status | jq -r '.running')
  if [ "$status" = "false" ]; then
    echo "Pipeline finished"
    break
  fi
  # 显示进度
  curl -s http://localhost:8080/api/trigger/status | jq '.progress'
  sleep 5
done
```

---

## 8. 微信登录

小程序通过 `wx.login()` 获取 code，换取 JWT token 进行身份验证。

### 请求

```
POST /api/auth/wx-login
Content-Type: application/json

{
  "code": "033aXX000xxx"
}
```

### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| code | string | 是 | 小程序 wx.login() 返回的临时登录凭证 |

### 响应

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "openid": "oXXXX...",
  "is_new_user": false
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| token | string | JWT token，有效期 7 天 |
| openid | string | 用户的微信 openid |
| is_new_user | bool | 是否为新用户 |

### 错误响应

**400 Bad Request** - code 无效或微信接口调用失败

```json
{
  "detail": "微信登录失败: invalid code"
}
```

### 小程序端示例

```javascript
// 登录获取 token
async function login() {
  const { code } = await wx.login();
  const res = await wx.request({
    url: 'https://api.example.com/api/auth/wx-login',
    method: 'POST',
    data: { code }
  });

  if (res.data.token) {
    wx.setStorageSync('token', res.data.token);
  }
}
```

---

## 9. 获取当前用户信息

获取已登录用户的信息。

### 请求

```
GET /api/auth/me
Authorization: Bearer {token}
```

### 响应

```json
{
  "openid": "oXXXX...",
  "created_at": "2025-12-18T10:30:00",
  "last_login": "2025-12-18T20:00:00",
  "login_count": 5,
  "bookmark_count": 12
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| openid | string | 用户的微信 openid |
| created_at | string | 账号创建时间 |
| last_login | string | 最后登录时间 |
| login_count | int | 登录次数 |
| bookmark_count | int | 收藏数量 |

---

## 10. 获取收藏列表

获取当前用户的所有收藏，按保存时间倒序排列。返回文章基本信息便于直接展示。

### 请求

```
GET /api/bookmarks
Authorization: Bearer {token}
```

### 响应

```json
{
  "total": 2,
  "bookmarks": [
    {
      "article_id": "doi_10_1038_s41586-024-xxxxx",
      "saved_at": "2025-12-18T15:30:00",
      "note": "",
      "title": "Single-cell analysis reveals...",
      "journal": "Nature",
      "date": "2025-12-18"
    },
    {
      "article_id": "t_a1b2c3d4e5f6",
      "saved_at": "2025-12-17T10:00:00",
      "note": "很有参考价值",
      "title": "CRISPR-based genome editing...",
      "journal": "Science",
      "date": "2025-12-17"
    }
  ]
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| total | int | 收藏总数 |
| bookmarks | array | 收藏列表 |
| bookmarks[].article_id | string | 文章 ID |
| bookmarks[].saved_at | string | 收藏时间 (ISO 格式) |
| bookmarks[].note | string | 用户备注 |
| bookmarks[].title | string | 文章标题 |
| bookmarks[].journal | string | 期刊名称 |
| bookmarks[].date | string | 文章所属日期 (用于获取详情) |

### 示例

```bash
curl -H "Authorization: Bearer {token}" http://localhost:8080/api/bookmarks
```

---

## 11. 添加收藏

收藏一篇文章。需要传入文章基本信息以便收藏列表直接展示。

### 请求

```
POST /api/bookmarks/{article_id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "note": "可选的备注",
  "title": "文章标题",
  "journal": "Nature",
  "date": "2025-12-18"
}
```

### 路径参数

| 参数 | 类型 | 描述 |
|------|------|------|
| article_id | string | 文章 ID |

### 请求体

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| note | string | 否 | 收藏备注 |
| title | string | 推荐 | 文章标题 (用于收藏列表显示) |
| journal | string | 推荐 | 期刊名称 |
| date | string | 否 | 文章所属日期 (后端可通过索引自动查找) |

### 响应

```json
{
  "status": "success",
  "message": "收藏成功",
  "bookmark": {
    "article_id": "doi_10_1038_s41586-024-xxxxx",
    "saved_at": "2025-12-18T15:30:00",
    "note": "",
    "title": "Single-cell analysis reveals...",
    "journal": "Nature",
    "date": "2025-12-18"
  }
}
```

### 错误响应

**409 Conflict** - 已收藏该文章

```json
{
  "detail": "已收藏该文章"
}
```

### 示例

```bash
# 添加收藏（包含文章信息）
curl -X POST -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{"title": "文章标题", "journal": "Nature", "date": "2025-12-18"}' \
  http://localhost:8080/api/bookmarks/doi_10_1038_s41586-024-xxxxx

# 添加收藏（带备注）
curl -X POST -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{"title": "文章标题", "journal": "Nature", "date": "2025-12-18", "note": "重要参考"}' \
  http://localhost:8080/api/bookmarks/doi_10_1038_s41586-024-xxxxx
```

---

## 12. 取消收藏

取消收藏一篇文章。

### 请求

```
DELETE /api/bookmarks/{article_id}
Authorization: Bearer {token}
```

### 路径参数

| 参数 | 类型 | 描述 |
|------|------|------|
| article_id | string | 文章 ID |

### 响应

```json
{
  "status": "success",
  "message": "已取消收藏"
}
```

### 错误响应

**404 Not Found** - 未找到该收藏

```json
{
  "detail": "未找到该收藏"
}
```

### 示例

```bash
curl -X DELETE -H "Authorization: Bearer {token}" \
  http://localhost:8080/api/bookmarks/doi_10_1038_s41586-024-xxxxx
```

---

## 13. 检查收藏状态

检查某篇文章是否已被收藏。

### 请求

```
GET /api/bookmarks/{article_id}/status
Authorization: Bearer {token}
```

### 路径参数

| 参数 | 类型 | 描述 |
|------|------|------|
| article_id | string | 文章 ID |

### 响应

```json
{
  "article_id": "doi_10_1038_s41586-024-xxxxx",
  "is_bookmarked": true
}
```

### 响应字段

| 字段 | 类型 | 描述 |
|------|------|------|
| article_id | string | 文章 ID |
| is_bookmarked | bool | 是否已收藏 |

### 示例

```bash
curl -H "Authorization: Bearer {token}" \
  http://localhost:8080/api/bookmarks/doi_10_1038_s41586-024-xxxxx/status
```

---

## 认证说明

需要认证的接口必须在请求头中携带 JWT token：

```
Authorization: Bearer {token}
```

### Token 获取流程

```
小程序 wx.login() → code
    ↓
POST /api/auth/wx-login {code}
    ↓
获得 token，本地存储
    ↓
后续请求携带 Authorization header
```

### Token 过期处理

Token 有效期为 7 天。过期后请求会返回 401 错误，此时需要重新调用 wx.login() 获取新 token。

```json
{
  "detail": "Token 无效或已过期"
}
```

---

## 错误码

| HTTP 状态码 | 描述 |
|-------------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 401 | 未认证或 token 无效 |
| 404 | 资源未找到 |
| 409 | 冲突 (已收藏 / 流水线正在运行) |
| 422 | 请求参数验证失败 |
| 500 | 服务器内部错误 |

---

## 文章 ID 生成规则

文章 ID 基于以下规则生成:

1. **有 DOI**: `doi_{doi}` (将 `/` 和 `.` 替换为 `_`)
   - 例: `10.1038/s41586-024-12345` → `doi_10_1038_s41586-024-12345`

2. **无 DOI**: `t_{md5_hash}` (标题的 MD5 哈希前12位)
   - 例: `t_a1b2c3d4e5f6`

---

## 数据来源

API 数据来自以下文件 (按优先级):

| 优先级 | 文件路径 | 说明 |
|--------|----------|------|
| 1 | `output/archive/{year}/{month}/{day}/daily_v{N}.json` | 永久归档 (支持多版本) |
| 2 | `output/daily.json` | 最新日报输出 (兼容) |

### 归档目录结构

```
output/archive/
├── index.json                    # 全局日期索引
└── 2025/
    └── 12/
        └── 18/
            ├── daily_v1.json     # 第一次运行 (08:00)
            ├── daily_v1.html
            ├── daily_v1.md
            ├── daily_v2.json     # 第二次运行 (18:00)
            ├── daily_v2.html
            ├── daily_v2.md
            └── metadata.json     # 元数据
```

---

## OpenAPI 文档

FastAPI 自动生成的交互式文档:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI JSON**: http://localhost:8080/openapi.json
