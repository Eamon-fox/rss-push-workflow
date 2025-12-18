# ScholarPipe 后端架构文档

## 概述

ScholarPipe 是一个学术论文聚合和 AI 摘要生成系统，采用 7 步流水线架构，支持微信小程序用户系统。

---

## 流水线步骤

| Step | 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|------|
| 1 | S1_aggregate | 抓取 RSS/PubMed | sources.yaml | NewsItem[] |
| 2 | S2_clean | HTML 清洗 | NewsItem[] | NewsItem[] |
| 3 | S3_dedup | 去重 | NewsItem[] | NewsItem[] |
| 4 | S5_enrich | 补充摘要 | NewsItem[] | NewsItem[] |
| 5 | S4_filter | 语义过滤 | NewsItem[] | NewsItem[] |
| 6 | S6_llm | 生成中文摘要 | NewsItem[] | NewsItem[] |
| 7 | S7_deliver | 输出结果 | NewsItem[] | JSON/HTML/MD |

---

## 缓存机制

### RSS/PubMed 数据源缓存

为避免短时间内重复请求外部数据源，系统对 RSS 和 PubMed 抓取结果进行缓存。

**配置**

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `CACHE_MAX_AGE_HOURS` | `src/S1_aggregate/rss.py` | 5 | RSS 缓存有效期（小时） |
| `CACHE_MAX_AGE_HOURS` | `src/S1_aggregate/pubmed.py` | 5 | PubMed 缓存有效期（小时） |

**工作原理**

1. 检查 `data/raw/{date}/{source}.json` 是否存在
2. 检查文件修改时间是否在 `CACHE_MAX_AGE_HOURS` 内
3. 如果缓存有效 → 直接读取，跳过网络请求
4. 如果缓存无效/不存在 → 发起请求，保存结果

**日志输出**

```
# 缓存命中
[Nature] Cache hit (70 items, 45min ago)

# 缓存未命中，正常抓取
[Nature] 70 items
```

**注意事项**

- 同一天内多次触发 pipeline，如果间隔 < 5 小时，数据源不会重新抓取
- 这不影响去重逻辑，已处理的文章仍会被 seen 记录过滤
- 如需强制刷新，可删除对应的缓存文件：`rm data/raw/2025-12-18/nature.json`

### Embedding 缓存

语义过滤使用的文本向量会缓存到 SQLite 数据库。

| 文件 | 说明 |
|------|------|
| `data/embedding_cache.db` | 文本 → 向量映射缓存 |

### LLM 摘要缓存

LLM 生成的中文摘要会缓存，避免重复调用。

| 文件 | 说明 |
|------|------|
| `data/llm_cache.db` | 原文 → 摘要映射缓存 |

---

## 数据清理策略

### 中间数据

每次 pipeline 运行结束后，会清理当天的中间处理数据：

| 目录 | 清理时机 | 说明 |
|------|----------|------|
| `data/cleaned/{date}` | 每次运行后 | HTML 清洗结果 |
| `data/deduped/{date}` | 每次运行后 | 去重结果 |
| `data/filtered/{date}` | 每次运行后 | 过滤结果 |

### Raw 数据（缓存）

Raw 数据作为缓存保留，但会清理过期数据：

| 目录 | 保留期限 | 说明 |
|------|----------|------|
| `data/raw/{date}` | 2 天 | RSS/PubMed 抓取缓存 |

配置位置：`src/cleanup.py` 中的 `RAW_DATA_RETENTION_DAYS`

### 受保护文件

以下文件永远不会被清理：

- `data/embedding_cache.db` - 向量缓存
- `data/llm_cache.db` - LLM 摘要缓存
- `data/seen.json` - 去重记录
- `data/users.json` - 用户数据
- `data/bookmarks.json` - 收藏数据
- `data/article_index.json` - 文章索引

### 永久存储

以下目录永久保存，不会被清理（收藏功能依赖）：

- `output/archive/` - 归档数据，收藏的文章需要能访问到

---

## 进度追踪

Pipeline 运行时会写入进度文件，供 API 查询。

**进度文件**

```
data/pipeline_progress.json
```

**文件格式**

```json
{
  "current_step": 5,
  "total_steps": 7,
  "step_name": "Filter",
  "step_desc": "语义过滤",
  "status": "running",
  "updated_at": "2025-12-18T21:01:30.123456",
  "input_count": 85,
  "output_count": 0
}
```

**生命周期**

1. 每个步骤开始时更新 `status: "running"`
2. 每个步骤结束时更新 `status: "completed"` 和计数
3. Pipeline 完成后删除文件

---

## 去重机制

### Seen 记录

已处理的文章通过 fingerprint 记录到 `data/seen.json`。

**Fingerprint 生成规则**

1. 有 DOI → 使用 DOI
2. 无 DOI → 使用标题 + 来源的 MD5 哈希

**记录时机**

| 场景 | 记录位置 |
|------|----------|
| 通过所有过滤的文章 | Step 7 结束时 |
| 被 Layer2 语义过滤的文章 | Step 5 Filter 内部 |

**清理策略**

Seen 记录保留 7 天，超过后自动清理。

配置位置：`src/S3_dedup/seen.py` 中的 `MAX_AGE_DAYS`

---

## 归档机制

每次 pipeline 运行成功后，结果会归档到版本化目录。

**目录结构**

```
output/archive/
├── index.json
└── 2025/
    └── 12/
        └── 18/
            ├── daily_v1.json
            ├── daily_v1.html
            ├── daily_v1.md
            ├── daily_v2.json
            ├── daily_v2.html
            ├── daily_v2.md
            └── metadata.json
```

**版本号规则**

- 同一天每次运行自动递增版本号
- metadata.json 记录每个版本的创建时间和统计信息

---

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NCBI_EMAIL` | PubMed API 邮箱 | - |
| `NCBI_API_KEY` | PubMed API Key | - |
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | - |
| `AGGREGATE_MAX_WORKERS` | 抓取并发数 | 源数量 |
| `SEMANTIC_PRESET` | 语义模型预设 (dev/prod/api) | dev |

---

## 目录结构

```
data/
├── raw/                    # RSS/PubMed 抓取缓存 (保留 2 天)
│   └── {date}/
│       ├── nature.json
│       ├── science.json
│       └── all.json
├── cleaned/                # HTML 清洗结果 (每次清理)
├── deduped/                # 去重结果 (每次清理)
├── filtered/               # 过滤结果 (每次清理)
├── embedding_cache.db      # 向量缓存 (永久)
├── llm_cache.db            # LLM 缓存 (永久)
├── seen.json               # 去重记录 (保留 7 天)
└── pipeline_progress.json  # 进度文件 (运行时)

output/
├── daily.json              # 最新输出
├── daily.html
├── daily.md
└── archive/                # 归档 (永久)
    └── {year}/{month}/{day}/
```

---

## 用户认证系统

### 概述

基于微信小程序静默登录 + JWT 的认证系统，用户无需手动授权即可使用收藏等功能。

### 认证流程

```
┌────────────┐     ┌──────────────┐     ┌────────────┐
│  小程序    │────►│   后端 API   │────►│  微信服务器 │
│ wx.login() │     │ /auth/wx-login│     │ code2session│
└────────────┘     └──────────────┘     └────────────┘
      │                   │                    │
      │    code           │                    │
      │──────────────────►│    code + secret   │
      │                   │───────────────────►│
      │                   │                    │
      │                   │◄────────openid─────│
      │                   │                    │
      │◄──JWT token───────│                    │
      │                   │                    │
```

### 模块文件

| 文件 | 功能 |
|------|------|
| `src/auth.py` | 微信登录 + JWT 生成/验证 |
| `src/bookmarks.py` | 收藏功能 CRUD |

### JWT 配置

| 配置项 | 值 | 说明 |
|--------|---|------|
| 算法 | HS256 | HMAC-SHA256 |
| 有效期 | 7 天 | 过期需重新登录 |
| Payload | openid, iat, exp | 用户标识 + 签发/过期时间 |

### 数据存储

**用户数据 (data/users.json)**

```json
{
  "openid_xxx": {
    "created_at": "2025-12-18T10:30:00",
    "last_login": "2025-12-18T20:00:00",
    "login_count": 5
  }
}
```

**收藏数据 (data/bookmarks.json)**

```json
{
  "openid_xxx": {
    "doi_10_1038_xxx": {
      "saved_at": "2025-12-18T15:30:00",
      "note": ""
    }
  }
}
```

### API 端点

| 端点 | 方法 | 认证 | 功能 |
|------|------|------|------|
| `/api/auth/wx-login` | POST | 无 | 微信登录，返回 JWT |
| `/api/auth/me` | GET | 需要 | 获取用户信息 |
| `/api/bookmarks` | GET | 需要 | 获取收藏列表 |
| `/api/bookmarks/{id}` | POST | 需要 | 添加收藏 |
| `/api/bookmarks/{id}` | DELETE | 需要 | 取消收藏 |
| `/api/bookmarks/{id}/status` | GET | 需要 | 检查收藏状态 |

### 认证中间件

需认证的接口通过 `get_current_user` 依赖验证：

```python
async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> str:
    # 解析 Authorization: Bearer {token}
    # 验证 JWT，返回 openid
    # 失败抛出 401 HTTPException
```

### 安全注意事项

1. **JWT Secret**: 生产环境应使用环境变量配置，不要硬编码
2. **微信 AppSecret**: 已配置在 `config/settings.yaml`，生产环境建议使用环境变量
3. **HTTPS**: 生产环境必须使用 HTTPS 传输 token
4. **Token 刷新**: 当前采用简单的过期重新登录策略

---

## 文章索引机制

### 概述

为支持通过 article_id 直接查找文章（无需 date 参数），系统维护一个全局索引。

### 索引文件

```
data/article_index.json
```

**格式**

```json
{
  "doi_10_1038_xxx": {
    "date": "2025-12-18",
    "version": 1
  },
  "t_a1b2c3d4e5f6": {
    "date": "2025-12-17",
    "version": 2
  }
}
```

### 工作流程

1. **归档时自动更新**: `archive_daily()` 执行后自动将文章 ID 添加到索引
2. **查询时使用**: `/api/articles/{id}` 不传 date 时，从索引获取文章位置
3. **重建索引**: 可运行 `rebuild_index_from_archive()` 从归档目录重建完整索引

### 模块文件

| 文件 | 功能 |
|------|------|
| `src/article_index.py` | 索引管理 (增删查) |
| `src/archive.py` | 归档时更新索引 |

### 命令行工具

```bash
# 重建索引
python -c "from src.article_index import rebuild_index_from_archive; print(rebuild_index_from_archive())"

# 查看索引统计
python -c "from src.article_index import get_index_stats; print(get_index_stats())"
```

---

## 依赖说明

### 用户系统新增依赖

| 包 | 版本 | 用途 |
|----|------|------|
| PyJWT | >=2.0.0 | JWT 生成/验证 |
| httpx | (已有) | 调用微信 API |
