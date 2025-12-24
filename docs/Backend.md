# ScholarPipe 后端架构文档

## 概述

ScholarPipe 是一个学术论文聚合和 AI 摘要生成系统，采用 7 步流水线架构，支持微信小程序用户系统。

---

## 流水线步骤

| Step | 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|------|
| 1 | S1_aggregate | 抓取 RSS/PubMed/API | sources.yaml | NewsItem[] |
| 2 | S2_clean | HTML 清洗 | NewsItem[] | NewsItem[] |
| 3 | S3_dedup | 去重 | NewsItem[] | NewsItem[] |
| 4 | S5_enrich | 补充摘要 | NewsItem[] | NewsItem[] |
| 5 | S4_filter | 语义过滤 | NewsItem[] | NewsItem[] |
| 6 | S6_llm | 生成中文摘要 | NewsItem[] | NewsItem[] |
| 7 | S7_deliver | 输出结果 | NewsItem[] | JSON/HTML/MD |

### 数据源类型

| 类型 | 模块 | 说明 |
|------|------|------|
| RSS | `rss.py` | 传统 RSS 订阅 (5-7 天窗口) |
| PubMed | `pubmed.py` | NCBI E-utilities 搜索 |
| bioRxiv API | `biorxiv_api.py` | bioRxiv/medRxiv 官方 API (30 天窗口) |
| OpenAlex | `openalex.py` | 开放学术数据库 (2.5B+ 文献) |
| Europe PMC | `europepmc.py` | 欧洲生命科学数据库 (预印本 + PubMed) |

---

## 缓存机制

### 数据源缓存

为避免短时间内重复请求外部数据源，系统对所有抓取结果进行缓存。

**配置**

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `CACHE_MAX_AGE_HOURS` | `src/S1_aggregate/rss.py` | 5 | RSS 缓存有效期（小时） |
| `CACHE_MAX_AGE_HOURS` | `src/S1_aggregate/pubmed.py` | 5 | PubMed 缓存有效期（小时） |
| `CACHE_MAX_AGE_HOURS` | `src/S1_aggregate/biorxiv_api.py` | 5 | bioRxiv API 缓存 |
| `CACHE_MAX_AGE_HOURS` | `src/S1_aggregate/openalex.py` | 5 | OpenAlex API 缓存 |
| `CACHE_MAX_AGE_HOURS` | `src/S1_aggregate/europepmc.py` | 5 | Europe PMC API 缓存 |

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

语义过滤使用的文本向量会缓存到 SQLite 数据库，支持跨源 DOI 查找。

| 文件 | 说明 |
|------|------|
| `data/embedding_cache.db` | 文本向量缓存 |

**表结构**

| 表名 | 主键 | 说明 |
|------|------|------|
| `embedding_cache` | text_hash | 文本 SHA256 前 16 位 → 向量 |
| `doi_cache_map` | doi | DOI → text_hash 关联表 |

**DOI 跨源缓存**

同一篇论文可能从多个来源获取（如 bioRxiv RSS + Europe PMC API），通过 DOI 关联实现跨源缓存命中：

1. 首次计算向量时，存入 `embedding_cache` (text_hash 为 key)
2. 如果文章有 DOI，同时在 `doi_cache_map` 注册 DOI → text_hash 映射
3. 下次遇到相同 DOI 的文章，直接通过 DOI 查找向量，无需重新计算

**优先级规则**

如果同一文本有多个模型的缓存，按以下优先级返回：
- Cloud API (DashScope, OpenAI 等): 100
- 高质量本地模型 (bge-m3, bge-large): 50
- 轻量本地模型 (MiniLM): 10

### LLM 摘要缓存

LLM 生成的中文摘要会缓存，以 DOI 或内容哈希为 key，避免重复调用。

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
| `data/raw/{date}` | 2 天 | RSS/API 抓取缓存 |

配置位置：`src/cleanup.py` 中的 `RAW_DATA_RETENTION_DAYS`

### 日志文件

日志文件会自动清理：

| 目录 | 保留期限 | 说明 |
|------|----------|------|
| `logs/{date}.log` | 30 天 | 每日运行日志 |

配置位置：`src/cleanup.py` 中的 `LOG_RETENTION_DAYS`

### 受保护文件

以下文件永远不会被清理：

- `data/embedding_cache.db` - 向量缓存
- `data/llm_cache.db` - LLM 摘要缓存
- `data/seen/` - 去重记录 (按用户隔离)
- `data/users.json` - 用户数据
- `data/bookmarks.json` - 收藏数据
- `data/article_index.json` - 文章索引
- `data/user_configs/` - 用户个性化配置

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

### Seen 记录 (按用户隔离)

已处理的文章通过 fingerprint 记录，支持按用户隔离。

**存储位置**

```
data/seen/
├── default.json      # 默认用户 (定时任务)
├── {openid1}.json    # 微信用户 1
└── {openid2}.json    # 微信用户 2
```

**Fingerprint 生成规则**

1. 有 DOI → 使用 DOI
2. 无 DOI → 使用标题 + 来源的 MD5 哈希

**记录时机**

| 场景 | 记录位置 |
|------|----------|
| 通过所有过滤的文章 | Step 7 结束时 |
| 被 Layer2 语义过滤的文章 | Step 5 Filter 内部 |

**用户隔离**

| 场景 | 用户标识 |
|------|----------|
| 定时任务 (cron) | `default` |
| 命令行指定 | `--user {user_id}` |
| API 个性化请求 | 微信 `openid` |

```bash
# 为特定用户运行 pipeline
python main.py --user user123

# 默认用户
python main.py
```

**清理策略**

Seen 记录保留 7 天，超过后自动清理。

配置位置：`src/S3_dedup/seen.py` 中的 `MAX_AGE_DAYS`

**迁移机制**

系统自动将旧的 `data/seen.json` 迁移到新格式 `data/seen/default.json`，原文件备份为 `data/seen.json.bak`。

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
├── seen/                   # 去重记录 (保留 7 天，按用户隔离)
│   ├── default.json        # 默认用户 (定时任务)
│   └── {openid}.json       # 微信用户
├── user_configs/           # 用户个性化配置 (永久)
│   └── {openid}.json
└── pipeline_progress.json  # 进度文件 (运行时)

output/
├── daily.json              # 最新输出
├── daily.html
├── daily.md
└── archive/                # 归档 (永久)
    └── {year}/{month}/{day}/
        ├── daily_v1.json   # 日报归档
        └── candidates.json # 候选池 (供个性化)
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
      "note": "",
      "title": "文章标题",
      "journal": "Nature",
      "date": "2025-12-18"
    }
  }
}
```

### 自动补充文章信息

添加收藏时，如果前端未传递 `title/journal/date`，后端会自动从文章索引查找并补充：

1. 通过 `article_id` 查询 `article_index.json` 获取文章位置
2. 从对应的归档文件加载文章数据
3. 自动填充 `title`、`journal`、`date` 字段

这确保收藏列表始终显示完整的文章信息，无需前端额外处理。

### API 端点

| 端点 | 方法 | 认证 | 功能 |
|------|------|------|------|
| `/api/auth/wx-login` | POST | 无 | 微信登录，返回 JWT |
| `/api/auth/me` | GET | 需要 | 获取用户信息 |
| `/api/bookmarks` | GET | 需要 | 获取收藏列表 |
| `/api/bookmarks/{id}` | POST | 需要 | 添加收藏 (自动补充文章信息) |
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

## 个性化推荐系统

### 概述

基于用户配置的个性化推荐系统，允许用户自定义语义锚点和 VIP 关键词，实现个性化文章排序。

### 架构设计

```
Pipeline 运行:
    Layer1 过滤 (关键词)
         │
         ├─────────────────────────► [Candidate Pool]
         │                            保存到 archive/{date}/candidates.json
         ▼
    Layer2 过滤 (语义评分)
         │
         ▼
    默认日报 (Top 20)

个性化请求 (/api/daily?personalized=true):
    │
    ▼
Candidate Pool (~100 条)
    │
    ▼ 加载用户配置
PersonalizedRanker
    │ - 用户语义锚点相似度
    │ - 用户 VIP 关键词加成
    │ - 负向锚点惩罚
    ▼
Top 20 (用户视角排序)
    │
    ▼ 按需生成 LLM 摘要 (缓存优先)
返回结果
```

### 候选池 (Candidate Pool)

通过 Layer1 关键词过滤的文章会保存到候选池，供不同用户的个性化排序使用。

**存储位置**

```
output/archive/{date}/candidates.json
```

**格式**

```json
[
  {
    "id": "doi_10_1038_xxx",
    "title": "文章标题",
    "content": "摘要内容",
    "doi": "10.1038/xxx",
    "link": "https://...",
    "source": "Nature",
    "pubdate": "2025-12-18",
    "default_score": 0.72
  }
]
```

**模块文件**

| 文件 | 功能 |
|------|------|
| `src/archive.py` | `archive_candidate_pool()` / `load_candidate_pool()` / `load_historical_candidates()` |

### 历史候选池

个性化请求会加载最近 1 年（365 天）的候选池，并按 `article_id` 去重合并。

**工作原理**

```python
def load_historical_candidates(days: int = 365) -> list[dict]:
    """
    加载并合并历史候选池（按 article_id 去重）。

    - 扫描 output/archive/{year}/{month}/{day}/candidates.json
    - 按日期倒序加载，最新文章优先
    - 相同 article_id 只保留首次出现（最新版本）
    - 返回去重后的文章列表
    """
```

**优势**

1. **新用户友好**: 新用户首次访问即可浏览过去一年相关文章
2. **高效查询**: 已归档的候选池是预计算结果，无需重新抓取/过滤
3. **自动去重**: 同一篇文章可能多天重复出现，通过 ID 去重

### 用户配置

每个用户可以自定义个性化参数，存储在独立的 JSON 文件中。

**存储位置**

```
data/user_configs/{openid}.json
```

**配置结构**

```json
{
  "openid": "用户 OpenID",
  "vip_keywords": {
    "tier1": {"multiplier": 1.50, "patterns": ["CRISPR", "gene therapy"]},
    "tier2": {"multiplier": 1.30, "patterns": ["immunotherapy"]},
    "tier3": {"multiplier": 1.15, "patterns": ["cancer"]}
  },
  "semantic_anchors": {
    "positive": ["基因编辑技术突破", "新型癌症治疗方法"],
    "negative": ["农业应用", "临床试验招募"]
  },
  "scoring_params": {
    "vip_max_multiplier": 1.80,
    "negative_threshold": 0.38,
    "negative_penalty": 0.60
  }
}
```

**模块文件**

| 文件 | 功能 |
|------|------|
| `src/user_config.py` | 用户配置 CRUD (load/save/update) |

### 锚点安全过滤

保存用户配置时会自动进行安全过滤：

| 过滤规则 | 限制值 | 说明 |
|----------|--------|------|
| 长度截断 | 3000 字符 | 单条锚点超出自动截断 |
| 数量限制 | 50 条 | 超出丢弃最旧的（保留最新） |
| 自动去重 | - | 相同内容只保留首次出现 |

**收藏与锚点的关系**

- 收藏夹 (`bookmarks.json`)：无数量限制
- 语义锚点 (`user_configs/{openid}.json`)：最多 50 条

前端收藏文章时可自动同步文章 content 到锚点。用户收藏超过 50 篇时，旧的锚点会被移除，但收藏记录保留。

**API 返回 hints**

更新锚点的 API 会返回清理信息，前端可据此提示用户：

```json
{
  "hints": {
    "positive": {"truncated": 0, "deduped": 1, "dropped": 2},
    "negative": {"truncated": 0, "deduped": 0, "dropped": 0}
  }
}
```

### 个性化评分引擎

**评分公式**

```
personalized_score = base_score × vip_mult × neg_penalty

- base_score: 文章与用户正向锚点的最大余弦相似度 (0-1)
- vip_mult: VIP 关键词加成 (tier1=1.50, tier2=1.30, tier3=1.15)
- neg_penalty: 负向锚点惩罚 (相似度 > 0.38 时降 40%)
```

**特殊情况**

- 用户无自定义配置 → 使用默认分数 (`default_score`)
- 用户无正向锚点但有 VIP 关键词 → 仅应用 VIP 加成

**模块文件**

| 文件 | 功能 |
|------|------|
| `src/personalize.py` | `PersonalizedRanker` 类 + `rerank_for_user()` |

### 缓存策略

| 数据 | 缓存位置 | 说明 |
|------|----------|------|
| 文章向量 | `embedding_cache.db` | 所有用户共享，按文章缓存 |
| LLM 摘要 | `llm_cache.db` | 所有用户共享，按需生成 |
| 用户锚点向量 | 内存 (会话内) | PersonalizedRanker 实例缓存 |

**按需 LLM**

个性化结果的 LLM 摘要采用"缓存优先 + 按需生成"策略：

1. 检查 `llm_cache.db` 是否已有摘要
2. 缓存命中 → 直接使用
3. 缓存未命中 → 调用 LLM API 生成并缓存

### API 端点

| 端点 | 方法 | 认证 | 功能 |
|------|------|------|------|
| `/api/daily?personalized=true` | GET | 需要 | 个性化日报 |
| `/api/config` | GET | 需要 | 获取用户配置 |
| `/api/config` | PUT | 需要 | 完整更新配置 |
| `/api/config/vip-keywords` | PATCH | 需要 | 更新 VIP 关键词 |
| `/api/config/semantic-anchors` | PATCH | 需要 | 更新语义锚点 |
| `/api/config/reset` | POST | 需要 | 重置为默认配置 |
| `/api/config/defaults` | GET | 需要 | 获取默认配置值 |

### 使用示例

**1. 设置语义锚点**

```bash
curl -X PATCH "http://localhost:8080/api/config/semantic-anchors" \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "positive": ["基因编辑治疗遗传病", "CAR-T 细胞疗法"],
    "negative": ["农业基因改良", "动物模型"]
  }'
```

**2. 获取个性化日报**

```bash
curl "http://localhost:8080/api/daily?personalized=true" \
  -H "Authorization: Bearer {token}"
```

---

## 依赖说明

### 用户系统新增依赖

| 包 | 版本 | 用途 |
|----|------|------|
| PyJWT | >=2.0.0 | JWT 生成/验证 |
| httpx | (已有) | 调用微信 API |
