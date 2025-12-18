# ScholarPipe 开发文档

> 学术资讯聚合 + AI 摘要日报系统

---

## 项目状态

**当前版本**: v0.3.0
**最后更新**: 2025-12-18

### 已完成功能

| 模块 | 功能 | 状态 |
|------|------|------|
| Pipeline | 7 步流水线完整实现 | ✅ |
| 数据源 | 75 个 RSS + PubMed 来源 | ✅ |
| 解析器 | 7 种差异化解析器 | ✅ |
| 过滤 | 双层混合过滤 (关键词 + 语义) | ✅ |
| LLM | 中文摘要生成 (多 provider) | ✅ |
| API | RESTful 后端接口 | ✅ |
| 缓存 | RSS/PubMed 5h 缓存 | ✅ |
| 缓存 | Embedding + LLM 缓存 | ✅ |
| 进度 | Pipeline 实时进度追踪 | ✅ |
| 归档 | 多版本归档系统 | ✅ |
| 部署 | systemd 服务 | ✅ |
| 认证 | 微信小程序静默登录 | ✅ |
| 收藏 | 文章收藏功能 | ✅ |

---

## 系统架构

### 流水线

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐
│ 1_aggregate │──►│  2_clean    │──►│  3_dedup    │──►│  4_enrich  │
│   (聚合)    │   │  (清洗)     │   │ (指纹去重)  │   │ (补充摘要) │
└─────────────┘   └─────────────┘   └─────────────┘   └────────────┘
       │                                                     │
       ▼                                                     ▼
  [5h 缓存]                                            [PubMed API]
                                                             │
┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  7_deliver  │◄──│   6_llm     │◄──│  5_filter   │◄─────────┘
│   (输出)    │   │  (AI摘要)   │   │ (语义过滤)  │
└─────────────┘   └─────────────┘   └─────────────┘
       │                │                  │
       ▼                ▼                  ▼
   [Archive]      [LLM Cache]      [Embedding Cache]
```

### 数据流漏斗

```
~500 items (75 来源)
    │
    ▼ [5h 内复用缓存]
3_dedup: 指纹去重 + seen 过滤
    │
    ▼ ~100 items (去重后)
5_filter Layer1: 生物学关键词
    │
    ▼ ~60 items
5_filter Layer2: 语义相似度 + VIP
    │
    ▼ ~30 items (threshold 0.50)
6_llm: Top 20 生成摘要
    │
    ▼
7_deliver: JSON + HTML + MD + Archive
```

---

## API 系统

**Base URL**: `http://localhost:8080`

### 公开接口

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/daily` | GET | 获取日报 |
| `/api/articles/{id}` | GET | 文章详情 (支持历史日期) |
| `/api/archive/dates` | GET | 历史版本列表 (扁平) |
| `/api/archive/stats` | GET | 归档统计 |
| `/api/trigger` | POST | 触发 Pipeline |
| `/api/trigger/status` | GET | 运行状态 + 实时进度 |
| `/api/auth/wx-login` | POST | 微信登录 |

### 需认证接口

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/auth/me` | GET | 获取当前用户信息 |
| `/api/bookmarks` | GET | 获取收藏列表 |
| `/api/bookmarks/{id}` | POST | 添加收藏 |
| `/api/bookmarks/{id}` | DELETE | 取消收藏 |
| `/api/bookmarks/{id}/status` | GET | 检查收藏状态 |

详细文档: [API.md](API.md)

---

## 缓存策略

| 缓存类型 | 位置 | TTL | 说明 |
|----------|------|-----|------|
| RSS/PubMed | `data/raw/{date}/` | 5h | 避免频繁请求数据源 |
| Embedding | `data/embedding_cache.db` | 永久 | 文本向量缓存 |
| LLM | `data/llm_cache.db` | 永久 | 摘要结果缓存 |
| Seen | `data/seen.json` | 7 天 | 去重指纹记录 |

详细文档: [docs/Backend.md](docs/Backend.md)

---

## 下一步开发计划

### Phase 1: 小程序前端 (MVP) - 开发中

**目标**: 微信小程序展示日报

| 任务 | 优先级 | 状态 | 说明 |
|------|--------|------|------|
| 首页列表 | P0 | 🔄 | 展示当日文章，按分数排序 |
| 文章详情 | P0 | 🔄 | 标题 + 摘要 + 原文链接 |
| 历史日期 | P1 | 🔄 | 选择日期查看历史日报 |
| 收藏功能 | P1 | 🔄 | 收藏/取消收藏文章 |
| 版本切换 | P2 | ⏳ | 同一天多版本切换 |
| 下拉刷新 | P1 | ⏳ | 刷新当日数据 |

**技术栈**: 微信小程序原生

### Phase 2: 用户系统 - 后端已完成

**目标**: 个性化推荐基础

| 任务 | 优先级 | 状态 | 说明 |
|------|--------|------|------|
| 静默登录 | P1 | ✅ | wx.login() + JWT |
| 收藏功能 | P1 | ✅ | 收藏/取消收藏/列表 |
| 反馈收集 | P1 | ⏳ | 点赞/点踩 |
| 阅读历史 | P2 | ⏳ | 记录已读文章 |

**认证流程**

```
小程序 wx.login() → code
    ↓
POST /api/auth/wx-login {code}
    ↓
后端调用微信接口换取 openid
    ↓
生成 JWT token (7 天有效)
    ↓
小程序存储 token
    ↓
后续请求 Authorization: Bearer {token}
```

**数据存储**

| 文件 | 说明 |
|------|------|
| `data/users.json` | 用户信息 (openid, 登录时间, 次数) |
| `data/bookmarks.json` | 收藏记录 (按 openid 分组) |

### Phase 3: 用户互动增强

**目标**: 丰富用户交互功能

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 点赞/点踩反馈 | P1 | 文章评价，用于改进推荐 |
| 阅读历史 | P2 | 记录已读文章，避免重复推荐 |
| 收藏夹分类 | P2 | 支持多个收藏夹 |
| 备注编辑 | P3 | 编辑收藏备注 |

### Phase 4: 个性化推荐

**目标**: 基于用户行为的推荐

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 用户画像 | P2 | 基于收藏/反馈构建兴趣向量 |
| 个性化排序 | P2 | 文章按用户兴趣重排序 |
| 关键词订阅 | P2 | 用户自定义 VIP 关键词 |
| 推送通知 | P3 | VIP 关键词命中时推送 |

### Phase 5: 运维增强

**目标**: 稳定性和可观测性

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 监控告警 | P2 | Pipeline 失败告警 |
| 日志分析 | P3 | 结构化日志 + 统计 |
| 定时任务优化 | P2 | cron 多时段触发策略 |
| 数据备份 | P3 | 自动备份 seen.json + cache |
| 用户数据导出 | P3 | 支持用户导出个人数据 |

---

## 技术债务

| 问题 | 影响 | 优先级 |
|------|------|--------|
| 测试覆盖不足 | 回归风险 | P2 |
| 错误处理不完善 | 部分异常未捕获 | P2 |
| 配置分散 | 多处硬编码常量 | P3 |
| 日志不够结构化 | 难以分析 | P3 |

---

## 配置文件

| 文件 | 说明 |
|------|------|
| `config/settings.yaml` | LLM + 语义模型 + 微信配置 |
| `config/filter.yaml` | 过滤阈值 + VIP 关键词 |
| `config/semantic_anchors.yaml` | 语义锚点 (分层) |
| `config/user_profile.yaml` | LLM prompt 模板 |
| `src/S1_aggregate/sources.yaml` | 数据源配置 |

### 微信配置 (config/settings.yaml)

```yaml
wechat:
  appid: wx1a609eecc9dc6972
  secret: f5b837d3b310ed986804bbbf7cabd4c0
```

---

## 运行命令

```bash
# 完整 Pipeline
python main.py

# 跳过抓取 (使用缓存)
python main.py --skip-fetch

# 跳过 LLM (测试用)
python main.py --no-llm

# 指定输出数量
python main.py --top 30

# 启动 API 服务
python api.py

# 服务管理
sudo systemctl start scholarpipe-api
sudo systemctl status scholarpipe-api
```

---

## 目录结构

```
rss-push-workflow/
├── main.py                    # Pipeline 入口
├── api.py                     # API 服务
├── src/
│   ├── S1_aggregate/          # 步骤1: 聚合
│   ├── S2_clean/              # 步骤2: 清洗
│   ├── S3_dedup/              # 步骤3: 去重
│   ├── S4_filter/             # 步骤4: 过滤
│   ├── S5_enrich/             # 步骤5: 补充
│   ├── S6_llm/                # 步骤6: LLM
│   ├── S7_deliver/            # 步骤7: 输出
│   ├── auth.py                # 微信登录 + JWT 认证
│   ├── bookmarks.py           # 收藏功能 CRUD
│   ├── archive.py             # 归档模块
│   ├── cleanup.py             # 清理模块
│   └── infra/                 # 基础设施
├── config/                    # 配置文件
├── data/                      # 数据目录
│   ├── users.json             # 用户数据
│   └── bookmarks.json         # 收藏数据
├── output/                    # 输出目录
├── logs/                      # 日志目录
└── docs/                      # 文档
    ├── API.md                 # API 文档
    └── Backend.md             # 后端文档
```

---

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `NCBI_EMAIL` | 是 | PubMed API 邮箱 |
| `NCBI_API_KEY` | 否 | PubMed API Key (提高限额) |
| `DASHSCOPE_API_KEY` | 条件 | DashScope Embedding API |
| `ZHIPU_API_KEY` | 条件 | 智谱 LLM API |
| `SILICONFLOW_API_KEY` | 条件 | SiliconFlow LLM API |
