# ScholarPipe 开发文档

> 学术资讯聚合 + AI 摘要早报

---

## 核心理念

- **把论文当新闻对待**
- 重点是**聚合**多源 + **清晰**呈现
- 用户感兴趣自己点链接看原文

---

## 工作流架构 (7 步)

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐   ┌────────────┐   ┌─────────────┐   ┌─────────────┐
│ 1_aggregate │──►│  2_clean    │──►│  3_dedup    │──►│  4_filter  │──►│  5_enrich  │──►│   6_llm     │──►│  7_deliver  │
│   (聚合)    │   │  (清洗)     │   │ (指纹去重)  │   │(混合过滤)  │   │ (PubMed)   │   │  (摘要)     │   │   (输出)    │
└─────────────┘   └─────────────┘   └─────────────┘   └────────────┘   └────────────┘   └─────────────┘   └─────────────┘
       │                                   │                 │                │                │
  sources.yaml                        seen.json      filter.yaml +      DOI→Abstract      LLM API
  parsers/                          (已检阅记录)   semantic_anchors                    (评分+摘要)
```

### 分层过滤漏斗

```
~500 items (采集自75个来源)
    ↓
3_dedup: 指纹比对
    ↓ 去除重复
~400 items
    ↓
4_filter Layer1: 生物学关键词粗筛
    ↓ 过滤非生物学文章
~250 items (~60%)
    ↓
4_filter Layer2: 语义相似度 + VIP关键词
    ↓ 按研究方向打分
~50 items (threshold 0.50)
    ↓
5_enrich: PubMed补充摘要 (可选)
    ↓
6_llm: AI 摘要生成
    ↓
7_deliver: 输出日报
```

---

## 第4步：混合过滤 (Hybrid Filter)

### 两层过滤

**Layer 1: 生物学关键词粗筛**
- 200个生物学词根 (config/filter.yaml)
- 快速过滤明显无关文章

**Layer 2: 语义 + VIP 加权**
- 向量相似度：文章 vs semantic_anchors.json
- VIP关键词分层加权 (tier1: ×1.50, tier2: ×1.30, tier3: ×1.15)
- 最终分数 = max_similarity × tier_multiplier × vip_multiplier + coverage_bonus

### 向量模型配置 (config/settings.yaml)

```yaml
semantic:
  preset: dev | prod | api | custom
  # dev: all-MiniLM-L6-v2 (CPU, 384dim, 80MB)
  # prod: BAAI/bge-m3 (GPU, 1024dim, 2GB)
  # api: DashScope text-embedding-v4 (云端)
```

---

## RSS 来源 (75个)

| 类别 | 来源数 | 说明 |
|------|--------|------|
| Nature 系列 | 14 | Nature, Nat Neuro, Nat Genet, Nat Comm, Nat Protocols... |
| Cell 系列 | 16 | Cell, Mol Cell, Neuron, Trends系列... |
| Science 系列 | 4 | Science, Science AOP, Science Adv, Science Signal |
| 其他顶刊 | 10 | PNAS, eLife, PLoS Bio, EMBO J, NAR... |
| bioRxiv | 12 | 各学科分类 |
| PubMed 搜索 | 8 | 主题检索 (RTCB, UPR, RNA splicing...) |
| 补充期刊 | 8 | Structure, FEBS, ACS Biochemistry... |
| 新闻/科普 | 3 | Phys.org, Science Daily, Quanta |

### 解析器架构

```
parsers/
├── base.py          # BaseParser 抽象类
├── nature.py        # Nature系列 (含MBoC, Comm Bio)
├── cell.py          # Cell Press系列
├── science.py       # Science系列
├── biorxiv.py       # bioRxiv/medRxiv
├── wiley.py         # FEBS Journal/Letters
├── acs.py           # ACS Biochemistry
└── wechat.py        # 公众号/新闻站
```

---

## 项目结构

```
rss-push-workflow/
├── src/
│   ├── models.py                 # NewsItem 数据模型
│   ├── infra/
│   │   └── llm.py                # LLM 客户端 (Zhipu/SiliconFlow/Ollama)
│   │
│   ├── S1_aggregate/             # 步骤1: 多源聚合
│   │   ├── rss.py                # RSS 抓取器
│   │   ├── pubmed.py             # PubMed 抓取 + DOI enrichment
│   │   ├── sources.yaml          # 来源配置
│   │   └── parsers/              # 差异化解析器
│   │
│   ├── S2_clean/                 # 步骤2: 内容清洗
│   │   └── html.py               # HTML清洗
│   │
│   ├── S3_dedup/                 # 步骤3: 指纹去重
│   │   ├── fingerprint.py        # 指纹计算 (DOI/Title hash)
│   │   └── filter.py             # 去重过滤器
│   │
│   ├── S4_filter/                # 步骤4: 混合过滤
│   │   ├── hybrid.py             # Layer1 + Layer2 过滤
│   │   ├── config.py             # 配置加载
│   │   └── bio_keywords.py       # 生物学关键词
│   │
│   ├── S5_enrich/                # 步骤5: PubMed补充
│   │   └── enrich.py             # DOI→PubMed abstract
│   │
│   ├── S6_llm/                   # 步骤6: LLM处理
│   │   └── process.py            # 摘要生成
│   │
│   └── S7_deliver/               # 步骤7: 输出
│       ├── console.py            # 控制台输出
│       └── html.py               # HTML输出
│
├── config/
│   ├── settings.yaml             # 全局配置 (LLM, semantic)
│   ├── filter.yaml               # 过滤配置 (VIP关键词, 阈值)
│   ├── semantic_anchors.json     # 语义锚点 (分层)
│   └── user_profile.yaml         # 用户画像 (LLM prompt)
│
├── data/
│   ├── raw/{date}/               # 原始抓取
│   ├── cleaned/{date}/           # 清洗后
│   ├── deduped/{date}/           # 去重后
│   ├── filtered/{date}/          # 过滤后
│   ├── seen.json                 # 历史指纹
│   └── embedding_cache.db        # 向量缓存
│
├── output/                       # 最终输出
└── main.py                       # 流程编排
```

---

## 配置文件说明

### config/settings.yaml

```yaml
llm:
  provider: zhipu | siliconflow | ollama
  zhipu:
    model: glm-4.6
    concurrency: 5
  siliconflow:
    model: Qwen/Qwen3-8B
    enable_thinking: true

semantic:
  preset: dev | prod | api | custom
  cache_enabled: true
  api:
    provider: dashscope
    model: text-embedding-v4
```

### config/filter.yaml

```yaml
final_threshold: 0.50

vip_keywords:
  tier1:
    multiplier: 1.50
    patterns: [RTCB, tRNA ligase, tRNA splicing]
  tier2:
    multiplier: 1.30
    patterns: [IRE1, XBP1, UPR, ...]
  tier3:
    multiplier: 1.15
    patterns: [enhancer, transposon, ...]

bio_keywords: [rna, dna, protein, ...]  # 200个词根
```

---

## 当前进度

- [x] 1_aggregate - RSS聚合 + 差异化解析器 (75来源, 7种解析器)
- [x] 2_clean - HTML清洗
- [x] 3_dedup - 指纹去重 (DOI/Title hash)
- [x] 4_filter - 混合过滤 (关键词 + 语义 + VIP加权)
- [x] 5_enrich - PubMed摘要补充 (enrich_via_pubmed)
- [x] 6_llm - 摘要生成 (Zhipu/SiliconFlow)
- [x] 7_deliver - 输出 (console/json/html)
- [x] main.py - 流程编排

### 质量优化功能

**负向锚点 (Negative Anchors)**
- 配置位置：`config/semantic_anchors.yaml` 的 `negative:` 部分
- 作用：匹配临床试验/农业应用/纯计算方法/综述评论等非目标内容时降权
- 参数：`config/filter.yaml` 中 `negative_anchor.threshold` 和 `negative_anchor.penalty`
- 公式：`final_score = ... × negative_penalty` (默认命中时 ×0.6)

**用户反馈收集**
- HTML输出带反馈按钮 (thumbs up/down)
- 点击跟踪 (自动记录)
- 浏览器端存储 + 导出JSON功能
- 后端存储：`data/feedback.json`
- 导入功能：`src/infra/feedback.py` 中 `import_from_browser_export()`
- 统计查看：`get_feedback_stats()`

### 已解决问题

- Science/PNAS RSS无摘要 → `enrich_via_pubmed: true` + `enrich_min_content_len: 200`
- FEBS content字段取错 → 解析器使用 `_extract_content()` 优先取更长内容
- 向量模型无法下载 → 支持 DashScope 云端 API
