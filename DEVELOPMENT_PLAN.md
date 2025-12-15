# ScholarPipe MVP 开发计划

> 学术资讯聚合 + AI 摘要早报

---

## 核心理念

- **把论文当新闻对待**
- 重点是**聚合**多源 + **清晰**呈现
- 用户感兴趣自己点链接看原文

---

## 工作流架构 (6 步)

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐   ┌────────────────┐   ┌─────────────┐
│ 1_aggregate │──►│  2_clean    │──►│  3_dedup    │──►│  4_filter  │──►│    5_llm       │──►│  6_deliver  │
│   (聚合)    │   │  (清洗)     │   │ (指纹去重)  │   │(关键词过滤)│   │ (语义去重+评分) │   │   (输出)    │
└─────────────┘   └─────────────┘   └─────────────┘   └────────────┘   └────────────────┘   └─────────────┘
       │                                   │                 │                  │
  sources.yaml                        seen.json        BIO_KEYWORDS         LLM API
  parsers/                          (已检阅记录)      (生物学词根)      (合并+评分+摘要)
```

### 分层过滤漏斗

```
226 items (采集)
    ↓
3_dedup: 指纹比对
    ↓ 去除重复
178 items (去重后)
    ↓
4_filter: 关键词过滤
    ↓ 过滤非生物学文章
120 items (~67%)
    ↓
5_llm: AI 打分
    ↓ 过滤低分
~N items (高分入选)
```

关键词过滤可节省 ~33% 的 LLM token 成本。

---

## 去重策略

### 核心原则：只要看一眼，就要记下来

**问题**：如果只记录"入选日报的条目"，被 AI 判低分丢弃的文章第二天还在 RSS 里，会被重复送给 AI 打分，浪费 token。

**解决**：记录所有**已检阅**的条目，无论最终是入选还是丢弃。

```
Day 1: 文章A → AI打分 2分 → 丢弃 → 记录指纹 ✓
Day 2: 文章A → 指纹比对 → 已存在 → 直接跳过（零成本）
```

### 指纹策略 (Fingerprinting)

为每篇文章计算唯一指纹，优先级：

| 优先级 | 规则 | 说明 |
|--------|------|------|
| 1 | DOI | 最可靠，标准化后直接作为 ID |
| 2 | Title Hash | 没有 DOI 时，标题归一化后计算 MD5 |

```python
def get_fingerprint(item: NewsItem) -> str:
    if item.doi:
        return normalize_doi(item.doi)  # "10.1038/s41586-xxx"
    return hash_title(item.title)       # "a1b2c3d4..."
```

归一化规则：
- DOI: 去前缀 `doi:`、`https://doi.org/`，小写
- Title: 小写 + 去标点 + 去多余空格 → MD5

### seen.json 格式

```json
{
  "10.1038/s41586-025-xxx": "2025-12-15T10:00:00",
  "a1b2c3d4e5f6...": "2025-12-15T10:00:00"
}
```

记录保留 7 天后自动清理（等文章从 RSS 源消失）。

### 去重流程

```
144 items (采集)
    ↓
3_dedup: 指纹比对 seen.json
    ↓ 过滤已见过的
~N items (真正新的)
    ↓
4_llm: AI 打分 + 语义合并
    ↓ 全量记录指纹（包括被丢弃的）
~M items (高分入选)
    ↓
5_deliver: 输出日报
```

---

## RSS 数据特点

| 来源 | 条目数 | 覆盖范围 | 日期字段 | 特点 |
|------|--------|----------|----------|------|
| Nature | ~75 | 5天 | `updated` | 按天更新 |
| Nature Neuro | ~8 | 4天 | `updated` | 按天更新 |
| Cell | ~21 | ~2个月 | `prism_publicationdate` | 按期更新 |
| Science | ~40 | 1天为主 | `prism_coverdate` | 集中发布 |

---

## 数据模型

```python
class NewsItem:
    """一条学术资讯"""

    # 核心内容
    title: str
    content: str          # 摘要/描述
    link: str             # 原文链接

    # 元数据
    authors: list[str]    # 作者列表
    doi: str              # DOI（用于去重）

    # 溯源
    source_name: str      # "Nature", "Cell", "Science"
    source_url: str       # RSS URL
    fetched_at: datetime

    # AI 处理结果
    score: float | None   # 0-10
    summary: str          # 200字中文摘要
```

---

## 项目结构

```
rss-push-workflow/
├── src/
│   ├── models.py                 # NewsItem 数据模型
│   ├── infra/                    # 基础设施
│   │   └── llm.py                # LLM 客户端 (Zhipu GLM-4-Flash)
│   │
│   ├── 1_aggregate/              # 步骤1: 多源聚合
│   │   ├── __init__.py           # fetch_all(), load_sources()
│   │   ├── __main__.py           # 独立运行
│   │   ├── rss.py                # RSS 抓取器
│   │   ├── sources.yaml          # 来源配置
│   │   └── parsers/              # 差异化解析器
│   │       ├── base.py
│   │       ├── nature.py
│   │       ├── cell.py
│   │       └── science.py
│   │
│   ├── 2_clean/                  # 步骤2: 内容清洗
│   │   ├── __init__.py           # batch_clean()
│   │   ├── __main__.py
│   │   └── html.py               # HTML清洗
│   │
│   ├── 3_dedup/                  # 步骤3: 指纹去重
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── fingerprint.py        # 指纹计算
│   │   ├── seen.py               # 历史记录管理
│   │   └── filter.py             # 去重过滤器
│   │
│   ├── 4_filter/                 # 步骤4: 关键词过滤
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── bio_keywords.py       # 生物学关键词过滤
│   │
│   ├── 5_llm/                    # 步骤5: LLM处理
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── process.py            # 语义去重+评分+摘要
│   │
│   └── 6_deliver/                # 步骤6: 输出
│       ├── __init__.py
│       ├── __main__.py
│       └── console.py            # 控制台/文件输出
│
├── data/
│   ├── raw/                      # 第一步产出：原始抓取
│   │   └── {date}/
│   │       ├── nature.json
│   │       ├── cell.json
│   │       └── all.json
│   ├── cleaned/                  # 第二步产出：清洗后
│   │   └── {date}/
│   │       └── all.json
│   ├── deduped/                  # 第三步产出：去重后
│   │   └── {date}/
│   │       └── all.json
│   ├── filtered/                 # 第四步产出：关键词过滤后
│   │   └── {date}/
│   │       └── all.json
│   └── seen.json                 # 历史指纹记录
│
├── output/                       # 最终输出
├── main.py                       # 流程编排
└── requirements.txt
```

---

## 模块接口

### 1_aggregate
```python
def fetch_all(sources=None, save_raw=True) -> list[NewsItem]
def load_sources() -> list[dict]
```
输出：`data/raw/{date}/*.json`

### 2_clean
```python
def batch_clean(items: list[NewsItem]) -> list[NewsItem]
```

### 3_dedup
```python
# fingerprint.py
def get_fingerprint(item: NewsItem) -> str

# seen.py
def load(filepath="data/seen.json") -> dict[str, str]
def save(seen: dict, filepath="data/seen.json") -> None
def mark_seen(seen: dict, fingerprint: str) -> None
def cleanup(seen: dict, max_age_days=7) -> dict

# filter.py
def filter_unseen(items: list[NewsItem], seen: dict) -> list[NewsItem]
```

### 4_filter
```python
# bio_keywords.py
BIO_KEYWORDS: list[str]  # 110个生物学词根

def has_bio_keyword(title: str, content: str) -> tuple[bool, list[str]]
def filter_bio(items: list[NewsItem]) -> tuple[list[NewsItem], list[NewsItem]]
# 返回 (通过的, 被过滤的)
```

### 5_llm
```python
def process_batch(items: list[NewsItem], score_threshold=6.0) -> list[NewsItem]
# 返回高分条目，同时记录所有处理过的指纹
```

### 6_deliver
```python
def to_console(items: list[NewsItem], stats: dict) -> None
def to_json(items: list[NewsItem], path: str) -> None
def to_markdown(items: list[NewsItem]) -> str
```

---

## 主流程 (main.py)

```python
def run():
    seen_records = dedup.load()

    # 1. 聚合
    items = aggregate.fetch_all()
    total = len(items)

    # 2. 清洗
    items = clean.batch_clean(items)

    # 3. 指纹去重（过滤已见过的）
    items, new_fingerprints = dedup.filter_unseen(items, seen_records)
    after_dedup = len(items)

    # 4. 关键词过滤（过滤非生物学文章）
    items, filtered = filter.filter_bio(items)
    after_filter = len(items)

    # 5. LLM 处理 + 记录（成功才记录，失败不记录可重试）
    try:
        results = llm.process_batch(items)

        # 成功：全量记录指纹（包括被丢弃的低分条目）
        dedup.mark_batch(seen_records, new_fingerprints)
        seen_records = dedup.cleanup(seen_records)
        dedup.save(seen_records)
    except Exception as e:
        print(f"LLM failed: {e}, not saving seen records (will retry next run)")
        raise

    # 6. 输出
    stats = {
        "total": total,
        "after_dedup": after_dedup,
        "after_filter": after_filter,
        "recommended": len(results)
    }
    deliver.to_console(results, stats)
```

**记录策略**：见过即记录，但 LLM 失败时不保存（下次可重试）

---

## 当前进度

- [x] models.py - NewsItem (含 authors, doi)
- [x] 1_aggregate - RSS 聚合 + 差异化解析器
- [x] 2_clean - HTML 清洗
- [x] 3_dedup - 指纹去重（DOI/Title hash）
- [x] 4_filter - 生物学关键词过滤（110个词根，~33%过滤率）
- [ ] 5_llm - 语义去重 + 评分 + 摘要
- [ ] 6_deliver - 输出
- [ ] main.py - 流程编排

### 已知问题

- Science AOP 解析器缺少摘要提取（content 为空），导致只能通过标题匹配关键词


sudo密码8300110fyM