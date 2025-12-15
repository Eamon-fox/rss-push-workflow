# ScholarPipe MVP 开发计划

> 学术资讯聚合 + AI 摘要早报

---

## 核心理念

- **把论文当新闻对待**
- 重点是**聚合**多源 + **清晰**呈现
- 用户感兴趣自己点链接看原文
- **无持久化**：数据在内存流转，输出后可丢弃

---

## 工作流架构 (5 步)

```
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌────────────────┐   ┌───────────┐
│ aggregate │──►│   clean   │──►│   dedup   │──►│  llm_process   │──►│  deliver  │
│  (聚合)   │   │  (清洗)   │   │(跨期去重) │   │(当期去重+评分) │   │  (输出)   │
└───────────┘   └───────────┘   └───────────┘   └────────────────┘   └───────────┘
                                      │                  │
                                 seen.json          LLM API
                               (历史记录)        (语义合并+评分+摘要)
```

### 两种去重

| 类型 | 位置 | 目的 | 方法 |
|------|------|------|------|
| **跨期去重** | dedup.py | 前几天推送过的不再推 | hash + seen.json |
| **当期去重** | llm_process.py | 本次不同来源报道同一研究 | LLM 语义合并 |

---

## 数据模型

```python
class NewsItem:
    """一条学术资讯"""

    # 核心内容
    title: str
    content: str          # 原始内容
    link: str             # 原文链接

    # 溯源
    source_name: str      # "Nature RSS", "PubMed"
    source_url: str
    fetched_at: datetime

    # AI 处理结果
    score: float | None   # 0-10
    summary: str          # 200字中文摘要

    # 计算属性
    content_hash: str     # 用于跨期去重
```

---

## 项目结构

```
scholar-pipe/
├── src/
│   ├── __init__.py
│   ├── models.py            # NewsItem 数据模型
│   ├── seen.py              # 历史记录存储
│   ├── aggregate.py         # 步骤1: 多源聚合
│   ├── clean.py             # 步骤2: 内容清洗
│   ├── dedup.py             # 步骤3: 跨期去重
│   ├── llm_process.py       # 步骤4: LLM处理 (当期去重+评分+摘要)
│   └── deliver.py           # 步骤5: 输出
├── config/
│   └── settings.yaml
├── data/
│   └── seen.json            # 历史 hash 记录
├── output/                  # 输出目录
├── main.py                  # 流程编排
└── requirements.txt
```

---

## 模块接口

### aggregate.py
```python
def fetch_rss(url: str, source_name: str) -> list[NewsItem]
def fetch_all(sources: list[dict]) -> list[NewsItem]
```

### clean.py
```python
def clean(item: NewsItem) -> NewsItem
def batch_clean(items: list[NewsItem]) -> list[NewsItem]
```

### seen.py
```python
def load(filepath: str) -> dict[str, str]
def save(seen: dict, filepath: str) -> None
def is_seen(seen: dict, hash: str, window_hours=72) -> bool
def mark_seen(seen: dict, hash: str) -> None
def cleanup(seen: dict, max_age_hours=168) -> dict
```

### dedup.py
```python
def filter_unseen(items: list[NewsItem], seen: dict, window_hours=72) -> list[NewsItem]
```

### llm_process.py
```python
def process_batch(items: list[NewsItem], api_key: str, score_threshold=6.0) -> list[NewsItem]
# 输入: 跨期去重后的条目
# 处理: 当期语义去重 + 评分 + 摘要
# 输出: 高分条目
```

### deliver.py
```python
def to_console(items: list[NewsItem], stats: dict) -> None
def to_json(items: list[NewsItem], path: str) -> None
def to_markdown(items: list[NewsItem]) -> str
```

---

## 主流程 (main.py)

```python
def run():
    seen_records = seen.load()

    # 1. 聚合
    items = aggregate.fetch_all(sources)

    # 2. 清洗
    items = clean.batch_clean(items)

    # 3. 跨期去重 (vs 历史)
    items = dedup.filter_unseen(items, seen_records)

    # 4. LLM 处理 (当期去重 + 评分 + 摘要)
    results = llm_process.process_batch(items, api_key)

    # 5. 输出
    deliver.to_console(results, stats)

    # 保存历史
    seen.save(seen_records)
```

---

## LLM 处理逻辑

**Prompt**：

```
以下是今天抓取的学术资讯。请：

1. 当期去重: 识别报道同一研究的条目，合并为一条
2. 评分: 为每条独立研究评分 (0-10)
3. 摘要: 为高分条目生成200字中文摘要

输入:
[1] RTCB mediates... (Nature RSS)
[2] RNA repair enzyme... (PubMed)  <- 同一研究
[3] 其他研究... (Cell RSS)

输出 JSON:
[
  {"merged_from": [1,2], "title": "...", "score": 9.2, "summary": "..."},
  {"merged_from": [3], "title": "...", "score": 7.5, "summary": "..."}
]
```

---

## 并行开发

| 开发线 | 模块 | 依赖 |
|--------|------|------|
| A | models.py | 无 |
| B | aggregate.py + clean.py | models |
| C | seen.py + dedup.py | models |
| D | llm_process.py | models |
| E | deliver.py | models |

---

## 开发顺序

```
Phase 0: models.py
    ↓
Phase 1: aggregate.py (跑通一个 RSS)
    ↓
Phase 2: clean.py + seen.py + dedup.py
    ↓
Phase 3: llm_process.py (核心 AI 逻辑)
    ↓
Phase 4: deliver.py + main.py
```

---

## 输出示例

```
══════════════════════════════════════════════════════════════════
  ScholarPipe 学术早报 | 2024-01-15
══════════════════════════════════════════════════════════════════

------------------------------------------------------------------
[9.2] RTCB mediates RNA repair in neurons
来源: Nature Neuroscience (合并自 2 个来源)

本研究揭示了 RNA 修复酶 RTCB 在神经元中的作用机制...

链接: https://www.nature.com/articles/s41593-024-xxx
------------------------------------------------------------------

══════════════════════════════════════════════════════════════════
 共抓取 147 条 | 跨期去重后 89 条 | 推荐 12 条
══════════════════════════════════════════════════════════════════
```
