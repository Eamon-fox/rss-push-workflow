# ScholarPipe MVP 开发计划

> 目标：搭建最小可运行框架，架构清晰，逐步迭代

---

## 核心架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Aggregator │───►│   Dedup     │───►│  Screener   │
│  (数据聚合)  │    │  (去重网关)  │    │  (AI筛选)   │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Delivery   │◄───│  Digester   │◄───│   Fetcher   │
│   (分发)    │    │  (深度解析)  │    │ (PDF下载)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

**数据流**: Paper 对象贯穿全流程，每个模块只做一件事

---

## MVP 范围 (V0.1)

| 模块 | MVP 实现 | 后续扩展 |
|------|----------|----------|
| Aggregator | 单个 RSS 源 | PubMed, BioRxiv |
| Dedup | DOI 精确匹配 | 标题指纹, 相似度 |
| Screener | 单个 LLM API | 多模型, 批量 |
| Fetcher | Sci-Hub 直链 | Playwright 校园网 |
| Digester | 仅保存摘要 | Marker 全文解析 |
| Delivery | 控制台输出 | WordPress, Zotero |

---

## 项目结构 (MVP)

```
scholar-pipe/
├── src/
│   ├── __init__.py
│   ├── models.py         # 统一数据模型
│   ├── database.py       # SQLite 操作
│   ├── aggregator.py     # 数据聚合
│   ├── dedup.py          # 去重
│   ├── screener.py       # AI 筛选
│   ├── fetcher.py        # PDF 下载
│   ├── digester.py       # 深度解析
│   └── delivery.py       # 分发
├── config/
│   └── settings.yaml     # 配置
├── data/                 # 数据库
├── downloads/            # PDF 文件
├── main.py               # 主入口
├── requirements.txt
└── pyproject.toml
```

---

## Phase 0: 骨架搭建

**目标**: 建立空壳框架，所有模块可调用

- [ ] 创建项目目录结构
- [ ] 定义核心数据模型 `Paper`
- [ ] 各模块占位实现（pass）
- [ ] main.py 串联调用
- [ ] 能运行不报错

---

## Phase 1: 数据模型 + 数据库

**目标**: Paper 对象能存取

```python
# models.py
class Paper:
    doi: str
    title: str
    abstract: str
    source: str
    score: float | None
    status: str  # new, screened, fetched, processed
    created_at: datetime
```

- [ ] Pydantic Paper 模型
- [ ] SQLite 建表
- [ ] CRUD 操作

---

## Phase 2: 聚合 + 去重

**目标**: 从 RSS 拉取，去重后入库

```python
# 伪代码
papers = aggregator.fetch("nature_rss")  # 拉取
for p in papers:
    if not dedup.exists(p.doi):          # 去重
        database.insert(p)               # 入库
```

- [ ] feedparser 解析 RSS
- [ ] DOI 去重检查
- [ ] 新论文入库

---

## Phase 3: AI 筛选

**目标**: 对新论文打分，过滤低分

```python
# 伪代码
new_papers = database.get_by_status("new")
for p in new_papers:
    score = screener.evaluate(p.title, p.abstract)
    database.update_score(p.doi, score)
```

- [ ] DeepSeek API 调用
- [ ] 简单 Prompt: 打分 0-10
- [ ] 更新数据库分数

---

## Phase 4: PDF 获取 (简化版)

**目标**: 高分论文尝试下载 PDF

```python
# 伪代码
high_score = database.get_by_score(min=6)
for p in high_score:
    pdf_path = fetcher.download(p.doi)
    database.update_pdf_path(p.doi, pdf_path)
```

- [ ] 构造下载 URL
- [ ] httpx 下载
- [ ] 保存文件

---

## Phase 5: 输出

**目标**: 汇总结果，简单展示

```python
# 伪代码
results = database.get_today_processed()
delivery.print_summary(results)
```

- [ ] 控制台格式化输出
- [ ] 可选: 保存为 JSON

---

## 开发顺序

```
Day 1: Phase 0 (骨架) + Phase 1 (模型/数据库)
Day 2: Phase 2 (聚合/去重)
Day 3: Phase 3 (AI筛选)
Day 4: Phase 4 (下载) + Phase 5 (输出)
Day 5: 联调测试
```

---

## 接口约定

每个模块对外暴露统一接口：

```python
# aggregator.py
def fetch(source_name: str) -> list[Paper]: ...

# dedup.py
def exists(doi: str) -> bool: ...
def check(paper: Paper) -> DedupeResult: ...

# screener.py
def evaluate(title: str, abstract: str) -> float: ...

# fetcher.py
def download(doi: str) -> str | None: ...  # 返回文件路径

# digester.py
def analyze(pdf_path: str) -> str: ...  # 返回解读文本

# delivery.py
def output(papers: list[Paper]) -> None: ...
```

---

## 配置示例

```yaml
# config/settings.yaml
database:
  path: data/history.db

sources:
  - name: nature_neuro
    type: rss
    url: https://www.nature.com/neuro.rss

screener:
  api: deepseek
  api_key: ${DEEPSEEK_API_KEY}
  threshold: 6

fetcher:
  download_dir: downloads/
  timeout: 30
```

---

## 立即开始

现在执行 Phase 0：创建骨架文件
