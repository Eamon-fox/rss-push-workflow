# Bug Log

## 2025-12-19: Pipeline 推送失败

### 问题现象
- 自动推送（cron）失败
- 手动触发（API）失败
- 错误信息：`PermissionError: [Errno 13] Permission denied`

### 根本原因

#### Bug 1: 文件权限冲突

**原因**：`scholarpipe-api.service` 配置为 `User=root`，导致通过 API 触发时创建的文件（日志、数据文件）所有者为 root。而 cron 作业以 eamon 用户运行，无法写入 root 所有的文件。

**影响文件**：
- `logs/2025-12-19.log`
- `data/pipeline_progress.json`
- `data/` 目录下多个文件
- `output/daily.json` 等

**修复**：
```bash
# 1. 修改 service 文件
# scholarpipe-api.service: User=root -> User=eamon

# 2. 修复现有文件权限
sudo chown -R eamon:eamon /opt/rss-push-workflow/data/
sudo chown -R eamon:eamon /opt/rss-push-workflow/logs/
sudo chown -R eamon:eamon /opt/rss-push-workflow/output/

# 3. 重启服务
sudo systemctl daemon-reload
sudo systemctl restart scholarpipe-api.service
```

---

#### Bug 2: LLMCache 单例竞态条件

**错误信息**：
```
AttributeError: 'LLMCache' object has no attribute 'conn'
```

**原因**：`LLMCache` 类使用单例模式，但在多线程环境下存在竞态条件。

原代码（`src/S6_llm/process.py:40-47`）：
```python
def __new__(cls):
    if cls._instance is None:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)  # 赋值
                cls._instance._db_lock = Lock()
                cls._instance._init_db()  # 初始化 conn
    return cls._instance
```

问题：当线程 A 执行 `cls._instance = super().__new__(cls)` 后，`_instance` 已非 None，但 `_init_db()` 尚未调用。此时线程 B 检查 `_instance is None` 为 False，直接返回未完全初始化的实例（缺少 `conn` 属性）。

**修复**：
```python
def __new__(cls):
    if cls._instance is None:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._db_lock = Lock()
                instance._init_db()
                cls._instance = instance  # 初始化完成后再赋值
    return cls._instance
```

**文件**：`src/S6_llm/process.py:40-47`

---

### 经验教训

1. **Service 用户一致性**：确保 systemd service 与 cron 使用相同用户，避免文件权限冲突
2. **单例模式线程安全**：双重检查锁定（Double-Checked Locking）需确保对象完全初始化后再发布引用

---

## 2025-12-19: 同类风险排查

### 问题与修改建议

#### 风险 1: 多入口写入导致权限冲突再现

**现象**：流水线/服务多处直接写入 `logs/`、`data/`、`output/`，一旦有不同用户运行（cron vs API vs 手工），仍可能复现 PermissionError。

**涉及**：
- `main.py`（日志与进度文件）
- `src/S1_aggregate/rss.py` / `src/S1_aggregate/pubmed.py`（raw 缓存）
- `src/S2_clean/html.py` / `src/S3_dedup/filter.py` / `src/S4_filter/hybrid.py`（中间数据）
- `src/S6_llm/process.py` / `src/S4_filter/embedding_cache.py`（SQLite 缓存）
- `src/S7_deliver/console.py` / `src/archive.py`（输出与归档）

**建议**：
- 统一执行用户（systemd/cron/手工一致），并在启动时校验目录所有权与权限。
- 对关键写入点增加异常兜底与清晰告警，避免整个流程硬失败。
- 必要时将日志/数据目录放到单独的可写路径，并明确权限策略（umask/ACL）。

---

#### 风险 2: JSON 读改写无锁导致并发数据丢失

**现象**：用户、收藏、反馈、去重记录均采用“读-改-写”且无锁，FastAPI 并发请求或多进程会导致丢更新或写坏文件。

**涉及**：
- `src/auth.py`（users.json）
- `src/bookmarks.py`（bookmarks.json）
- `src/infra/feedback.py`（feedback.json）
- `src/S3_dedup/seen.py`（seen.json）
- `src/article_index.py`（article_index.json，虽有原子替换但无并发锁）

**建议**：
- 增加文件锁（如 `fcntl`/`portalocker`）或用 SQLite 替代 JSON 存储。
- 统一用“临时文件 + 原子替换”，并加写入失败重试/告警。

---

#### 风险 3: SQLite 缓存并发访问缺乏锁与超时策略

**现象**：Embedding/LLM 缓存使用共享连接且无统一锁，线程并发可能触发 `database locked` 或写入冲突。

**涉及**：
- `src/S4_filter/embedding_cache.py`
- `src/S4_filter/hybrid.py`（全局缓存单例）
- `src/S6_llm/process.py`（LLMCache）

**建议**：
- 为 SQLite 操作加线程锁，或改为“每线程连接 + busy_timeout”。
- 设置连接超时并记录重试次数，避免随机失败。

---

#### 风险 4: 运行互斥仅在 API 进程内生效

**现象**：`PipelineStatus` 仅在 API 进程内存中互斥，无法阻止 cron 与 API 同时运行，可能并发写同一批输出/缓存文件。

**涉及**：
- `api.py`（trigger/status）

**建议**：
- 引入跨进程锁（如 `flock`/PID 文件），或将定时触发统一迁移到 systemd timer/同一入口。

---

## 2025-12-19: /api/articles 返回 500

### 问题现象
- `/api/articles/{article_id}` 多次返回 500（前端报错）
- 日志报错：`AttributeError: 'list' object has no attribute 'get'`

### 根本原因

**原因**：`article_index.json` 内容被写成了 JSON 数组（list），而接口逻辑期望为 dict。`get_article_location()` 调用 `.get()` 时触发异常，导致 500。

**堆栈位置**：
- `api.py`（get_article → get_article_location）
- `src/article_index.py`（`get_article_location` 对 index 直接调用 `.get()`）

### 修复建议

- **读取校验**：`src/article_index.py` 中加载后校验类型，非 dict 则返回空索引并记录告警，避免 500。
- **自动修复**：检测到非法结构时触发 `rebuild_index_from_archive()` 重新构建索引。
- **写入保护**：为 index 文件加锁或统一“临时文件 + 原子替换 + 备份”，避免并发/异常写入损坏。
- **紧急处理**：删除损坏的 `data/article_index.json` 或执行重建脚本恢复索引。
