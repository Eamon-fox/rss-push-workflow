"""Async task management for background job processing."""

import threading
import uuid
from datetime import datetime
from typing import Optional

from src.core import now_iso


class TaskStore:
    """
    简单的内存任务存储（单进程适用）

    任务状态: pending → running → done/failed

    Usage:
        store = TaskStore()
        task_id = store.create("user_openid", "generate_daily")
        store.set_running(task_id)
        store.update_progress(task_id, step=1, step_name="Loading...")
        store.set_done(task_id, {"count": 10})
    """

    def __init__(self):
        self._tasks: dict[str, dict] = {}
        self._user_tasks: dict[str, str] = {}  # openid → latest task_id
        self._lock = threading.Lock()

    def create(self, openid: str, task_type: str = "generate_daily") -> str:
        """创建新任务，返回 task_id"""
        task_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "openid": openid,
                "type": task_type,
                "status": "pending",
                "created_at": now_iso(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "result": None,
                # 进度信息
                "progress": {
                    "step": 0,  # 当前步骤编号 (1-5)
                    "total_steps": 5,  # 总步骤数
                    "step_name": "",  # 当前步骤名称
                    "detail": "",  # 详细描述
                    "current": 0,  # 当前项（如正在处理第几篇）
                    "total": 0,  # 总项数（如共多少篇）
                    "percent": 0,  # 总体进度百分比 0-100
                },
            }
            self._user_tasks[openid] = task_id
        return task_id

    def get(self, task_id: str) -> Optional[dict]:
        """获取任务信息"""
        return self._tasks.get(task_id)

    def get_user_task(self, openid: str) -> Optional[dict]:
        """获取用户最新任务"""
        task_id = self._user_tasks.get(openid)
        if task_id:
            return self._tasks.get(task_id)
        return None

    def set_running(self, task_id: str):
        """标记为运行中"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "running"
                self._tasks[task_id]["started_at"] = now_iso()

    def update_progress(
        self,
        task_id: str,
        step: int,
        step_name: str,
        detail: str = "",
        current: int = 0,
        total: int = 0,
    ):
        """
        更新任务进度

        Args:
            task_id: 任务 ID
            step: 当前步骤编号 (1-5)
            step_name: 步骤名称
            detail: 详细描述
            current: 当前处理项
            total: 总项数
        """
        with self._lock:
            if task_id in self._tasks:
                progress = self._tasks[task_id]["progress"]
                progress["step"] = step
                progress["step_name"] = step_name
                progress["detail"] = detail
                progress["current"] = current
                progress["total"] = total
                # 计算总体百分比：前4步各占10%，第5步(摘要生成)占60%
                if step <= 4:
                    progress["percent"] = step * 10
                else:
                    # 摘要生成步骤：40% + (current/total) * 60%
                    base = 40
                    if total > 0:
                        progress["percent"] = base + int((current / total) * 60)
                    else:
                        progress["percent"] = base

    def set_done(self, task_id: str, result: dict = None):
        """标记为完成"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "done"
                self._tasks[task_id]["finished_at"] = now_iso()
                self._tasks[task_id]["result"] = result
                # 完成时进度设为 100%
                self._tasks[task_id]["progress"]["percent"] = 100
                self._tasks[task_id]["progress"]["step_name"] = "完成"

    def set_failed(self, task_id: str, error: str):
        """标记为失败"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["finished_at"] = now_iso()
                self._tasks[task_id]["error"] = error

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """清理过期任务，返回清理数量"""
        cutoff = datetime.now().timestamp() - max_age_hours * 3600
        removed = 0
        with self._lock:
            to_delete = []
            for task_id, task in self._tasks.items():
                created = datetime.fromisoformat(task["created_at"]).timestamp()
                if created < cutoff:
                    to_delete.append(task_id)
            for task_id in to_delete:
                del self._tasks[task_id]
                removed += 1
        return removed


# 全局单例
_global_store: Optional[TaskStore] = None


def get_task_store() -> TaskStore:
    """Get global task store instance."""
    global _global_store
    if _global_store is None:
        _global_store = TaskStore()
    return _global_store
