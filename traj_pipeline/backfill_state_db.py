# -*- coding: utf-8 -*-
"""全量回填 Hermes workspace 任务的 state.db(token 开销数据)。

对已下载但未携带 state.db 的 Hermes 任务, 从远端 workspace_obs 补下载
profiles/assistant1/state.db 及相关 WAL 文件。

用法:
    python backfill_state_db.py [--dry-run] [--max-tasks N]

默认读取 data_viewer/tasks.json 中 source_type=workspace 且包含 Hermes 的任务;
--dry-run 只列出需补的任务, 不下;
--max-tasks N 限制最大补下任务数(用于小规模测试)。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
DEFAULT_CONFIG = ROOT / "data_viewer" / "config.json"
TASKS_FILE = ROOT / "data_viewer" / "tasks.json"


def load_tasks():
    if not TASKS_FILE.exists():
        print(f"[ERR] tasks.json 不存在: {TASKS_FILE}")
        return []
    with open(TASKS_FILE, encoding="utf-8") as f:
        return json.load(f).get("tasks", [])


def load_config():
    if DEFAULT_CONFIG.exists():
        with open(DEFAULT_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    return {"obsutil_path": "/home/w00802407/obsutil/obsutil"}


def is_hermes_task(task: dict) -> bool:
    """判断是否是 Hermes workspace 任务(名字含 hermes 或 harness 字段为 hermes)。"""
    if task.get("source_type") != "workspace":
        return False
    name = (task.get("name") or "").lower()
    obs = (task.get("workspace_obs") or "").lower()
    return "hermes" in name or "hermes" in obs


def need_backfill(task: dict) -> bool:
    """检查本地是否已有 state.db, 避免重复下载。"""
    output_dir = ROOT / task["output_dir"]
    origin = output_dir / "origin"
    if not origin.is_dir():
        return False  # 还没下载任何数据
    # 找任意一个 Hermes task 目录
    for task_dir in origin.iterdir():
        if not task_dir.is_dir():
            continue
        state_db = task_dir / "profiles" / "assistant1" / "state.db"
        if state_db.exists():
            return False  # 已有, 不需要补
        # 找到了一个 task 目录但没 state.db → 需要补
        # 只检查第一个 task 目录就 break, 因为同组任务要么全有要么全无
        return True
    return False


def backfill_one_task(task: dict, obsutil: str, dry_run: bool = False):
    """对单个 Hermes task 补下载 state.db。"""
    workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
    if not workspace_obs:
        print(f"  [SKIP] workspace_obs 为空")
        return False

    output_dir = ROOT / task["output_dir"]
    origin = output_dir / "origin"
    origin.mkdir(parents=True, exist_ok=True)

    dest = str(origin)
    cmd = [obsutil, "cp", f"{workspace_obs}/", dest, "-r", "-f",
           "-include", "*profiles/assistant*/state.db*"]

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return True

    print(f"  下载中...", end=" ", flush=True)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=600)
        if res.returncode != 0:
            print(f"[FAIL] {res.stdout[-300:]}")
            return False
        # 检查是否下载到了文件
        files = list(origin.rglob("state.db*"))
        if files:
            sizes = {f.name: f.stat().st_size for f in files}
            print(f"OK ({sizes})")
            return True
        else:
            print(f"OK (但未找到 state.db, 可能远端也没有)")
            return True
    except Exception as e:
        print(f"[ERR] {e}")
        return False


def main():
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    max_tasks = None
    for arg in sys.argv:
        if arg.startswith("--max-tasks="):
            max_tasks = int(arg.split("=", 1)[1])

    tasks = load_tasks()
    cfg = load_config()
    obsutil = cfg.get("obsutil_path", "/home/w00802407/obsutil/obsutil")

    if force:
        # --force 模式: 对所有 Hermes workspace 任务直接补 state.db, 不管是否有 origin 目录
        hermes_tasks = [t for t in tasks if is_hermes_task(t)]
    else:
        hermes_tasks = [t for t in tasks if is_hermes_task(t) and need_backfill(t)]
    if not hermes_tasks:
        print("没有需要回填的 Hermes 任务(所有任务已有 state.db 或非 Hermes)。")
        return

    if max_tasks:
        hermes_tasks = hermes_tasks[:max_tasks]

    total = len(hermes_tasks)
    ok = 0
    fail = 0

    print(f"{'[DRY-RUN] ' if dry_run else ''}共 {total} 个 Hermes 任务需要回填 state.db:\n")
    for i, t in enumerate(hermes_tasks, 1):
        print(f"[{i}/{total}] {t.get('name', '?')} ({t.get('id', '?')})")
        if backfill_one_task(t, obsutil, dry_run):
            ok += 1
        else:
            fail += 1
        print()

    print(f"完成: {ok} 成功, {fail} 失败, 共 {total} 任务")
    if dry_run:
        print("(dry-run 模式, 并未实际下载)")


if __name__ == "__main__":
    main()
