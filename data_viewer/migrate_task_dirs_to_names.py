#!/usr/bin/env python3
"""把存量以 t_xxx 命名的任务文件夹改名为任务名, 并同步更新 tasks.json 的 output_dir。

与 server._task_dir_name 同一套口径(清洗/去重), 保证后续新任务与既有任务命名一致。
任务文件夹未采集(不存在)时只更新 tasks.json 的 output_dir, 不移动任何数据。

用法(在 data_viewer/ 下):
    .venv/bin/python migrate_task_dirs_to_names.py            # 仅打印计划
    .venv/bin/python migrate_task_dirs_to_names.py --apply    # 执行(先备份 tasks.json)
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import (  # noqa: E402
    TASKS_FILE, load_tasks, update_task, _task_dir_name,
)

if __name__ == "__main__":
    apply = "--apply" in sys.argv
    tasks = load_tasks()
    plan = []
    # 用同一份可变列表边算边改 output_dir(与 --apply 时逐任务落盘一致), 保证 dry-run == 实跑
    for t in tasks:
        old_od = Path(t["output_dir"])
        old_base = old_od.name
        new_name = _task_dir_name(t.get("name") or "", t["id"], exclude_dir=old_base,
                                  tasks_list=tasks)
        new_od = old_od.parent / new_name
        if new_name == old_base:
            continue
        plan.append((t["id"], t.get("name"), old_od, new_od))
        t["output_dir"] = str(new_od)

    print(f"任务总数: {len(tasks)}, 需要改名: {len(plan)}")
    for tid, name, old_od, new_od in plan:
        state = "目录存在,将移动" if old_od.exists() else "目录不存在,仅更新 tasks.json"
        print(f"  {tid}: {old_od.name} → {new_od.name}  ({state})")

    if not apply:
        print("\n[dry-run] 未执行任何改动; 确认无误后加 --apply 执行")
        sys.exit(0)

    # 备份 tasks.json
    backup = TASKS_FILE.with_name(TASKS_FILE.name + ".bak-migrate")
    if not backup.exists():
        shutil.copy2(TASKS_FILE, backup)
        print(f"\n已备份 tasks.json -> {backup.name}")

    done = errors = 0
    for tid, name, old_od, new_od in plan:
        try:
            if old_od.exists():
                if new_od.exists():
                    raise FileExistsError(f"目标目录已存在: {new_od.name}")
                old_od.rename(new_od)
            update_task(tid, output_dir=str(new_od))
            print(f"  ✅ {tid}: {old_od.name} → {new_od.name}")
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {tid}: {e}")
            errors += 1

    print(f"\n完成: {done} 个任务已改名, {errors} 个失败")
    if errors:
        print("有失败项, 请检查后再重跑(脚本可重复执行, 已完成的会跳过)")
