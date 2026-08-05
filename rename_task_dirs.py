#!/usr/bin/env python3
"""将 data_viewer/pipeline_output/tasks/ 下的 t_xxx 文件夹重命名为任务名称，
并更新 tasks.json。"""
import json
import os
import shutil
from pathlib import Path

TASKS_DIR = Path("data_viewer/pipeline_output/tasks")
TASKS_FILE = Path("data_viewer/tasks.json")

with open(TASKS_FILE) as f:
    tasks = json.load(f)["tasks"]

print(f"任务总数: {len(tasks)}")
print(f"目标目录: {TASKS_DIR.resolve()}")
print()

renames = []
skipped = []
errors = []

for t in tasks:
    task_id = t["id"]
    name = t["name"]
    old_dir_rel = t["output_dir"]
    old_dir_name = Path(old_dir_rel).name  # e.g. "t_8698e68a" or "0719-finance-..."
    new_dir_name = name
    new_dir_rel = f"pipeline_output/tasks/{new_dir_name}"

    old_path = TASKS_DIR / old_dir_name
    new_path = TASKS_DIR / new_dir_name

    # 如果已经以任务名为文件夹名，跳过
    if old_dir_name == new_dir_name:
        skipped.append(f"{task_id}: 已一致 ({old_dir_name})")
        continue

    if not old_path.exists() and not old_path.is_symlink():
        skipped.append(f"{task_id}: 源文件夹不存在 ({old_dir_name})")
        continue

    if new_path.exists():
        errors.append(f"{task_id}: 目标文件夹已存在 ({new_dir_name})")
        continue

    renames.append({
        "task_id": task_id,
        "name": name,
        "old_name": old_dir_name,
        "new_name": new_dir_name,
        "old_path": old_path,
        "new_path": new_path,
    })

# 先打印计划
print("=" * 60)
print(f"需要重命名的文件夹: {len(renames)}")
print()
for r in renames:
    print(f"  {r['task_id']}: {r['old_name']} → {r['new_name']}")

if errors:
    print(f"\n❌ 错误 ({len(errors)}):")
    for e in errors:
        print(f"  {e}")

if skipped:
    print(f"\n⏭ 跳过 ({len(skipped)}):")
    for s in skipped:
        print(f"  {s}")

# 执行重命名
renamed_count = 0
for r in renames:
    try:
        os.rename(r["old_path"], r["new_path"])
        print(f"  ✅ {r['old_name']} → {r['new_name']}")
        renamed_count += 1
    except OSError as e:
        print(f"  ❌ {r['old_name']} → {r['new_name']}: {e}")
        errors.append(str(e))

# 更新 tasks.json
if renamed_count > 0:
    with open(TASKS_FILE) as f:
        data = json.load(f)

    for r in renames:
        for task in data["tasks"]:
            if task["id"] == r["task_id"]:
                old_val = task["output_dir"]
                task["output_dir"] = f"pipeline_output/tasks/{r['new_name']}"
                print(f"  📝 tasks.json: {r['task_id']} output_dir: {old_val} → {task['output_dir']}")
                break

    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n完成: {renamed_count} 个文件夹重命名, tasks.json 已更新")
