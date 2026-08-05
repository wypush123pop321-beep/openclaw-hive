#!/usr/bin/env python3
"""按层级(L0/L1/L1.5/L2/L3/TASK_DONE)统计 opus4.8 的 thinkingSignature。"""
import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data_viewer")
TASKS_DIR = DATA_DIR / "pipeline_output/tasks"

with open(DATA_DIR / "tasks.json") as f:
    all_tasks = json.load(f)["tasks"]

opus_tasks = [t for t in all_tasks if "opu4.8" in t.get("groups", [])]

def get_level(s):
    """与 server.py _get_session_level 一致。"""
    if s.get("has_eval") and s.get("completion") == 1:
        return "L3"
    has_eval = bool(s.get("has_eval"))
    comp = s.get("completion")
    if has_eval and isinstance(comp, (int, float)) and comp >= 0.5:
        return "L2"
    if has_eval:
        return "L1.5"
    if s.get("passed_gate"):
        return "L1"
    return "L0"

def count_think_sig_in_file(fp):
    n = 0
    try:
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "message":
                    continue
                msg = obj.get("message") or {}
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("thinkingSignature"):
                            n += 1
    except OSError:
        pass
    return n

def find_traj(origin_dir, session):
    sd = origin_dir / session
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        for fp in sd.glob(pat):
            if "trajectory" not in fp.name:
                return fp
    for pat in ["profiles/assistant*/sessions/*.json", "profiles/main/sessions/*.json"]:
        for fp in sd.glob(pat):
            return fp
    return None

# 按层级聚合: {task_id: {level: {traj_count, think_sig_sum}}}
level_stats = defaultdict(lambda: {"traj_count": 0, "think_sig_sum": 0})
task_level_summary = {}

for t in opus_tasks:
    tid = t["id"]
    name = t["name"]
    origin = DATA_DIR / t["output_dir"] / "origin"
    stats_file = DATA_DIR / t["output_dir"] / "filter_stats.json"

    if not stats_file.exists():
        print(f"[{tid}] {name} ❌ no filter_stats.json")
        continue

    with open(stats_file) as f:
        data = json.load(f)

    sessions = data.get("per_session", [])
    print(f"[{tid}] {name}: {len(sessions)} sessions in filter_stats")

    local_counts = defaultdict(lambda: {"traj_count": 0, "think_sig_sum": 0})
    processed = 0
    no_traj = 0

    for s in sessions:
        session_name = s.get("session", "")
        if not session_name:
            continue
        fp = find_traj(origin, session_name)
        if not fp:
            no_traj += 1
            continue

        level = get_level(s)
        # Also check task_done
        is_task_done = bool(s.get("task_done"))

        n = count_think_sig_in_file(fp)
        local_counts[level]["traj_count"] += 1
        local_counts[level]["think_sig_sum"] += n

        if is_task_done:
            local_counts["T_DONE"]["traj_count"] += 1
            local_counts["T_DONE"]["think_sig_sum"] += n

        processed += 1
        if processed % 2000 == 0:
            print(f"  进度: {processed}/{len(sessions)}")

    print(f"  有轨迹: {processed}, 无轨迹: {no_traj}")

    # 打印该任务结果
    task_level_summary[tid] = {"name": name}
    for lvl in ["L0", "L1", "L1.5", "L2", "L3", "T_DONE"]:
        if local_counts[lvl]["traj_count"] > 0:
            avg = local_counts[lvl]["think_sig_sum"] / local_counts[lvl]["traj_count"]
            task_level_summary[tid][lvl] = {
                "traj_count": local_counts[lvl]["traj_count"],
                "think_sig_sum": local_counts[lvl]["think_sig_sum"],
                "avg": round(avg, 2),
            }
            level_stats[lvl]["traj_count"] += local_counts[lvl]["traj_count"]
            level_stats[lvl]["think_sig_sum"] += local_counts[lvl]["think_sig_sum"]
        else:
            task_level_summary[tid][lvl] = {"traj_count": 0, "think_sig_sum": 0, "avg": 0}

    # 打印摘要行
    parts = []
    for lvl in ["L0", "L1", "L1.5", "L2", "L3", "T_DONE"]:
        c = local_counts[lvl]
        if c["traj_count"] > 0:
            parts.append(f"{lvl}:{c['traj_count']}条,avg={c['think_sig_sum']/c['traj_count']:.1f}")
    print(f"  " + " | ".join(parts))
    print()

# 总汇总
print("=" * 70)
print(f"{'层级':<10} {'轨迹数':>8} {'thinkSig总数':>12} {'平均值':>8}")
print("-" * 70)
for lvl in ["L0", "L1", "L1.5", "L2", "L3", "T_DONE"]:
    c = level_stats[lvl]
    if c["traj_count"] > 0:
        avg = c["think_sig_sum"] / c["traj_count"]
        print(f"{lvl:<10} {c['traj_count']:>8} {c['think_sig_sum']:>12} {avg:>8.2f}")

print("-" * 70)
print(f"\n注: 只统计有本地轨迹文件的任务(t_cad9e497/972054de/4a554e5a/13f146a8/"
      f"a39014a3/77aca9a2/8698e68a/9435d285/5ab19a84/3aeb8cfe)，忽略轨迹不足的任务")

# 保存
with open(DATA_DIR / "think_sig_by_level.json", "w") as f:
    json.dump({
        "by_task": task_level_summary,
        "summary": {lvl: level_stats[lvl] for lvl in ["L0", "L1", "L1.5", "L2", "L3", "T_DONE"]},
    }, f, ensure_ascii=False, indent=2)
print(f"\n已保存到 data_viewer/think_sig_by_level.json")
