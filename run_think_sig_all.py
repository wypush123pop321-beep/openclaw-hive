#!/usr/bin/env python3
"""统计所有 opus4.8 任务的 thinkingSignature 平均值。"""
import json
import os
import random
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data_viewer/pipeline_output/tasks")

# 从 tasks.json 读出所有 opu4.8 任务
with open("data_viewer/tasks.json") as f:
    all_tasks = json.load(f)["tasks"]

opus_tasks = [t for t in all_tasks if "opu4.8" in t.get("groups", [])]
print(f"opu4.8 任务总数: {len(opus_tasks)}")

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

def find_traj(session_dir):
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        for fp in session_dir.glob(pat):
            if "trajectory" not in fp.name:
                return fp
    for pat in ["profiles/assistant*/sessions/*.json", "profiles/main/sessions/*.json"]:
        for fp in session_dir.glob(pat):
            return fp
    return None

random.seed(42)
MAX_SAMPLE = 3000  # 每个任务最多采样3000条

all_results = {}
grand_total_ts = 0
grand_total_traj = 0

for t in opus_tasks:
    tid = t["id"]
    name = t["name"]
    origin = Path("data_viewer") / t["output_dir"] / "origin"

    if not origin.is_dir():
        print(f"\n[{tid}] {name}")
        print(f"  ❌ 无本地数据 (origin 不存在)")
        all_results[tid] = {"name": name, "status": "no_data", "total_sessions": 0}
        continue

    sessions = sorted([d.name for d in origin.iterdir() if d.is_dir()])

    # 找有轨迹文件的 session
    with_traj = []
    for s in sessions:
        fp = find_traj(origin / s)
        if fp:
            with_traj.append(s)

    total_sessions = len(sessions)
    traj_available = len(with_traj)

    if traj_available == 0:
        print(f"\n[{tid}] {name}")
        print(f"  session 数: {total_sessions}, 有轨迹文件: 0")
        all_results[tid] = {"name": name, "status": "no_traj", "total_sessions": total_sessions}
        continue

    # 采样
    sample = with_traj
    if len(sample) > MAX_SAMPLE:
        sample = random.sample(sample, MAX_SAMPLE)

    results = []
    for s in sample:
        fp = find_traj(origin / s)
        n = count_think_sig_in_file(fp)
        results.append(n)

    total_ts = sum(results)
    avg = total_ts / len(results) if results else 0
    med = sorted(results)[len(results)//2]

    dist = Counter(results)

    print(f"\n[{tid}] {name}")
    print(f"  session 数: {total_sessions}, 有轨迹: {traj_available}, 采样: {len(results)}")
    print(f"  thinkingSignature 总数: {total_ts}")
    print(f"  平均每条: {avg:.2f}")
    print(f"  中位数: {med}")
    print(f"  0次占比: {dist.get(0, 0)/len(results)*100:.1f}%")

    grand_total_ts += total_ts
    grand_total_traj += len(results)

    all_results[tid] = {
        "name": name,
        "status": "done",
        "total_sessions": total_sessions,
        "traj_available": traj_available,
        "sampled": len(results),
        "total_think_sig": total_ts,
        "avg_per_trajectory": round(avg, 4),
        "median": med,
        "pct_zero": round(dist.get(0, 0)/len(results)*100, 1),
    }

# 总平均
all_avg = grand_total_ts / grand_total_traj if grand_total_traj else 0
print(f"\n{'='*60}")
print(f"所有 opus4.8 任务汇总")
print(f"统计轨迹总数: {grand_total_traj}")
print(f"thinkingSignature 总数: {grand_total_ts}")
print(f"加权平均每条: {all_avg:.2f}")

all_results["__summary__"] = {
    "total_tasks": len(opus_tasks),
    "total_trajectories_sampled": grand_total_traj,
    "total_think_sig": grand_total_ts,
    "weighted_avg_per_trajectory": round(all_avg, 4),
}

with open("data_viewer/think_sig_all_results.json", "w") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存到 data_viewer/think_sig_all_results.json")
