#!/usr/bin/env python3
"""统计一个 opus4.8 任务的 assistant 轨迹中 thinkingSignature 出现次数。"""
import json
import os
import glob as glob_mod
import sys
from pathlib import Path

TASK_DIR = Path("data_viewer/pipeline_output/tasks/t_8698e68a")
ORIGIN = TASK_DIR / "origin"
SAMPLE_SIZE = 2000

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
    # Hermes
    for pat in ["profiles/assistant*/sessions/*.json", "profiles/main/sessions/*.json"]:
        for fp in session_dir.glob(pat):
            return fp
    return None

# Gather all sessions with trajectory files
sessions = sorted([d.name for d in ORIGIN.iterdir() if d.is_dir()])
print(f"Total session dirs: {len(sessions)}")

all_traj = []
for s in sessions:
    fp = find_traj(ORIGIN / s)
    if fp:
        all_traj.append(s)

print(f"With assistant trajectory: {len(all_traj)}")

# Full analysis (11K is reasonable to process all)
import random
random.seed(42)

# Use all trajectories since 11K is manageable
results = []
count = 0
for s in all_traj:
    fp = find_traj(ORIGIN / s)
    n = count_think_sig_in_file(fp)
    results.append(n)
    count += 1
    if count % 2000 == 0:
        print(f"  processed {count}/{len(all_traj)}")

total = sum(results)
avg = total / len(results) if results else 0
med = sorted(results)[len(results)//2]

print(f"\n===== 统计结果 =====")
print(f"任务: openclaw_生活学习_part3_0727_lessrubrics_opus48_oc_0727_0138")
print(f"总 session 数: {len(sessions)}")
print(f"有轨迹文件: {len(all_traj)}")
print(f"统计轨迹数: {len(results)}")
print(f"thinkingSignature 总数: {total}")
print(f"平均每条轨迹: {avg:.2f}")
print(f"中位数: {med}")

from collections import Counter
dist = Counter(results)
print(f"\n分布 (出现次数前20):")
for val, cnt in sorted(dist.most_common(20)):
    pct = cnt / len(results) * 100
    bar = "█" * int(pct / 2)
    print(f"  {val:>4}次: {cnt:>5}条 ({pct:5.1f}%) {bar}")

# Save
out = TASK_DIR / "think_sig_stats.json"
with open(out, "w") as f:
    json.dump({
        "task": "t_8698e68a",
        "name": "openclaw_生活学习_part3_0727_lessrubrics_opus48_oc_0727_0138",
        "total_sessions": len(sessions),
        "with_trajectory": len(all_traj),
        "stat_count": len(results),
        "total_think_sig": total,
        "avg_per_trajectory": round(avg, 4),
        "median": med,
    }, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {out}")
