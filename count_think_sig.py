#!/usr/bin/env python3
"""下载 opus4.8 任务 t_9435d285 的 assistant 轨迹并统计 thinkingSignature。"""
import json
import os
import subprocess
import random
from pathlib import Path

HERE = Path(__file__).parent
OBSUTIL = "/home/w00802407/obsutil/obsutil"
CONFIG = HERE / "data_viewer" / "config.json"
PROFILES = HERE / "data_viewer" / "obs_profiles.json"

# 从 obs_profiles.json 读 east4-asset-b 凭据
with open(PROFILES) as f:
    profiles = json.load(f)
cred = profiles.get("east4-asset-b", {})
OBS_CRED = ["-i", cred["ak"], "-k", cred["sk"], "-e", cred["endpoint"]]

# 从 tasks.json 读 workspace_obs
with open(HERE / "data_viewer" / "tasks.json") as f:
    tasks = json.load(f)["tasks"]
opus_task = next(t for t in tasks if t["id"] == "t_9435d285")
WORKSPACE_OBS = opus_task["workspace_obs"].rstrip("/") + "/"

ORIGIN_DIR = Path(opus_task["output_dir"]) / "origin"
print(f"workspace_obs: {WORKSPACE_OBS}")
print(f"origin_dir: {ORIGIN_DIR}")

def obs_run(args):
    cmd = [OBSUTIL] + args + OBS_CRED
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(f"[WARN] rc={r.returncode}: {r.stderr[:300]}")
    return r.stdout

def ensure_traj(session):
    sess_dir = ORIGIN_DIR / session
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        for fp in sess_dir.glob(pat):
            if "trajectory" not in fp.name:
                return fp
    # download
    obs_path = WORKSPACE_OBS + session + "/"
    cmd = ["cp", obs_path, str(ORIGIN_DIR), "-r", "-f",
           "-include", "*assistant*sessions*.jsonl",
           "-include", "*agents/main/sessions/*.jsonl",
           "-exclude", "*.trajectory.jsonl"]
    obs_run(cmd)
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        for fp in sess_dir.glob(pat):
            if "trajectory" not in fp.name:
                return fp
    return None

def count_think_sig(path):
    n = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
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

# Main
sessions = sorted(d.name for d in ORIGIN_DIR.iterdir() if d.is_dir())
print(f"session 总数: {len(sessions)}")

# 先检查本地已有的
have = []
for s in sessions:
    sd = ORIGIN_DIR / s
    found = False
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        for fp in sd.glob(pat):
            if "trajectory" not in fp.name:
                have.append(s)
                found = True
                break
        if found:
            break
print(f"已有轨迹的 session: {len(have)}")

# 采样: 2000 条
random.seed(42)
target = 2000
sample = have[:]
if len(sample) < target:
    more = [s for s in sessions if s not in sample]
    to_dl = random.sample(more, min(target - len(sample), len(more)))
    print(f"需下载 {len(to_dl)} 个 session...")
    for i, s in enumerate(to_dl):
        if (i+1) % 200 == 0:
            print(f"  进度 {i+1}/{len(to_dl)}")
        fp = ensure_traj(s)
        if fp:
            sample.append(s)

print(f"实际可统计: {len(sample)}")

results = []
for s in sample:
    sd = ORIGIN_DIR / s
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        for fp in sd.glob(pat):
            if "trajectory" not in fp.name:
                n = count_think_sig(fp)
                results.append({"session": s, "thinking_signatures": n})
                break
        else:
            continue
        break

total = sum(r["thinking_signatures"] for r in results)
avg = total / len(results) if results else 0
print(f"\n===== 结果 =====")
print(f"统计轨迹数: {len(results)}")
print(f"thinkingSignature 总数: {total}")
print(f"平均每条轨迹: {avg:.2f}")
print(f"中位数: {sorted(r['thinking_signatures'] for r in results)[len(results)//2]}")

from collections import Counter
dist = Counter(r["thinking_signatures"] for r in results)
print(f"\n分布 (top 15):")
for val, cnt in sorted(dist.most_common(15)):
    print(f"  {val}: {cnt}条 ({cnt/len(results)*100:.1f}%)")

out = HERE / "data_viewer" / "think_sig_stats.json"
with open(out, "w") as f:
    json.dump({
        "total_sessions": len(sessions),
        "sampled": len(results),
        "total_think_sig": total,
        "avg_per_trajectory": round(avg, 4),
        "details": results,
    }, f, ensure_ascii=False, indent=2)
print(f"\n保存到 {out}")
