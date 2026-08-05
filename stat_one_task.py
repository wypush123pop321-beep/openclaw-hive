#!/usr/bin/env python3
"""统计单个任务的 thinkingSignature 按层级分布。"""
import json
from pathlib import Path
from collections import defaultdict

TASK_NAME = "openclaw_生活学习_part3_0727_lessrubrics_opus48_oc_0727_0138"
TASK_DIR = Path(f"data_viewer/pipeline_output/tasks/{TASK_NAME}")
ORIGIN = TASK_DIR / "origin"

def get_level(s):
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

def count_think_sig(fp):
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

# 读 filter_stats
with open(TASK_DIR / "filter_stats.json") as f:
    data = json.load(f)

sessions = data.get("per_session", [])
print(f"总 sessions: {len(sessions)}")

lvl_counts = defaultdict(lambda: {"traj": 0, "ts": 0})

for s in sessions:
    sn = s.get("session", "")
    if not sn:
        continue
    fp = find_traj(ORIGIN / sn)
    if not fp:
        continue

    level = get_level(s)
    n = count_think_sig(fp)

    lvl_counts[level]["traj"] += 1
    lvl_counts[level]["ts"] += n

    if s.get("task_done"):
        lvl_counts["T_DONE"]["traj"] += 1
        lvl_counts["T_DONE"]["ts"] += n

print(f"\n{'='*70}")
print(f"任务: {TASK_NAME}")
print(f"{'='*70}")
print(f"{'层级':<10} {'轨迹数':>8} {'thinkSig总数':>12} {'平均值':>8} {'占比':>8}")
print(f"{'-'*50}")
for lvl in ["L0", "L1", "L1.5", "L2", "L3", "L1~L3"]:
    if lvl == "L1~L3":
        traj = sum(lvl_counts[l]["traj"] for l in ["L1", "L1.5", "L2", "L3"])
        ts = sum(lvl_counts[l]["ts"] for l in ["L1", "L1.5", "L2", "L3"])
    else:
        c = lvl_counts[lvl]
        traj = c["traj"]
        ts = c["ts"]
    pct = traj / len(sessions) * 100
    avg = ts / traj if traj > 0 else 0
    print(f"{lvl:<10} {traj:>8} {ts:>12} {avg:>8.2f} {pct:>7.1f}%")

# T_DONE 单独一行
c = lvl_counts["T_DONE"]
pct = c["traj"] / len(sessions) * 100
avg = c["ts"] / c["traj"] if c["traj"] > 0 else 0
print(f"{'T_DONE':<10} {c['traj']:>8} {c['ts']:>12} {avg:>8.2f} {pct:>7.1f}%")
print(f"{'='*70}")

# 保存
out = {"task": TASK_NAME, "total_sessions": len(sessions)}
for lvl in ["L0", "L1", "L1.5", "L2", "L3", "L1~L3", "T_DONE"]:
    if lvl == "L1~L3":
        traj = sum(lvl_counts[l]["traj"] for l in ["L1", "L1.5", "L2", "L3"])
        ts = sum(lvl_counts[l]["ts"] for l in ["L1", "L1.5", "L2", "L3"])
    else:
        c = lvl_counts[lvl]
        traj = c["traj"]
        ts = c["ts"]
    out[lvl] = {"traj_count": traj, "think_sig_sum": ts,
                "avg": round(ts / traj, 4) if traj > 0 else 0}

with open(TASK_DIR / "think_sig_stats.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {TASK_DIR / 'think_sig_stats.json'}")
