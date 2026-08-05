#!/usr/bin/env python3
"""下载 opus4.8 任务 t_9435d285 的 assistant 轨迹并统计 thinkingSignature。"""
import json
import os
import subprocess
import sys
import glob as glob_mod
from pathlib import Path

HERE = Path(__file__).parent.parent  # data_viewer/
OBSUTIL = "/home/w00802407/obsutil/obsutil"
OBS_CRED = ["-i", "HPUAEGWMKUI3IZO3HKL0", "-k", "kuKJHgufvXcXNw7AFgRBFVsKsMCGD0pPFMSB12Tt", "-e", "obs.cn-east-4.myhuaweicloud.com"]

WORKSPACE_OBS = "obs://s3-asset-b-hd-cce-aifm-nlp-exp/openclaw_trajs/0728_lessrubrics_opus48_oc_0728_1652/"
ORIGIN_DIR = HERE / "pipeline_output" / "tasks" / "t_9435d285" / "origin"

def obs_cmd(args):
    cmd = [OBSUTIL] + args + OBS_CRED
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"[WARN] obsutil rc={r.returncode}: {r.stderr[:200]}")
    return r.stdout

def download_session_traj(session):
    """按需下载一个 session 的 assistant 轨迹文件。"""
    sess_dir = ORIGIN_DIR / session
    # 检查是否已有轨迹文件
    traj_files = []
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        traj_files.extend(sess_dir.glob(pat))
    traj_files = [f for f in traj_files if "trajectory" not in f.name]
    if traj_files:
        return traj_files[0]  # 已有

    # 从 OBS 下载该 session 的 assistant 轨迹
    obs_path = f"{WORKSPACE_OBS}{session}/"
    dest = str(ORIGIN_DIR)
    include_patterns = ["*assistant*sessions*.jsonl", "*agents/main/sessions/*.jsonl"]
    cmd = ["cp", obs_path, dest, "-r", "-f"]
    for p in include_patterns:
        cmd += ["-include", p]
    cmd += ["-exclude", "*.trajectory.jsonl"]

    try:
        out = obs_cmd(cmd)
    except subprocess.TimeoutExpired:
        return None

    # 再次检查
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        cands = list(sess_dir.glob(pat))
        cands = [f for f in cands if "trajectory" not in f.name]
        if cands:
            return cands[0]
    return None

def count_thinking_signature(traj_path):
    """统计一个轨迹文件中 thinkingSignature 的出现次数。"""
    count = 0
    try:
        with open(traj_path, encoding="utf-8", errors="replace") as f:
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
                            count += 1
    except OSError as e:
        print(f"  读取失败 {traj_path}: {e}")
    return count


# 主流程
existing_sessions = sorted([d for d in os.listdir(ORIGIN_DIR) if (ORIGIN_DIR / d).is_dir()])
total_sessions = len(existing_sessions)
print(f"共有 {total_sessions} 个 session 目录")

# 先检查哪些已有轨迹文件
already_have = []
for s in existing_sessions:
    sess_dir = ORIGIN_DIR / s
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        cands = list(sess_dir.glob(pat))
        cands = [f for f in cands if "trajectory" not in f.name]
        if cands:
            already_have.append(s)
            break

print(f"已有轨迹文件的 session: {len(already_have)}")

# 采样统计：检查已有 + 下载一批
sampled_sessions = already_have[:]  # 先用已有的

# 然后抽样下载一批（首次尝试下载少量以测试）
import random
random.seed(42)
need = 500 - len(sampled_sessions)
if need > 0:
    candidates = [s for s in existing_sessions if s not in sampled_sessions]
    to_download = random.sample(candidates, min(need, len(candidates)))
    print(f"将下载 {len(to_download)} 个 session 样本...")
    for i, s in enumerate(to_download):
        if (i+1) % 100 == 0:
            print(f"  进度 {i+1}/{len(to_download)}")
        fp = download_session_traj(s)
        if fp:
            sampled_sessions.append(s)

print(f"最终有轨迹文件的 session 数: {len(sampled_sessions)}")

# 统计 thinkingSignature
results = []
for s in sampled_sessions:
    sess_dir = ORIGIN_DIR / s
    for pat in ["agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"]:
        cands = list(sess_dir.glob(pat))
        cands = [f for f in cands if "trajectory" not in f.name]
        if cands:
            n = count_thinking_signature(cands[0])
            # 也统计 assistant_rounds 数和 tool_calls
            results.append({"session": s, "thinking_signatures": n})
            break

total_ts = sum(r["thinking_signatures"] for r in results)
avg_ts = total_ts / len(results) if results else 0

print(f"\n===== 统计结果 =====")
print(f"统计轨迹数: {len(results)}")
print(f"thinkingSignature 总数: {total_ts}")
print(f"平均每条轨迹 thinkingSignature 数: {avg_ts:.2f}")

# 输出分布
sig_counts = sorted(set(r["thinking_signatures"] for r in results))
print(f"\n分布:")
for c in sig_counts[:20]:
    n = sum(1 for r in results if r["thinking_signatures"] == c)
    print(f"  {c}次: {n}条 ({n/len(results)*100:.1f}%)")
if len(sig_counts) > 20:
    print(f"  ... 还有 {len(sig_counts)-20} 个不同值")

# 保存结果
out_path = HERE / "think_sig_stats.json"
with open(out_path, "w") as f:
    json.dump({
        "total_sessions": total_sessions,
        "sampled_sessions": len(results),
        "total_thinking_signatures": total_ts,
        "avg_thinking_signatures_per_trajectory": round(avg_ts, 4),
        "details": results,
    }, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存到 {out_path}")
