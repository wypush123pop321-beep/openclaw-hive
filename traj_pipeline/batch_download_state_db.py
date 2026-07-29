#!/usr/bin/env python3
"""逐个 task 子目录并行下载 Hermes workspace 的 state.db。"""
import subprocess, sys, os, json, concurrent.futures
from pathlib import Path

OBSUTIL = "/home/w00802407/obsutil/obsutil"
WS = "obs://rl-agentdata/openclaw_trajs/traj_glm_hermes_0719_1525/"
DEST = Path("pipeline_output/tasks/t_959ab577/origin")

# 先列出 workspace 下的所有 task 子目录
print("列出 task 子目录...")
res = subprocess.run([OBSUTIL, "ls", WS, "-d", "-limit", "1000"],
                     capture_output=True, text=True, encoding="utf-8", timeout=30)
task_dirs = [line.strip() for line in (res.stdout or "").splitlines()
             if line.strip().endswith("/") and line.strip() != WS.rstrip("/")]
print(f"找到 {len(task_dirs)} 个 task 目录")

# 只下载已有的 task 目录(state.db 约 1MB/task, 跳过不存在的)
local_task_names = set(d.name for d in DEST.iterdir() if d.is_dir()) if DEST.is_dir() else set()
print(f"本地已有 {len(local_task_names)} 个 task 目录")

def download_one(task_obs):
    leaf = os.path.basename(task_obs.rstrip("/"))
    if not leaf:
        return None
    dest = DEST / leaf / "profiles" / "assistant1"
    os.makedirs(dest, exist_ok=True)
    cmd = [OBSUTIL, "cp", f"{task_obs}profiles/assistant1/state.db", str(dest / "state.db"), "-f"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if r.returncode != 0:
            # 可能 OBS 没这个文件, 忽略
            if "does not exist" in (r.stdout or "") or "No such" in (r.stdout or ""):
                pass
            else:
                return (leaf, r.stdout[-100:])
        else:
            return (leaf, "OK")
    except subprocess.TimeoutExpired:
        return (leaf, "TIMEOUT")
    return None

# 并行下载, 每次最多 4 个
ok = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futs = [pool.submit(download_one, td) for td in task_dirs]
    for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
        r = fut.result()
        if r:
            if r[1] == "OK":
                ok += 1
                print(f"  [{i}/{len(task_dirs)}] {r[0]} ✓")
            elif r[1] is not None:
                print(f"  [{i}/{len(task_dirs)}] {r[0]} FAIL: {r[1]}")

files = list(DEST.rglob("state.db"))
print(f"\n完成。共 {ok} 个 state.db 下载成功, 本地总计 {len(files)} 个 state.db 文件")
