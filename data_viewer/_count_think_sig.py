#!/usr/bin/env python3
"""下载 opus4.8 任务 t_9435d285 的 assistant 轨迹并统计 thinkingSignature 出现次数。"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
OBSUTIL = "/home/w00802407/obsutil/obsutil"
AK = "HPUAEGWMKUI3IZO3HKL0"
SK = "kuKJHgufvXcXNw7AFgRBFVsKsMCGD0pPFMSB12Tt"
ENDPOINT = "obs.cn-east-4.myhuaweicloud.com"

WORKSPACE_OBS = "obs://s3-asset-b-hd-cce-aifm-nlp-exp/openclaw_trajs/0728_lessrubrics_opus48_oc_0728_1652/"
ORIGIN_DIR = HERE / "pipeline_output" / "tasks" / "t_9435d285" / "origin"

# 已存在的 session 目录列表
existing_sessions = sorted([d for d in os.listdir(ORIGIN_DIR) if os.path.isdir(ORIGIN_DIR / d)])
print(f"已有 {len(existing_sessions)} 个 session 目录")

# 先列一下 OBS 根目录，看看结构
def obs_list(path, depth=1):
    cmd = [OBSUTIL, "ls", path, "-i", AK, "-k", SK, "-e", ENDPOINT, "-limit", "50", "-depth", str(depth)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.stdout, r.stderr

print("查看 OBS 根目录结构...")
out, err = obs_list(WORKSPACE_OBS, depth=1)
print("stdout:", out[:1000])
if err:
    print("stderr:", err[:500])
