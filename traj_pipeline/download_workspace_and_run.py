# -*- coding: utf-8 -*-
"""
从一个 OBS **workspace 根路径** 按需下载各 task 的必要文件, 然后用 traj_stats 分析,
产出与平台一致 schema 的 filter_stats.json。

与 download_and_run.py 的区别:
  - download_and_run.py 吃两个 session_analysis 目录(assistant + evaluator)。
  - 本脚本吃一个 workspace 根路径, 其下每个子目录是一个 task, task 内同时含 assistant 与
    evaluator 的原始轨迹。整份 workspace 很大, 所以**按需下载**——每个 task 只取:
      * assistant 非 trajectory 的 session jsonl : agents/assistant*/sessions/*.jsonl (排除 *.trajectory.jsonl)
      * 主 log                                    : logs/<task>.log            (排除 *_use.log)
    实测每 task 只需 ~79KB(全量约 1.37MB), 省 ~94%。

口径(与现有平台 filter_stats.json 对齐, 供总览漏斗跨来源相加):
  filtered_count    = 可用轨迹数(所有 task 的 assistant 轨迹, 无「≥3 工具调用/纯轮」门槛)
  with_eval_count   = 有 evaluator 首轮(turn=1)裁决的数(含 completion=null)
  completion_ge_0.5 = turn=1 completion 为数值且 >= 0.5
  completion_eq_1   = turn=1 completion == 1
  dropped_count     = 0 (workspace 无 session_report.xlsx 的错误备注筛选)

用法:
  python download_workspace_and_run.py <workspace_obs> <out_dir> [--obsutil PATH] [--max-tasks N]

示例:
  python download_workspace_and_run.py \
    "obs://rl-agentdata/openclaw_trajs/traj_glm_oc_0719_1228/" \
    output --obsutil ~/bin/obsutil
"""
import os
import sys
import json
import time
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(HERE)          # traj_stats.py 在仓库根目录
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import traj_stats  # noqa: E402  复用 process_root / 分析函数

DEFAULT_OBSUTIL = r"D:\tools\obsutil_windows_amd64_5.8.3\obsutil.exe"

# 下这几类文件, 排除体积大的 trajectory 轨迹与无关的 *_use.log。
# 两套 harness 的取分/详情路径不同, 只下各自真正会被读到的文件:
#
# openclaw (agents/*/sessions/*.jsonl):
#   - assistant 会话 jsonl : 统计工具调用/纯轮, 详情渲染 assistant 轨迹
#   - evaluator 会话 jsonl : 兜底裁决来源(log 无数值分时回退这里取分), 及详情渲染 evaluator 轨迹 —— 必需
#   - 主 log               : 首轮 evaluator 裁决(有的任务裁决只写在这里) —— 必需
#   assistant 侧目录名有的是 assistant1、有的是 main(不同 harness 变体), 两者都要下
#
# Hermes (profiles/*/sessions/*.json + logs/trajectories/*/query1.json):
#   - profiles assistant sessions : 统计工具调用/纯轮, 详情渲染 assistant 轨迹 —— 必需
#   - query1.json                 : 分数(evaluations[])与详情裁决都在这里, 是 Hermes 唯一取分来源 —— 必需
#   - profiles evaluator sessions : Hermes 取分只读 query1.json, 详情页 evaluator 恒为 None, 从不读它 —— 不下(省 ~176K/task)
#   - profiles/*/logs/*.log       : Hermes 统计路径完全不读(只 openclaw 读主 log) —— 不下(省 ~72K/task)
#
# 因此:
#   - evaluator sessions 只保留 openclaw 的 .jsonl(*evaluator*sessions*.jsonl), Hermes 的 .json 不进 include
#   - profiles sessions 只 include assistant 侧(assistant*/main), 不用会连带 evaluator 的通配 *profiles/*/sessions/*.json
#   - log 只保留 openclaw 主 log(logs/<task>.log), 用 EXCLUDE 挡掉 profiles 内的 agent.log/errors.log
# 注意 exclude 的 *trajectory* 会误伤 logs/trajectories/ 路径, 故 query1.json 的 include
# 走完整路径 *logs/trajectories/*query*.json, 并在 EXCLUDE 用更精确的 *.trajectory.jsonl
INCLUDE_PATTERNS = ["*assistant*sessions*.jsonl", "*agents/main/sessions/*.jsonl",
                    "*logs*.log", "*evaluator*sessions*.jsonl",
                    "*logs/trajectories/*query*.json",
                    "*profiles/assistant*/sessions/*.json",
                    "*profiles/main/sessions/*.json",
                    "*profiles/assistant*/state.db*"]
# 宽泛的 *logs*.log include 会连带匹配一些噪音日志, 逐条 exclude 挡掉(统计/详情都不读):
#   - *profiles/*/logs/*  : Hermes 的 profiles/assistant1/logs/agent.log、errors.log
#   - *_logs/*.log        : npm 调试日志 profiles/*/home/.npm/_logs/*-debug-0.log
# 两条都不影响顶层 logs/<task>.log(openclaw/Hermes 主 log, task 根目录下第一层 logs/, 取分用)。
EXCLUDE_PATTERNS = ["*.trajectory.jsonl", "*_use.log", "*profiles/*/logs/*", "*_logs/*.log"]


def obs_leaf(obs_path):
    """路径末段目录名, 如 .../00001_xxx_q1/ -> 00001_xxx_q1。"""
    return os.path.basename(obs_path.rstrip("/").rstrip("\\"))


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def list_task_dirs(obsutil, workspace_obs):
    """用 obsutil ls -d 枚举 workspace 下的直接子目录(task), 处理 Next marker 翻页。

    返回 task 的 obs URL 列表(每个以 / 结尾)。
    """
    workspace_obs = workspace_obs if workspace_obs.endswith("/") else workspace_obs + "/"
    tasks = []
    marker = None
    while True:
        cmd = [obsutil, "ls", workspace_obs, "-d", "-limit", "1000"]
        if marker:
            cmd += ["-marker", marker]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        if res.returncode != 0:
            raise RuntimeError(f"obsutil ls 失败(退出码 {res.returncode}): {res.stdout}\n{res.stderr}")

        next_marker = None
        for raw in (res.stdout or "").splitlines():
            line = raw.strip()
            # 子目录行: 以 workspace_obs 开头、比它多一段并以 / 结尾, 排除它自身
            if line.startswith(workspace_obs) and line.endswith("/") and line != workspace_obs:
                if line not in tasks:
                    tasks.append(line)
            elif line.startswith("Next marker:"):
                next_marker = line.split(":", 1)[1].strip()

        if not next_marker:
            break
        marker = next_marker
    return tasks


def download_task(obsutil, task_obs, dest_dir):
    """按需下载单个 task 的必要文件到 dest_dir/<leaf>/ (include/exclude 过滤)。"""
    os.makedirs(dest_dir, exist_ok=True)
    task_obs = task_obs if task_obs.endswith("/") else task_obs + "/"
    cmd = [obsutil, "cp", task_obs, dest_dir, "-r", "-f"]
    for p in INCLUDE_PATTERNS:
        cmd += ["-include", p]
    for p in EXCLUDE_PATTERNS:
        cmd += ["-exclude", p]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"obsutil cp 失败(退出码 {res.returncode}) {task_obs}: "
                           f"{(res.stdout or '')[-400:]}")


def build_platform_stats(origin):
    """在 origin(下含各 task 子目录) 上跑 traj_stats, 按 workspace 门槛口径映射成 filter_stats.json 结构。

    与 traj_stats 的 7 层逐层嵌套一一对应(每档都是上一档的子集):
      L0   = 总轨迹数(所有 assistant 轨迹)                    -> filtered_count + dropped_count
      L1   = ge3_and_plain_round(≥3 工具调用 且 有纯轮)        -> filtered_count
      L1.5 = L1 内有 turn=1 数值 completion(不含 null)         -> with_eval_count
      L2   = L1.5 内 completion >= 0.5                          -> completion_ge_0.5
      L3   = L1.5 内 completion == 1                            -> completion_eq_1
    L1 以下全部收敛到「通过门槛」的子集内统计; 未过门槛的轨迹计入 dropped_count。
    """
    per_task = traj_stats.process_root(origin)

    per_session = []
    kept = with_eval = ge05 = eq1 = 0
    dropped = 0
    for row in per_task:
        comp = row.get("evaluator_completion")
        has_score = isinstance(comp, (int, float))         # L1.5: 有首轮数值分(不含 null)
        # L1 门槛: 直接读 traj_stats 算好的 passed_gate(openclaw/Hermes 同式: ≥3工具调用+纯轮),
        # 不在此重算, 保证两套 harness 口径统一。
        passed = bool(row.get("passed_gate"))
        if passed:
            kept += 1
            if has_score:                                  # L1.5: 门槛内且有数值分
                with_eval += 1
            if has_score and comp >= 0.5:                  # L2
                ge05 += 1
            if has_score and comp == 1:                    # L3
                eq1 += 1
        else:
            dropped += 1
        per_session.append({
            "session": row["task"],           # 用 task 目录名; 详情暂不支持
            "passed_gate": passed,            # 是否通过 L1 门槛(≥3工具调用+纯轮)
            "has_eval": has_score,            # L1.5 口径: 有 turn=1 数值 completion
            "eval_qc": "",
            "completion": comp,
            "tool_calls": row.get("tool_calls"),
            "plain_rounds": row.get("plain_rounds"),
            "trajectory": row.get("trajectory"),
            "harness": row.get("harness", "openclaw"),
        })

    return {
        "filtered_count":    kept,            # L1
        "with_eval_count":   with_eval,       # L1.5
        "completion_ge_0.5": ge05,            # L2
        "completion_eq_1":   eq1,             # L3
        "dropped_count":     dropped,         # 未过 L1 门槛的轨迹(L0 - L1)
        "note": ("来源=workspace(原始轨迹按需下载); L0=总轨迹数, "
                 "L1=ge3_and_plain_round(≥3工具调用+有纯轮), "
                 "L1.5=L1内有 turn=1 数值 completion(不含 null), "
                 "L2/L3=L1.5内 turn=1 completion>=0.5 / ==1"),
        "source_type": "workspace",
        "per_session": per_session,
    }


def main():
    ap = argparse.ArgumentParser(
        description="从 OBS workspace 根路径按需下载并统计(平台对齐口径)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workspace_obs", help="workspace 根路径: obs://.../<batch>/ (其下每个子目录是一个 task)")
    ap.add_argument("out_dir",       help="输出目录(下载落到 <out_dir>/origin/, 结果 filter_stats.json 落到 <out_dir>/)")
    ap.add_argument("--obsutil", default=DEFAULT_OBSUTIL, help=f"obsutil 路径(默认 {DEFAULT_OBSUTIL})")
    ap.add_argument("--max-tasks", type=int, default=0, help="仅处理前 N 个 task(0=全部, 调试用)")
    ap.add_argument("--concurrency", type=int, default=8, help="并发下载的 task 数(默认 8)")
    a = ap.parse_args()

    if not os.path.exists(a.obsutil):
        ap.error(f"obsutil 不存在: {a.obsutil}")

    origin = os.path.join(a.out_dir, "origin")
    os.makedirs(origin, exist_ok=True)

    print(f"[1] 枚举 task 子目录 <- {a.workspace_obs}", flush=True)
    tasks = list_task_dirs(a.obsutil, a.workspace_obs)
    if a.max_tasks and a.max_tasks > 0:
        tasks = tasks[:a.max_tasks]
    total = len(tasks)
    print(f"      共 {total} 个 task", flush=True)
    if total == 0:
        sys.exit(f"[error] workspace 下未枚举到任何 task 子目录: {a.workspace_obs}")

    concurrency = max(1, a.concurrency)
    print(f"[2] 按需下载(每 task 仅 assistant sessions jsonl + 主 log; 并发 {concurrency})", flush=True)
    t0 = time.time()
    failed = []
    done = 0
    lock = threading.Lock()
    last_report = [0.0]   # 用列表以便闭包内可改

    def worker(task_obs):
        """下载单个 task; 返回 (leaf, error_or_None)。异常不外抛, 单 task 失败不拖垮整批。"""
        leaf = obs_leaf(task_obs)
        try:
            download_task(a.obsutil, task_obs, origin)
            return leaf, None
        except Exception as exc:
            return leaf, str(exc)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            leaf, err = fut.result()
            with lock:
                done += 1
                if err:
                    failed.append(leaf)
                    print(f"      [warn] 跳过(下载失败) {leaf}: {err}", flush=True)
                # 进度限频: 每秒最多一行, 或在收尾时打印, 避免 12k task 刷屏
                now = time.time()
                if now - last_report[0] >= 1.0 or done == total:
                    last_report[0] = now
                    rate = done / (now - t0) if now > t0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"    [{done}/{total}] 已完成 (失败 {len(failed)}), "
                          f"{rate:.1f} task/s, 预计剩余 {eta:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"      下载完成: {human(dir_size(origin))} / {dt:.1f}s, 失败 {len(failed)} 个", flush=True)

    print(f"[3] 统计(平台对齐口径) -> filter_stats.json", flush=True)
    stats = build_platform_stats(origin)
    out_path = os.path.join(a.out_dir, "filter_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    total_traj = stats["filtered_count"] + stats["dropped_count"]
    print(f"[done] 结果 -> {out_path}", flush=True)
    print(f"      总轨迹数(L0)               : {total_traj}", flush=True)
    print(f"      合格轨迹 ≥3工具+纯轮(L1)   : {stats['filtered_count']}", flush=True)
    print(f"      L1内有 evaluator 裁决(L1.5): {stats['with_eval_count']}", flush=True)
    print(f"      completion >= 0.5 (L2)    : {stats['completion_ge_0.5']}", flush=True)
    print(f"      completion == 1  (L3)     : {stats['completion_eq_1']}", flush=True)


if __name__ == "__main__":
    main()
