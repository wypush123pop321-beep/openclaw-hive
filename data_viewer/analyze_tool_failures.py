#!/usr/bin/env python3
"""统计一个任务下所有轨迹里 tool_call 的失败次数(显式错误标记口径)。

快速采集口径本地只有 traj_stats_result.json / query1.json, openclaw 的全量 assistant
轨迹(.jsonl)并未下载, 故本模块先按需批量补下全量 assistant 轨迹, 再逐 session 用
traj_stats 的失败计数器统计, 结果写回 <output_dir>/filter_stats.json:
  - 顶层 tool_fail_stats = {total_tool_calls, total_tool_fails, fail_rate,
                            analyzed_sessions, analyzed_at}
  - per_session[] 每条补 tool_calls / tool_fail_count

既可被 server 后台线程 import 调用 analyze(task, progress, now_iso), 也可 CLI 单跑:
    python3 analyze_tool_failures.py <task_id> [<task_id> ...]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import traj_stats  # noqa: E402

from server import (  # noqa: E402
    load_tasks, load_config, origin_dir, stats_path, _obs_cred_args_for_task,
)

# 全量 assistant 轨迹: openclaw 的 agents/*/sessions/*.jsonl + Hermes 的 query1.json。
TRAJ_INCLUDE = [
    "*agents/assistant*/sessions/*.jsonl",
    "*agents/main/sessions/*.jsonl",
    "*logs/trajectories/*query*.json",
]
TRAJ_EXCLUDE = ["*.trajectory.jsonl", "*_use.log", "*profiles/*/logs/*", "*_logs/*.log"]


def _flatten_leaf(origin: Path, workspace_obs: str) -> None:
    """obsutil cp <prefix>/ dest 会多建一层 <leaf> 目录(origin/<leaf>/<session>/...),
    与既有 origin/<session>/ 结构差一层, 就地剥掉并合并进已有 session 目录。"""
    leaf = workspace_obs.rstrip("/").rsplit("/", 1)[-1]
    nested = origin / leaf
    if not nested.is_dir():
        return
    for sess_dir in nested.iterdir():
        if not sess_dir.is_dir():
            continue
        for src in sess_dir.rglob("*"):
            if not src.is_file():
                continue
            dst = origin / sess_dir.name / src.relative_to(sess_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                src.replace(dst)
    shutil.rmtree(nested, ignore_errors=True)


def download_trajectories(task: dict) -> None:
    """批量补下该任务所有 session 的全量 assistant 轨迹到 origin(增量; 已存在的按 -f 覆盖)。"""
    workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
    if not workspace_obs:
        print("  [跳过下载] 任务无 workspace_obs(非 workspace 来源)")
        return
    cfg = load_config()
    origin = origin_dir(task["output_dir"])
    origin.mkdir(parents=True, exist_ok=True)
    cmd = [cfg["obsutil_path"], "cp", workspace_obs + "/", str(origin), "-r", "-f"]
    for p in TRAJ_INCLUDE:
        cmd += ["-include", p]
    for p in TRAJ_EXCLUDE:
        cmd += ["-exclude", p]
    cmd += _obs_cred_args_for_task(task)
    print(f"  下载全量 assistant 轨迹: {workspace_obs} -> {origin}")
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print(f"  [警告] obsutil cp 退出码 {res.returncode}: {(res.stderr or '')[-500:]}")
    _flatten_leaf(origin, workspace_obs)


def _session_counts(task_dir: Path):
    """定位单个 session 的全量 assistant 轨迹并计数, 返回 (tool_calls, tool_fails)。
    优先 openclaw 的 agents/assistant*/sessions/*.jsonl(评测方 evaluator 不计; 无 assistant*
    时回退 main), 无则回退 Hermes 的 logs/trajectories/*/query*.json。都没有则 (0, 0)。
    一个 session 目录可能含多个轨迹文件(续跑/子 agent 各一份 jsonl, 或多个 query 子目录),
    须逐个累加而非只取首个, 否则会漏计。"""
    # openclaw: assistant* 优先, 缺失才用 main, 避免二者并存时重复计数; 命中的这一类全部累加。
    for pattern in ("agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"):
        jsonls = [p for p in sorted(task_dir.glob(pattern)) if "trajectory" not in p.name]
        if jsonls:
            calls = fails = 0
            for jsonl in jsonls:
                c, f = traj_stats.count_tool_failures_openclaw(str(jsonl))
                calls += c
                fails += f
            return (calls, fails)
    # Hermes: 每个 trajectories 子目录一份 query*.json, 各自是一段 turn 集, 全部累加。
    calls = fails = 0
    found = False
    for q in sorted(task_dir.glob("logs/trajectories/*/query*.json")):
        found = True
        c, f = traj_stats.count_tool_failures_query1(str(q))
        calls += c
        fails += f
    return (calls, fails) if found else (0, 0)


def analyze(task: dict, progress=None, now_iso: str = None) -> dict:
    """下载 + 分析 + 写回。progress(done, total) 可选, 用于后台线程上报进度。
    返回写入 filter_stats.json 的 tool_fail_stats。"""
    download_trajectories(task)

    sp = stats_path(task["output_dir"])
    if not sp.exists():
        raise FileNotFoundError(f"无 filter_stats.json: {sp}")
    with open(sp, encoding="utf-8") as f:
        data = json.load(f)
    sessions = data.get("per_session", [])
    origin = origin_dir(task["output_dir"])

    total_calls = total_fails = analyzed = 0
    n = len(sessions)
    for i, row in enumerate(sessions):
        session = row.get("session")
        if not session or "/" in session or ".." in session:
            row["tool_calls"] = row.get("tool_calls", 0)
            row["tool_fail_count"] = 0
            continue
        task_dir = origin / session
        calls, fails = _session_counts(task_dir) if task_dir.is_dir() else (0, 0)
        row["tool_calls"] = calls
        row["tool_fail_count"] = fails
        total_calls += calls
        total_fails += fails
        analyzed += 1
        if progress and (i % 200 == 0 or i == n - 1):
            progress(i + 1, n)

    tf_stats = {
        "total_tool_calls": total_calls,
        "total_tool_fails": total_fails,
        "fail_rate": round(total_fails / total_calls, 4) if total_calls else 0.0,
        "analyzed_sessions": analyzed,
        "analyzed_at": now_iso,
    }
    data["tool_fail_stats"] = tf_stats
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return tf_stats


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    from datetime import datetime, timezone
    tasks = {t["id"]: t for t in load_tasks()}
    for tid in sys.argv[1:]:
        t = tasks.get(tid)
        if not t:
            print(f"[未找到任务] {tid}")
            continue
        print(f"[{tid}] {t.get('name', '?')}")
        res = analyze(t, now_iso=datetime.now(timezone.utc).isoformat())
        print(f"  tool_fail_stats: {res}")


if __name__ == "__main__":
    main()
