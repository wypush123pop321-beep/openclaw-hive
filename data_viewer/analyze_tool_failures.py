#!/usr/bin/env python3
"""统计一个任务下所有轨迹里 tool_call 的失败情况与 Windows 环境工具调用。

快速采集口径本地只有 traj_stats_result.json / query1.json, openclaw 的全量 assistant
轨迹(.jsonl)并未下载, 故本模块先按需批量补下全量 assistant 轨迹, 再逐 session 用
traj_stats 的计数器统计, 结果写回 <output_dir>/filter_stats.json:
  - 顶层 tool_fail_stats = {total_tool_calls, total_tool_fails, traj_total,
                            traj_with_fail, fail_rate, analyzed_sessions, analyzed_at}
    fail_rate = traj_with_fail / traj_total, 即「出现过调用失败的轨迹数 / 总轨迹数」,
    按 session 粒度(每个 session 算一条轨迹), 分母只算确实有轨迹数据的 session
    (目录缺失或目录里没有轨迹文件的 session 不计入), 不再按调用总次数加权。
    total_tool_calls / total_tool_fails 保留原始失败调用次数作明细; analyzed_sessions 与
    traj_total 同值(有轨迹数据的 session 数), 保留为兼容字段。
  - 顶层 win_stats = {total_tool_calls, win_tool_calls, win_rate, traj_total,
                      traj_with_win, win_traj_rate, analyzed_at}
    win_rate = win_tool_calls / total_tool_calls(调用次数占比); win_traj_rate =
    traj_with_win / traj_total(出现过工具误用/环境不符调用的轨迹占比)。口径(宽)见
    traj_stats(count_tool_stats_* 顶部注释): ① 调用了本环境不存在的工具(toolResult 里
    "Tool X not found", 任意 X 均计——Claude-Code/Windows 工具名、被误当工具的 shell
    命令、拼写幻觉, 经 _tool_not_found_count) / ② 入参含 file_path 键 / ③ 入参为
    Windows 风格路径(②③见 _is_windows_tool_call)。字段名沿用 win_* 仅为兼容。与
    tool_fail_stats 同一遍扫描算得, 不额外下载轨迹。
  - per_session[] 每条补 tool_calls / tool_fail_count / win_tool_calls

既可被 server 后台线程 import 调用 analyze(task, progress, now_iso), 也可 CLI 单跑:
    python3 analyze_tool_failures.py <task_id> [<task_id> ...]
"""
import json
import shutil
import subprocess
import sys
import threading
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
    """兼容旧版下载布局: 旧版 obsutil cp 会多建一层 <leaf> 目录(origin/<leaf>/<session>/...),
    与既有 origin/<session>/ 结构差一层, 就地剥掉并合并进已有 session 目录。
    新版下载已用 -flat 直接落 origin/<session>/, 本函数仅负责清理历史遗留的 <leaf> 目录。"""
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


def _expected_traj_sessions(task: dict) -> int:
    """下载进度分母: filter_stats.json 的 per_session 条数(预期有轨迹的 session 数)。"""
    sp = stats_path(task["output_dir"])
    if not sp.exists():
        return 0
    try:
        with open(sp, encoding="utf-8") as f:
            return len(json.load(f).get("per_session", []))
    except (json.JSONDecodeError, OSError):
        return 0


def _count_downloaded_traj_sessions(origin: Path) -> int:
    """粗略统计 origin 下已落地轨迹文件的 session 数(下载进度用)。
    一次 glob 拿全部候选轨迹文件, 按 session(origin 下一级目录名)去重;
    evaluator 侧不计。与 _session_counts 的定位口径一致, 但只数「session 有无轨迹」,
    不拆具体文件, 保证几秒一次的轮询开销可控。"""
    sessions = set()
    for p in origin.glob("*/agents/*/sessions/*.jsonl"):
        if "trajectory" in p.name:
            continue
        parts = p.relative_to(origin).parts
        if len(parts) < 5 or parts[1] != "agents" or parts[2].startswith("evaluator"):
            continue
        sessions.add(parts[0])
    for q in origin.glob("*/logs/trajectories/*/query*.json"):
        if len(q.relative_to(origin).parts) >= 4:
            sessions.add(q.relative_to(origin).parts[0])
    return len(sessions)


def download_trajectories(task: dict, progress=None) -> None:
    """批量补下该任务所有 session 的全量 assistant 轨迹到 origin(增量)。
    -flat 去掉 OBS 前缀末段(leaf), 直接落 origin/<session>/(与既有结构一致);
    -u 只拉取有变化的源, 已存在且未变化的本地文件跳过, 重复「统计工具失败」
    不会全量重下; -f 仅为断点续传不挂起, 不强制覆盖。

    progress(done, total, phase="download") 可选: obsutil cp 是单个子进程、不暴露
    逐文件进度, 故开后台线程每 4s 用本地已落盘轨迹 session 数近似上报
    (total 取 per_session 条数, done 取 origin 下已有轨迹文件的 session 数)。"""
    workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
    if not workspace_obs:
        print("  [跳过下载] 任务无 workspace_obs(非 workspace 来源)")
        return
    cfg = load_config()
    origin = origin_dir(task["output_dir"])
    origin.mkdir(parents=True, exist_ok=True)
    cmd = [cfg["obsutil_path"], "cp", workspace_obs + "/", str(origin), "-r", "-f", "-u", "-flat"]
    for p in TRAJ_INCLUDE:
        cmd += ["-include", p]
    for p in TRAJ_EXCLUDE:
        cmd += ["-exclude", p]
    cmd += _obs_cred_args_for_task(task)
    print(f"  下载全量 assistant 轨迹: {workspace_obs} -> {origin}")

    total = _expected_traj_sessions(task)
    stop = threading.Event()

    def monitor():
        # 先立即上报一次(增量下载时本地已有文件, 一开始就接近 100%), 之后每 4s 一次。
        while not stop.is_set():
            try:
                done = _count_downloaded_traj_sessions(origin)
            except OSError:
                done = 0
            if stop.is_set():
                break  # 停止期间不再上报, 防止迟到的 phase="download" 覆盖 analyze 阶段
            if progress:
                progress(done, total, phase="download")
            if stop.wait(4):
                break

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    finally:
        stop.set()
        mon.join(timeout=2)
    if res.returncode != 0:
        print(f"  [警告] obsutil cp 退出码 {res.returncode}: {(res.stderr or '')[-500:]}")
    _flatten_leaf(origin, workspace_obs)


def _session_counts(task_dir: Path):
    """定位单个 session 的全量 assistant 轨迹并计数, 返回
    (tool_calls, tool_fails, win_tool_calls, r1, r2, r3, found)。
    优先 openclaw 的 agents/assistant*/sessions/*.jsonl(评测方 evaluator 不计; 无 assistant*
    时回退 main), 无则回退 Hermes 的 logs/trajectories/*/query*.json。都没有则
    (0, 0, 0, 0, 0, 0, False)。found 表示该 session 确实存在轨迹文件(无轨迹数据时不算「一条
    轨迹」, 不进分母)。一个 session 目录可能含多个轨迹文件(续跑/子 agent 各一份 jsonl, 或多个
    query 子目录), 须逐个累加而非只取首个, 否则会漏计。
    win_tool_calls = 该 session 内命中「工具误用/环境不符」宽口径的工具调用数;
    r1/r2/r3 = 3 条规则各自命中的调用数(见 traj_stats.count_tool_rule_stats_*)。
    """
    # openclaw: assistant* 优先, 缺失才用 main, 避免二者并存时重复计数; 命中的这一类全部累加。
    for pattern in ("agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"):
        jsonls = [p for p in sorted(task_dir.glob(pattern)) if "trajectory" not in p.name]
        if jsonls:
            calls = fails = win = r1 = r2 = r3 = 0
            for jsonl in jsonls:
                c, f, w, a1, a2, a3 = traj_stats.count_tool_rule_stats_openclaw(str(jsonl))
                calls += c
                fails += f
                win += w
                r1 += a1
                r2 += a2
                r3 += a3
            return (calls, fails, win, r1, r2, r3, True)
    # Hermes: 每个 trajectories 子目录一份 query*.json, 各自是一段 turn 集, 全部累加。
    calls = fails = win = r1 = r2 = r3 = 0
    found = False
    for q in sorted(task_dir.glob("logs/trajectories/*/query*.json")):
        found = True
        c, f, w, a1, a2, a3 = traj_stats.count_tool_rule_stats_query1(str(q))
        calls += c
        fails += f
        win += w
        r1 += a1
        r2 += a2
        r3 += a3
    return (calls, fails, win, r1, r2, r3, found)


def analyze(task: dict, progress=None, now_iso: str = None, skip_download: bool = False) -> dict:
    """下载 + 分析 + 写回。progress(done, total, phase=None) 可选, 用于后台线程上报进度;
    phase 为 download(补下轨迹, 以本地已落盘轨迹 session 数近似 done, total 取
    per_session 条数) 或 analyze(逐 session 分析, done/total 为已分析/总 session 数)。
    skip_download=True 时跳过增量下载(本地轨迹已齐全的快速路径), 直接逐 session 分析。
    返回写入 filter_stats.json 的 tool_fail_stats。"""
    if progress:
        progress(0, 0, phase="analyze" if skip_download else "download")
    if not skip_download:
        download_trajectories(task, progress)

    sp = stats_path(task["output_dir"])
    if not sp.exists():
        raise FileNotFoundError(f"无 filter_stats.json: {sp}")
    with open(sp, encoding="utf-8") as f:
        data = json.load(f)
    sessions = data.get("per_session", [])
    origin = origin_dir(task["output_dir"])

    total_calls = total_fails = total_win = analyzed = traj_with_fail = traj_with_win = 0
    # 分规则「命中该规则的轨迹数」(一条轨迹可同时命中多条, 故三者之和 ≥ traj_with_win)
    traj_with_r1 = traj_with_r2 = traj_with_r3 = 0
    n = len(sessions)
    if progress:
        progress(0, n, phase="analyze")
    for i, row in enumerate(sessions):
        session = row.get("session")
        if not session or "/" in session or ".." in session:
            row["tool_calls"] = row.get("tool_calls", 0)
            row["tool_fail_count"] = 0
            row["win_tool_calls"] = 0
            row["win_rule_calls"] = [0, 0, 0]
            continue
        task_dir = origin / session
        if task_dir.is_dir():
            calls, fails, win, r1, r2, r3, found = _session_counts(task_dir)
        else:
            calls, fails, win, r1, r2, r3, found = 0, 0, 0, 0, 0, 0, False
        row["tool_calls"] = calls
        row["tool_fail_count"] = fails
        row["win_tool_calls"] = win
        row["win_rule_calls"] = [r1, r2, r3]  # 该轨迹 3 条规则各自命中的调用数
        total_calls += calls
        total_fails += fails
        total_win += win
        if found:  # 只有确实存在轨迹数据的 session 才算一条「轨迹」, 进分母
            analyzed += 1
            if fails > 0:
                traj_with_fail += 1
            if win > 0:
                traj_with_win += 1
            if r1 > 0:
                traj_with_r1 += 1
            if r2 > 0:
                traj_with_r2 += 1
            if r3 > 0:
                traj_with_r3 += 1
        # 每次都显式带 phase="analyze": 下载监控线程 stop 时可能残留一记
        # phase="download" 的迟到写(mon.join 超时不杀线程), 不带上会被一直钉在下载阶段。
        if progress and (i % 200 == 0 or i == n - 1):
            progress(i + 1, n, phase="analyze")

    tf_stats = {
        "total_tool_calls": total_calls,
        "total_tool_fails": total_fails,
        "traj_total": analyzed,
        "traj_with_fail": traj_with_fail,
        "fail_rate": round(traj_with_fail / analyzed, 4) if analyzed else 0.0,
        "analyzed_sessions": analyzed,
        "analyzed_at": now_iso,
    }
    data["tool_fail_stats"] = tf_stats
    # Windows 工具调用统计: 复用上面同一遍扫描的结果(不额外下载/读轨迹)。
    # win_rate = windows 调用次数 / 总调用次数; win_traj_rate = 出现过 windows 调用的
    # 轨迹数 / 有轨迹数据的轨迹数。
    win_stats = {
        "total_tool_calls": total_calls,
        "win_tool_calls": total_win,
        "win_rate": round(total_win / total_calls, 4) if total_calls else 0.0,
        "traj_total": analyzed,
        "traj_with_win": traj_with_win,
        "win_traj_rate": round(traj_with_win / analyzed, 4) if analyzed else 0.0,
        # 分规则命中的轨迹数(不动上面的汇总口径; 一条轨迹可同时命中多条规则):
        #   r1 = 调用本环境不存在的工具; r2 = file_path 入参键; r3 = Windows 风格路径值。
        "traj_with_r1": traj_with_r1,
        "traj_with_r2": traj_with_r2,
        "traj_with_r3": traj_with_r3,
        "r1_traj_rate": round(traj_with_r1 / analyzed, 4) if analyzed else 0.0,
        "r2_traj_rate": round(traj_with_r2 / analyzed, 4) if analyzed else 0.0,
        "r3_traj_rate": round(traj_with_r3 / analyzed, 4) if analyzed else 0.0,
        "analyzed_at": now_iso,
    }
    data["win_stats"] = win_stats
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return tf_stats


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    skip = "--skip-download" in args
    task_ids = [a for a in args if not a.startswith("--")]
    from datetime import datetime, timezone
    tasks = {t["id"]: t for t in load_tasks()}
    for tid in task_ids:
        t = tasks.get(tid)
        if not t:
            print(f"[未找到任务] {tid}")
            continue
        print(f"[{tid}] {t.get('name', '?')}" + (" [跳过下载]" if skip else ""))
        res = analyze(t, now_iso=datetime.now(timezone.utc).isoformat(),
                      skip_download=skip)
        print(f"  tool_fail_stats: {res}")


if __name__ == "__main__":
    main()
