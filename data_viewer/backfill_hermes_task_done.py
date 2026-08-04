#!/usr/bin/env python3
"""对已采集的 Hermes 任务回填 task_done_count。

Hermes 全量采集口径此前未下载主 log(logs/<session>.log), 而「【Task_Done】」标记只在
主 log 正文里(check_task_done_in_logs_dir 扫 <session>.log), 导致 filter_stats.json 的
task_done_count 恒为 0。本脚本:
  1) 用 workspace_obs 批量补下各 session 的主 log 到 origin(与 download_workspace_and_run.py
     的 HERMES_INCLUDE 现口径一致);
  2) 重新按 check_task_done_in_logs_dir 扫标记, 重算 task_done_count / per_session[].task_done
     / token_stats.T_DONE / char_len_stats.T_DONE, 覆盖写回 filter_stats.json。

用法:  python3 backfill_hermes_task_done.py <task_id> [<task_id> ...]
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

LOG_INCLUDE = ["*logs*.log"]
LOG_EXCLUDE = ["*.trajectory.jsonl", "*_use.log", "*profiles/*/logs/*", "*_logs/*.log"]


def download_logs(task: dict) -> None:
    """批量补下该任务所有 session 的主 log 到 origin(增量, 已存在的 -f 覆盖但同样便宜)。"""
    workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
    if not workspace_obs:
        print(f"  [跳过下载] 任务无 workspace_obs")
        return
    cfg = load_config()
    origin = origin_dir(task["output_dir"])
    cmd = [cfg["obsutil_path"], "cp", workspace_obs + "/", str(origin), "-r", "-f"]
    for p in LOG_INCLUDE:
        cmd += ["-include", p]
    for p in LOG_EXCLUDE:
        cmd += ["-exclude", p]
    cmd += _obs_cred_args_for_task(task)
    print(f"  下载主 log: {workspace_obs} -> {origin}")
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print(f"  [警告] obsutil cp 退出码 {res.returncode}: {res.stderr[-500:]}")
    # obsutil cp <prefix>/ dest 会把 prefix 末段目录一并建出, 落成 origin/<leaf>/<session>/...,
    # 与既有 origin/<session>/ 结构差一层。就地把这层剥掉(合并进已有 session 目录)。
    leaf = workspace_obs.rstrip("/").rsplit("/", 1)[-1]
    nested = origin / leaf
    if nested.is_dir():
        print(f"  展平多下的一层目录: {leaf}/")
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


def recompute(task: dict) -> None:
    sp = stats_path(task["output_dir"])
    if not sp.exists():
        print(f"  [跳过] 无 filter_stats.json: {sp}")
        return
    with open(sp, encoding="utf-8") as f:
        data = json.load(f)
    sessions = data.get("per_session", [])
    origin = origin_dir(task["output_dir"])

    count = 0
    td_sum_char = td_cnt_char = 0
    td_sum_tk = td_cnt_tk = 0
    for row in sessions:
        session = row.get("session")
        if not session or "/" in session or ".." in session:
            row["task_done"] = False
            continue
        logs_dir = origin / session / "logs"
        done = traj_stats.check_task_done_in_logs_dir(str(logs_dir), session)
        row["task_done"] = done
        if done:
            count += 1
            cl = row.get("char_len")
            if isinstance(cl, (int, float)) and cl > 0:
                td_sum_char += cl
                td_cnt_char += 1
            tk = row.get("total_tokens")
            if isinstance(tk, (int, float)) and tk > 0:
                td_sum_tk += tk
                td_cnt_tk += 1

    old = data.get("task_done_count")
    data["task_done_count"] = count
    if td_cnt_tk:
        data.setdefault("token_stats", {})["T_DONE"] = {
            "avg_total_tokens": round(td_sum_tk / td_cnt_tk),
            "sum_total": td_sum_tk, "count": td_cnt_tk,
        }
    if td_cnt_char:
        data.setdefault("char_len_stats", {})["T_DONE"] = {
            "avg_char_len": round(td_sum_char / td_cnt_char),
            "sum_total": td_sum_char, "count": td_cnt_char,
        }
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  task_done_count: {old} -> {count}  (共 {len(sessions)} session)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    want = set(sys.argv[1:])
    tasks = {t["id"]: t for t in load_tasks()}
    for tid in want:
        t = tasks.get(tid)
        if not t:
            print(f"[未找到任务] {tid}")
            continue
        print(f"[{tid}] {t.get('name', '?')}")
        download_logs(t)
        recompute(t)


if __name__ == "__main__":
    main()
