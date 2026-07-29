# -*- coding: utf-8 -*-
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
PIPELINE_SCRIPT = HERE.parent / "traj_pipeline" / "download_and_run.py"
WORKSPACE_PIPELINE_SCRIPT = HERE.parent / "traj_pipeline" / "download_workspace_and_run.py"
STATIC_DIR = HERE / "static"

# 复用轨迹统计里的 evaluator 首轮定位逻辑(首轮 = 编号最小的 turn, 见 #4)
sys.path.insert(0, str(HERE.parent))
import traj_stats  # noqa: E402
CONFIG_FILE = HERE / "config.json"
OBS_PROFILES_FILE = HERE / "obs_profiles.json"  # 额外 OBS 账号/桶凭证, gitignored, 不提交
TASKS_FILE = HERE / "tasks.json"
STATS_FILE_NAME = "filter_stats.json"

DEFAULT_CONFIG = {
    "output_base_dir": str(HERE / "pipeline_output"),
    "obsutil_path": "/home/w00802407/obsutil/obsutil",
    "port": 8080,
}

def _merge_tier_stats(stats_list, avg_key):
    """跨任务合并按 tier 聚合的统计（token_stats / char_len_stats 共用）。

    stats_list: 每个元素是一个 dict 的 token_stats 或 char_len_stats 字段（可能为 None），
                其内部各 tier 结构为 {sum_total, count, ...}。
    avg_key: 结果里平均值字段名（"avg_total_tokens" 或 "avg_char_len"）。
    返回合并后的 {L0: {<avg_key>: ...}, L1: {...}, ...}，若无数据返回 None。
    """
    tiers = ("L0", "L1", "T_DONE", "L1.5", "L2", "L3")
    merged = {}
    for tier in tiers:
        sum_t = 0
        cnt = 0
        for ts in stats_list:
            if not ts or tier not in ts:
                continue
            t = ts[tier]
            sum_t += t.get("sum_total", 0)
            cnt += t.get("count", 0)
        if cnt == 0:
            continue
        merged[tier] = {avg_key: round(sum_t / cnt)}
    return merged if merged else None


def _merge_token_stats(stats_list):
    return _merge_tier_stats(stats_list, "avg_total_tokens")


def _merge_char_len_stats(stats_list):
    return _merge_tier_stats(stats_list, "avg_char_len")


STAT_SUM_KEYS = ["filtered_count", "with_eval_count", "completion_ge_0.5", "completion_eq_1", "dropped_count",
                 "task_done_count"]

_job_lock = threading.Lock()
_job_state = {
    "task_id": None,
    "running": False,
    "started_at": None,
    "progress": "",
    "log_tail": [],
    "last_run_time": None,
    "last_duration_seconds": None,
    "last_error": None,
    "last_exit_code": None,
    "process": None,  # 当前运行的子进程对象
}
_LOG_TAIL_MAX = 40

_tasks_lock = threading.Lock()


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return DEFAULT_CONFIG.copy()


def load_obs_profiles() -> dict:
    """读取额外 OBS 账号/桶的凭证(AK/SK/endpoint), 按 profile 名字索引。

    存在 OBS_PROFILES_FILE(gitignored, 不提交)里, 不存在则返回空——所有 task 都走
    obsutil 全局默认凭证(现有行为, 零影响)。"""
    if OBS_PROFILES_FILE.exists():
        with open(OBS_PROFILES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _obs_cred_args_for_task(task: dict) -> list:
    """task 指定了非 default 的 obs_profile 时, 返回覆盖 obsutil 全局默认凭证的
    ["-i", ak, "-k", sk, "-e", endpoint]; 否则返回 [](走全局默认, 现有桶零影响)。"""
    profile_name = task.get("obs_profile") or "default"
    if profile_name == "default":
        return []
    profile = load_obs_profiles().get(profile_name)
    if not profile:
        return []
    return ["-i", profile["ak"], "-k", profile["sk"], "-e", profile["endpoint"]]


def _obs_cli_flags_for_task(task: dict) -> list:
    """同 _obs_cred_args_for_task, 但用于调用 traj_pipeline 子脚本(它们接的是
    --obs-ak/--obs-sk/--obs-endpoint, 自己再转成 obsutil 的 -i/-k/-e)。"""
    profile_name = task.get("obs_profile") or "default"
    if profile_name == "default":
        return []
    profile = load_obs_profiles().get(profile_name)
    if not profile:
        return []
    return ["--obs-ak", profile["ak"], "--obs-sk", profile["sk"], "--obs-endpoint", profile["endpoint"]]


# ── 任务注册表 ────────────────────────────────────────────────────────────────
def load_tasks() -> list:
    with _tasks_lock:
        if not TASKS_FILE.exists():
            return []
        with open(TASKS_FILE, encoding="utf-8") as f:
            return json.load(f).get("tasks", [])


def save_tasks(tasks: list):
    with _tasks_lock:
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump({"tasks": tasks}, f, ensure_ascii=False, indent=2)


def find_task(task_id: str) -> Optional[dict]:
    for t in load_tasks():
        if t["id"] == task_id:
            return t
    return None


def update_task(task_id: str, **fields):
    tasks = load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            t.update(fields)
            break
    save_tasks(tasks)


def delete_task_data(output_dir: str):
    """删除任务对应的数据文件。

    legacy 任务的 output_dir 就是 pipeline_output 根目录本身(其他新任务的数据都存在
    pipeline_output/tasks/<id>/ 下面), 如果对它整个 rmtree 会把 tasks/ 目录下所有其他
    任务的数据一并删掉, 所以这种情况只删该任务自己名下的几个产物文件/目录, 不动 tasks/ 子目录。
    """
    import shutil

    cfg = load_config()
    base_dir = Path(cfg["output_base_dir"]).resolve()
    out_dir = Path(output_dir).resolve()

    if out_dir == base_dir:
        for name in ("filter_stats.json", "filtered_sessions.txt", "pangu_filtered.jsonl",
                     "pangu_filtered_truncated.jsonl", "pangu_filtered_fold.jsonl", "origin"):
            target = out_dir / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
    elif out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)


def migrate_legacy_task_if_needed():
    """首次启动时, 若已存在旧版单数据源跑出来的 pipeline_output/filter_stats.json,
    把它包装成一个"历史数据"任务, 不移动/不重跑, 只是纳入任务列表。"""
    if TASKS_FILE.exists():
        return

    cfg = load_config()
    legacy_output_dir = Path(cfg["output_base_dir"])
    legacy_stats = legacy_output_dir / STATS_FILE_NAME

    tasks = []
    if legacy_stats.exists():
        # 旧 config.json(改造前)里可能还留着 assistant_obs/evaluator_obs, 尽力找回填充展示用
        assistant_obs = ""
        evaluator_obs = ""
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                old_cfg = json.load(f)
            assistant_obs = old_cfg.get("assistant_obs", "")
            evaluator_obs = old_cfg.get("evaluator_obs", "")
        except Exception:
            pass

        mtime = datetime.fromtimestamp(legacy_stats.stat().st_mtime).isoformat()
        tasks.append({
            "id": "legacy",
            "name": "历史数据（迁移导入）",
            "assistant_obs": assistant_obs,
            "evaluator_obs": evaluator_obs,
            "output_dir": str(legacy_output_dir),
            "created_at": mtime,
            "last_run_time": mtime,
            "last_duration_seconds": None,
            "last_exit_code": 0,
            "last_error": None,
        })

    save_tasks(tasks)


# ── 按任务定位数据文件 ──────────────────────────────────────────────────────────
def stats_path(output_dir: str) -> Path:
    return Path(output_dir) / STATS_FILE_NAME


def origin_dir(output_dir: str) -> Path:
    return Path(output_dir) / "origin"


_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[_T](\d{2})[-:](\d{2})[-:](\d{2})")


def _latest_json(session_dir: Path) -> Optional[Path]:
    """与 traj_pipeline/run_pipeline.py 的 latest_json 逻辑一致: 取文件名时间戳最新的 json。"""
    files = sorted(session_dir.glob("*.json"))
    if not files:
        return None

    def ts(p: Path):
        m = _TS_RE.search(p.name)
        return m.group(0) if m else p.name

    return sorted(files, key=ts)[-1]


def find_session_json(session: str, output_dir: str) -> Optional[Path]:
    """在 <output_dir>/origin/*/<session>/ 下查找该 session 的最新原始轨迹 json。
    assistant 目录名每次下载会变(取决于 obs 路径末段), 所以用通配符搜, 不写死目录名。"""
    if "/" in session or ".." in session:
        return None
    for candidate in origin_dir(output_dir).glob(f"*/{session}"):
        if candidate.is_dir():
            found = _latest_json(candidate)
            if found:
                return found
    return None


# 仅作极端情况的安全上限(防单条消息达数百 KB 撑爆响应); 正常内容不截断,
# 详情视图靠前端「默认折叠 + 点击展开看全文」控制可读性(见 #1/#3)。
_MAX_MSG_CHARS = 200000


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and "text" in b
        )
    return ""


def _simplify_message(msg: dict) -> dict:
    """把原始消息裁剪成前端渲染需要的最小字段, 并截断超长内容, 避免一次性把整份轨迹(可达 700KB+)全丢给前端。"""
    role = msg.get("role")
    text = _extract_text(msg.get("content"))
    truncated = False
    if len(text) > _MAX_MSG_CHARS:
        text = text[:_MAX_MSG_CHARS]
        truncated = True

    out = {"role": role, "content": text, "truncated": truncated}

    reasoning = msg.get("reasoning_content")
    if reasoning:
        if len(reasoning) > _MAX_MSG_CHARS:
            reasoning = reasoning[:_MAX_MSG_CHARS]
            out["reasoning_truncated"] = True
        out["reasoning_content"] = reasoning

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        out["tool_calls"] = [
            {
                "name": tc.get("function", {}).get("name"),
                "arguments": tc.get("function", {}).get("arguments"),
            }
            for tc in tool_calls
            if isinstance(tc, dict)
        ]

    if msg.get("tool_call_id"):
        out["tool_call_id"] = msg["tool_call_id"]

    return out


# ── 流水线执行(全局同一时刻只允许一个任务在跑) ──────────────────────────────────
def _append_log_line(line: str):
    """流水线子进程用 \\r 刷新下载进度行, 也用 \\n 输出常规日志。
    两者都要能实时体现在 job_state 里: 最新一行当作 progress, 历史行滚动进 log_tail。"""
    line = line.rstrip()
    if not line:
        return
    with _job_lock:
        _job_state["progress"] = line
        _job_state["log_tail"].append(line)
        if len(_job_state["log_tail"]) > _LOG_TAIL_MAX:
            _job_state["log_tail"] = _job_state["log_tail"][-_LOG_TAIL_MAX:]


def _stream_subprocess(cmd, env=None):
    """按字符读取子进程输出, 遇 \\r/\\n 断行(与 download_and_run.py 里 obsutil 的读取方式一致),
    这样下载进度这种用 \\r 原地刷新的行也能被实时捕获, 而不必等进程退出才能拿到全部输出。

    env: 若不为 None, 覆盖子进程环境变量(用于传 SSH_PASSWORD_* 而不出现在 cmd 里, 避免 ps 泄露)。"""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=env,
    )
    # 保存进程对象到全局状态，供终止接口使用
    with _job_lock:
        _job_state["process"] = proc

    buf = ""
    all_output = []
    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            break
        if ch in "\r\n":
            if buf.strip():
                _append_log_line(buf)
                all_output.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        _append_log_line(buf)
        all_output.append(buf)
    proc.wait()

    # 清除进程对象
    with _job_lock:
        _job_state["process"] = None

    return proc.returncode, "\n".join(all_output)


def run_pipeline(task_id: str, ssh_passwords: Optional[dict] = None):
    """ssh_passwords: 可选 {"assistant": "...", "evaluator": "..."}, 仅用于来源是 ssh:// 的这一次下载,
    只作为子进程环境变量传递, 调用结束后不再持有, 从不写入 _job_state / tasks.json / 任何文件。"""
    task = find_task(task_id)
    if not task:
        return

    with _job_lock:
        if _job_state["running"]:
            return
        _job_state["running"] = True
        _job_state["task_id"] = task_id
        _job_state["started_at"] = datetime.now().isoformat()
        _job_state["progress"] = "正在启动..."
        _job_state["log_tail"] = []
        _job_state["last_error"] = None

    t0 = time.time()
    cfg = load_config()
    out_dir = task["output_dir"]
    source_type = task.get("source_type", "session_analysis")
    try:
        os.makedirs(out_dir, exist_ok=True)
        if source_type == "workspace":
            # 单一 workspace 根路径, 按需下载各 task 的必要文件后统计(平台对齐口径)
            cmd = [
                sys.executable,
                "-u",
                str(WORKSPACE_PIPELINE_SCRIPT),
                task["workspace_obs"],
                out_dir,
                "--obsutil", cfg["obsutil_path"],
                "--concurrency", str(task.get("concurrency", 8)),
            ] + _obs_cli_flags_for_task(task)
            # Hermes harness 自动加 --hermes 开关，使用 Hermes 专用 include/exclude
            name_lower = (task.get("name") or "").lower()
            obs_lower = (task.get("workspace_obs") or "").lower()
            if "hermes" in name_lower or "hermes" in obs_lower:
                cmd.append("--hermes")
        else:
            cmd = [
                sys.executable,
                "-u",
                str(PIPELINE_SCRIPT),
                task["assistant_obs"],
                task["evaluator_obs"],
                out_dir,
                "--obsutil", cfg["obsutil_path"],
            ] + _obs_cli_flags_for_task(task)
        proc_env = dict(os.environ)
        # obsutil 默认读大写 HTTP_PROXY/HTTPS_PROXY, 但环境中它们可能指向被 Docker 网段
        # 冲突拦住的内网代理(proxycn-spl, 172.19.177.155), 导致 "no route to host"。
        # 用小写版的正确值覆盖(proxysg-spl, 172.29.14.129, 能通)。
        if "http_proxy" in proc_env:
            proc_env["HTTP_PROXY"] = proc_env["http_proxy"]
        if "https_proxy" in proc_env:
            proc_env["HTTPS_PROXY"] = proc_env["https_proxy"]
        if ssh_passwords:
            if ssh_passwords.get("assistant"):
                proc_env["SSH_PASSWORD_ASSISTANT"] = ssh_passwords["assistant"]
            if ssh_passwords.get("evaluator"):
                proc_env["SSH_PASSWORD_EVALUATOR"] = ssh_passwords["evaluator"]
        exit_code, full_output = _stream_subprocess(cmd, env=proc_env)
        last_run_time = datetime.now().isoformat()
        last_duration = round(time.time() - t0, 1)
        last_error = None
        if exit_code != 0:
            last_error = full_output[-3000:].strip()
        with _job_lock:
            _job_state["last_run_time"] = last_run_time
            _job_state["last_duration_seconds"] = last_duration
            _job_state["last_exit_code"] = exit_code
            _job_state["last_error"] = last_error
    except Exception as exc:
        last_run_time = datetime.now().isoformat()
        last_duration = round(time.time() - t0, 1)
        exit_code = None
        last_error = str(exc)
        with _job_lock:
            _job_state["last_run_time"] = last_run_time
            _job_state["last_duration_seconds"] = last_duration
            _job_state["last_error"] = last_error
    finally:
        with _job_lock:
            _job_state["running"] = False
            _job_state["started_at"] = None

    update_task(
        task_id,
        last_run_time=last_run_time,
        last_duration_seconds=last_duration,
        last_exit_code=exit_code,
        last_error=last_error,
    )


app = FastAPI(title="Trajectory Viewer")


@app.on_event("startup")
def _startup():
    migrate_legacy_task_if_needed()
    # 后台异步回填旧任务的 token_stats / task_done_count，避免阻塞 startup
    threading.Thread(target=_backfill_token_stats, daemon=True).start()
    threading.Thread(target=_backfill_task_done, daemon=True).start()


def _backfill_token_stats():
    """启动时回填旧任务的 token_stats/char_len_stats（新任务在 pipeline 阶段已写入）。

    遍历所有已有 filter_stats.json 但缺 token_stats 或 char_len_stats 的任务，对 origin
    目录重新调用 traj_stats.process_root() 提取 token/char_len 数据，计算各 tier 平均值后写回。
    """
    tasks = load_tasks()
    updated = 0
    for t in tasks:
        sp = stats_path(t["output_dir"])
        if not sp.exists():
            continue
        try:
            with open(sp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("token_stats") and data.get("char_len_stats"):
            continue                # 均已有, 跳过

        origin = origin_dir(t["output_dir"])
        if not origin.is_dir():
            continue

        # 跑 traj_stats 提取 token/char_len
        per_task = traj_stats.process_root(str(origin))
        if not per_task:
            continue

        # 同 build_platform_stats 的 tier 聚合逻辑
        l0_idx, l1_idx, l15_idx, l2_idx, l3_idx, td_idx = [], [], [], [], [], []
        for i, row in enumerate(per_task):
            comp = row.get("evaluator_completion")
            has_score = isinstance(comp, (int, float))
            passed = bool(row.get("passed_gate"))
            if bool(row.get("task_done")):
                td_idx.append(i)
            if passed:
                l1_idx.append(i)
                if has_score:
                    l15_idx.append(i)
                if has_score and isinstance(comp, (int, float)) and comp >= 0.5:
                    l2_idx.append(i)
                if has_score and comp == 1:
                    l3_idx.append(i)
            l0_idx.append(i)

        def _avg(entries, indices, field, avg_key):
            sum_total = 0
            cnt = 0
            for idx in indices:
                row = entries[idx]
                total = row.get(field)
                if total is None:
                    continue
                sum_total += total
                cnt += 1
            if cnt == 0:
                return None
            return {"sum_total": sum_total, "count": cnt,
                    avg_key: round(sum_total / cnt)}

        token_stats = {}
        char_len_stats = {}
        for tier, indices in [("L0", l0_idx), ("L1", l1_idx), ("T_DONE", td_idx),
                              ("L1.5", l15_idx), ("L2", l2_idx), ("L3", l3_idx)]:
            avg = _avg(per_task, indices, "total_tokens", "avg_total_tokens")
            if avg is not None:
                token_stats[tier] = avg
            char_avg = _avg(per_task, indices, "char_len", "avg_char_len")
            if char_avg is not None:
                char_len_stats[tier] = char_avg

        if not token_stats and not char_len_stats:
            continue

        # 写入前重新读取，避免与 _backfill_task_done 并行写入时覆盖对方数据
        try:
            with open(sp, encoding="utf-8") as f:
                latest = json.load(f)
        except (json.JSONDecodeError, OSError):
            latest = data
        if token_stats:
            latest["token_stats"] = token_stats
        if char_len_stats:
            latest["char_len_stats"] = char_len_stats
        try:
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(latest, f, ensure_ascii=False, indent=2)
            updated += 1
            print(f"[backfill] token_stats/char_len_stats -> {t.get('name', '?')}")
        except OSError:
            pass

    if updated:
        print(f"[backfill] 共回填 {updated} 个任务的 token_stats/char_len_stats")


def _backfill_task_done():
    """启动时回填旧任务(filter_stats.json 缺 task_done_count)的【Task_Done】统计。

    workspace 来源的任务里 per_session[].session 就是 origin 下的 task 目录名, 可直接定位
    <session>/logs/<session>.log 检查「【Task_Done】」标记；session_analysis 来源没有这个目录
    结构, 对应 log 路径必然不存在, has_task_done_marker 会自然返回 False(计数恒为 0)。
    """
    tasks = load_tasks()
    updated = 0
    for t in tasks:
        sp = stats_path(t["output_dir"])
        if not sp.exists():
            continue
        try:
            with open(sp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "task_done_count" in data:
            continue                # 已回填过, 跳过

        sessions = data.get("per_session", [])
        if not sessions:
            continue

        origin = origin_dir(t["output_dir"])
        count = 0
        td_sum_char = 0
        td_cnt_char = 0
        td_sum_tk = 0
        td_cnt_tk = 0
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

        # 写入前重新读取，避免与 _backfill_token_stats 并行写入时覆盖对方数据
        try:
            with open(sp, encoding="utf-8") as f:
                latest = json.load(f)
        except (json.JSONDecodeError, OSError):
            latest = data
        latest["task_done_count"] = count
        # 也回填 per_session 里 task_done
        for ls, row in zip(latest.get("per_session", []), sessions):
            ls["task_done"] = row.get("task_done", False)
        if td_cnt_tk:
            ts_td = latest.setdefault("token_stats", {})
            ts_td["T_DONE"] = {
                "avg_total_tokens": round(td_sum_tk / td_cnt_tk),
                "sum_total": td_sum_tk, "count": td_cnt_tk,
            }
        if td_cnt_char:
            cs_td = latest.setdefault("char_len_stats", {})
            cs_td["T_DONE"] = {
                "avg_char_len": round(td_sum_char / td_cnt_char),
                "sum_total": td_sum_char, "count": td_cnt_char,
            }
        data = latest
        try:
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            updated += 1
            print(f"[backfill] task_done_count={count} -> {t.get('name', '?')}")
        except OSError:
            pass

    if updated:
        print(f"[backfill] 共回填 {updated} 个任务的 task_done_count")


def _task_summary(task: dict) -> dict:
    p = stats_path(task["output_dir"])
    summary = dict(task)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        summary["available"] = True
        summary["session_total"] = len(data.get("per_session", []))
        for k in STAT_SUM_KEYS:
            summary[k] = data.get(k, 0)
        if data.get("token_stats"):
            summary["token_stats"] = data["token_stats"]
        if data.get("char_len_stats"):
            summary["char_len_stats"] = data["char_len_stats"]
    else:
        summary["available"] = False
        summary["session_total"] = 0
        for k in STAT_SUM_KEYS:
            summary[k] = 0
    return summary


@app.get("/api/tasks")
def api_list_tasks():
    return {"tasks": [_task_summary(t) for t in load_tasks()]}


@app.get("/api/obs-profiles")
def api_list_obs_profiles():
    """给前端建任务表单用的下拉选项, 只给 profile 名字, 绝不把 ak/sk 传到前端。"""
    return {"profiles": ["default"] + sorted(load_obs_profiles().keys())}


@app.get("/api/groups")
def api_list_groups():
    """列出所有分组名称（从所有任务的 groups 字段去重汇总）。"""
    tasks = load_tasks()
    all_groups = set()
    for t in tasks:
        for g in (t.get("groups") or []):
            if isinstance(g, str) and g.strip():
                all_groups.add(g.strip())
    return {"groups": sorted(all_groups)}


@app.get("/api/stats/by-group")
def api_stats_by_group(group: Optional[str] = None):
    """按分组统计轨迹数（漏斗汇总）。

    group=None 或空字符串时，空字符串表示"未分组"；group=<name> 时只统计该组内任务。
    """
    tasks = load_tasks()
    if group is not None:
        if group == '':
            # 空字符串 = 未分组
            tasks = [t for t in tasks if not (t.get("groups") and len(t.get("groups")))]
        else:
            # 指定分组
            tasks = [t for t in tasks if group in (t.get("groups") or [])]

    total = filtered = with_eval = ge05 = eq1 = dropped = session_total = task_done = 0
    token_stats_list = []
    char_len_stats_list = []
    for t in tasks:
        sp = stats_path(t["output_dir"])
        if not sp.exists():
            continue
        with open(sp, encoding="utf-8") as f:
            data = json.load(f)
        filtered += data.get("filtered_count", 0)
        with_eval += data.get("with_eval_count", 0)
        ge05 += data.get("completion_ge_0.5", 0)
        eq1 += data.get("completion_eq_1", 0)
        dropped += data.get("dropped_count", 0)
        task_done += data.get("task_done_count", 0)
        session_total += len(data.get("per_session", []))
        if data.get("token_stats"):
            token_stats_list.append(data["token_stats"])
        if data.get("char_len_stats"):
            char_len_stats_list.append(data["char_len_stats"])

    total = filtered + dropped
    result = {
        "group": group,
        "task_count": len(tasks),
        "filtered_count": filtered,
        "dropped_count": dropped,
        "with_eval_count": with_eval,
        "completion_ge_0.5": ge05,
        "completion_eq_1": eq1,
        "task_done_count": task_done,
        "available": total > 0,
    }
    merged = _merge_token_stats(token_stats_list)
    if merged:
        result["token_stats"] = merged
    char_merged = _merge_char_len_stats(char_len_stats_list)
    if char_merged:
        result["char_len_stats"] = char_merged
    return result


@app.put("/api/tasks/{task_id}/groups")
def api_update_task_groups(task_id: str, body: dict):
    """更新任务的分组（覆盖式）。

    body: {"groups": ["组1", "组2", ...]}
    """
    groups = body.get("groups") or []
    if not isinstance(groups, list):
        return JSONResponse({"success": False, "message": "groups 必须是数组"}, status_code=400)
    # 去重、去空、trim
    groups = sorted(set(g.strip() for g in groups if isinstance(g, str) and g.strip()))

    update_task(task_id, groups=groups)
    return {"success": True, "groups": groups}


def _extract_ssh_passwords(body: dict) -> dict:
    """从请求体里取出一次性使用的 SSH 密码, 只在这次下载调用里传给子进程环境变量,
    不落进 task/tasks.json(那里只存不含密码的 ssh://user@host/path)。"""
    return {
        "assistant": (body.get("assistant_ssh_password") or "").strip(),
        "evaluator": (body.get("evaluator_ssh_password") or "").strip(),
    }


def _missing_ssh_passwords(task: dict, ssh_passwords: dict) -> list:
    """重新采集时, 若来源是 ssh://, 密码从未持久化, 必须每次重新提交, 这里检查哪些侧缺密码。"""
    missing = []
    if task["assistant_obs"].startswith("ssh://") and not ssh_passwords.get("assistant"):
        missing.append("assistant")
    if task["evaluator_obs"].startswith("ssh://") and not ssh_passwords.get("evaluator"):
        missing.append("evaluator")
    return missing


@app.post("/api/tasks")
def api_create_task(body: dict):
    name = (body.get("name") or "").strip()
    source_type = (body.get("source_type") or "session_analysis").strip()
    workspace_obs = (body.get("workspace_obs") or "").strip()
    assistant_obs = (body.get("assistant_obs") or "").strip()
    evaluator_obs = (body.get("evaluator_obs") or "").strip()
    obs_profile = (body.get("obs_profile") or "default").strip() or "default"

    # workspace 下载并发数: 缺省 8, 限制在 [1, 64] 防止误填导致 OBS 限流/进程过多
    try:
        concurrency = int(body.get("concurrency") or 8)
    except (TypeError, ValueError):
        concurrency = 8
    concurrency = max(1, min(64, concurrency))

    if source_type == "workspace":
        if not name or not workspace_obs:
            return JSONResponse(
                {"success": False, "message": "任务名称、workspace 来源不能为空"},
                status_code=400,
            )
        ssh_passwords = {}  # workspace 来源仅支持 obs://, 不涉及 SSH 密码
    else:
        if not name or not assistant_obs or not evaluator_obs:
            return JSONResponse(
                {"success": False, "message": "任务名称、assistant 来源、eval 来源均不能为空"},
                status_code=400,
            )
        ssh_passwords = _extract_ssh_passwords(body)
        missing = _missing_ssh_passwords(
            {"assistant_obs": assistant_obs, "evaluator_obs": evaluator_obs}, ssh_passwords
        )
        if missing:
            return JSONResponse(
                {"success": False, "message": f"服务器路径来源({'/'.join(missing)})缺少密码"},
                status_code=400,
            )

    cfg = load_config()
    task_id = "t_" + uuid.uuid4().hex[:8]
    output_dir = str(Path(cfg["output_base_dir"]) / "tasks" / task_id)
    task = {
        "id": task_id,
        "name": name,
        "source_type": source_type,
        "workspace_obs": workspace_obs,
        "obs_profile": obs_profile,
        "concurrency": concurrency,
        "assistant_obs": assistant_obs,
        "evaluator_obs": evaluator_obs,
        "output_dir": output_dir,
        "created_at": datetime.now().isoformat(),
        "last_run_time": None,
        "last_duration_seconds": None,
        "last_exit_code": None,
        "last_error": None,
    }
    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)

    with _job_lock:
        already_running = _job_state["running"]
    if already_running:
        return {
            "success": True,
            "task": task,
            "started": False,
            "message": "任务已创建，但当前有其他任务正在采集中，请稍后在任务列表手动点击「重新采集」",
        }

    threading.Thread(target=run_pipeline, args=(task_id, ssh_passwords), daemon=True).start()
    return {"success": True, "task": task, "started": True, "message": "任务已创建，正在采集数据"}


@app.post("/api/tasks/{task_id}/trigger")
def api_trigger_task(task_id: str, body: dict = None):
    task = find_task(task_id)
    if not task:
        return JSONResponse({"success": False, "message": "任务不存在"}, status_code=404)

    ssh_passwords = _extract_ssh_passwords(body or {})
    missing = _missing_ssh_passwords(task, ssh_passwords)
    if missing:
        return JSONResponse(
            {"success": False, "message": f"服务器路径来源({'/'.join(missing)})缺少密码，请重新输入",
             "need_password": missing},
            status_code=400,
        )

    with _job_lock:
        if _job_state["running"]:
            return {"success": False, "message": "已有任务正在采集中，请稍候"}
    threading.Thread(target=run_pipeline, args=(task_id, ssh_passwords), daemon=True).start()
    return {"success": True, "message": "已开始采集"}


@app.patch("/api/tasks/{task_id}")
def api_rename_task(task_id: str, body: dict):
    task = find_task(task_id)
    if not task:
        return JSONResponse({"success": False, "message": "任务不存在"}, status_code=404)
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"success": False, "message": "任务名称不能为空"}, status_code=400)
    update_task(task_id, name=name)
    return {"success": True, "task": find_task(task_id)}


@app.delete("/api/tasks/{task_id}")
def api_delete_task(task_id: str):
    task = find_task(task_id)
    if not task:
        return JSONResponse({"success": False, "message": "任务不存在"}, status_code=404)

    with _job_lock:
        if _job_state["running"] and _job_state["task_id"] == task_id:
            return JSONResponse(
                {"success": False, "message": "该任务正在采集中，无法删除，请等待采集结束"},
                status_code=409,
            )

    delete_task_data(task["output_dir"])
    tasks = [t for t in load_tasks() if t["id"] != task_id]
    save_tasks(tasks)
    return {"success": True, "message": "任务已删除"}


@app.get("/api/stats")
def api_stats():
    """跨所有已登记任务的汇总统计。"""
    tasks = load_tasks()
    summary = {k: 0 for k in STAT_SUM_KEYS}
    session_total = 0
    available = False
    token_stats_list = []
    char_len_stats_list = []
    for t in tasks:
        p = stats_path(t["output_dir"])
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        available = True
        session_total += len(data.get("per_session", []))
        for k in STAT_SUM_KEYS:
            summary[k] += data.get(k, 0)
        if data.get("token_stats"):
            token_stats_list.append(data["token_stats"])
        if data.get("char_len_stats"):
            char_len_stats_list.append(data["char_len_stats"])

    summary["available"] = available
    summary["session_total"] = session_total
    summary["task_count"] = len(tasks)
    merged = _merge_token_stats(token_stats_list)
    if merged:
        summary["token_stats"] = merged
    char_merged = _merge_char_len_stats(char_len_stats_list)
    if char_merged:
        summary["char_len_stats"] = char_merged
    return summary


@app.get("/api/tasks/{task_id}/sessions")
def api_task_sessions(
    task_id: str,
    page: int = 1,
    page_size: int = 20,
    has_eval: Optional[bool] = None,
    completion_filter: Optional[str] = None,  # "ge05" | "eq1" | "no_eval"
):
    task = find_task(task_id)
    if not task:
        return JSONResponse({"success": False, "message": "任务不存在"}, status_code=404)

    p = stats_path(task["output_dir"])
    if not p.exists():
        return {"sessions": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 0}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    sessions = data.get("per_session", [])

    if has_eval is not None:
        sessions = [s for s in sessions if s.get("has_eval") is has_eval]
    if completion_filter == "ge05":
        sessions = [s for s in sessions if isinstance(s.get("completion"), (int, float)) and s["completion"] >= 0.5]
    elif completion_filter == "eq1":
        sessions = [s for s in sessions if s.get("completion") == 1]
    elif completion_filter == "no_eval":
        sessions = [s for s in sessions if not s.get("has_eval")]

    total = len(sessions)
    start = (page - 1) * page_size
    return {
        "sessions": sessions[start: start + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


_VERDICT_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_verdict(data: dict) -> Optional[dict]:
    """evaluator 的最终裁决(completion/reason/rubric_checks 逐条核验)存在原始 json 顶层的
    response.content 里, 是一段独立于 messages[] 的(可能被```json```包裹的) JSON 文本。
    之前只裁剪 messages, 这段裁决内容完全没有传给前端, 所以需要单独解析出来。"""
    resp = data.get("response")
    if not isinstance(resp, dict):
        return None
    text = _extract_text(resp.get("content"))
    if "rubric_checks" not in text:
        return None
    m = _VERDICT_FENCE_RE.search(text)
    raw = m.group(1) if m else text.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, str):
            obj = json.loads(obj)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    gate_status = obj.get("gate_status") if isinstance(obj.get("gate_status"), dict) else {}
    rubric_checks = [
        {
            "kind": "gate" if rc.get("rubric_id") in gate_status else "reward",
            "criterion": rc.get("criterion"),
            "passed": rc.get("passed"),
            "evidence": rc.get("evidence"),
        }
        for rc in (obj.get("rubric_checks") or []) if isinstance(rc, dict)
    ]
    rubric_checks = _sort_rubric_gate_first(rubric_checks)
    return {
        "completion": obj.get("completion"),
        "reason": obj.get("reason"),
        "inclination": obj.get("inclination"),
        "rubric_checks": rubric_checks,
    }


def _sort_rubric_gate_first(rubric_checks: list) -> list:
    """把 gate 项排在 reward 项前面(稳定排序, 组内保持原顺序)。见 #2。"""
    return sorted(rubric_checks, key=lambda rc: 0 if rc.get("kind") == "gate" else 1)


def _load_simplified_trajectory(session: str, output_dir: str) -> Optional[dict]:
    """按 session(目录名) 定位原始轨迹 json 并裁剪为前端渲染用的结构。
    assistant/evaluator 两侧轨迹目录结构一致, 都是 <output_dir>/origin/*/<session>/*.json,
    所以同一套查找+裁剪逻辑对两者都适用。"""
    json_path = find_session_json(session, output_dir)
    if not json_path:
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages", [])
    return {
        "session": session,
        "source_file": str(json_path),
        "model": data.get("model"),
        "message_count": len(messages),
        "messages": [_simplify_message(m) for m in messages],
        "verdict": _extract_verdict(data),
    }


# ── workspace 来源的轨迹查看(assistant .jsonl + log 裁决 + 按需拉 evaluator) ──────

def _simplify_workspace_message(role: str, parts: list) -> Optional[dict]:
    """把 workspace assistant/evaluator .jsonl 的一条 message 的 content 部件列表,
    映射成前端已认的结构 {role, content, reasoning_content, tool_calls, truncated}。
    部件类型: thinking / text / toolCall / (toolResult 侧的) text。"""
    texts, reasonings, tool_calls = [], [], []
    for p in parts:
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t == "text":
            if p.get("text"):
                texts.append(p["text"])
        elif t == "thinking":
            if p.get("thinking"):
                reasonings.append(p["thinking"])
        elif t == "toolCall":
            tool_calls.append({
                "name": p.get("name"),
                "arguments": p.get("arguments"),
            })

    content = "\n".join(texts)
    reasoning = "\n".join(reasonings)
    truncated = False
    if len(content) > _MAX_MSG_CHARS:
        content = content[:_MAX_MSG_CHARS]
        truncated = True

    out = {"role": role, "content": content, "truncated": truncated}
    if reasoning:
        if len(reasoning) > _MAX_MSG_CHARS:
            reasoning = reasoning[:_MAX_MSG_CHARS]
            out["reasoning_truncated"] = True
        out["reasoning_content"] = reasoning
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _load_workspace_jsonl_trajectory(jsonl_path: Path, session: str) -> Optional[dict]:
    """解析 workspace 的原始 .jsonl(session/message 混合事件流),裁剪为前端渲染结构。
    只取 type=='message' 的行; content 可能是字符串(user)或部件列表(assistant/toolResult)。"""
    if not jsonl_path or not jsonl_path.exists():
        return None
    messages = []
    model = None
    # 统计 token 使用量 (仅统计 assistant 的 usage)
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0

    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            otype = obj.get("type")
            if otype == "model_change" and obj.get("model"):
                model = obj["model"]
                continue
            if otype != "message":
                continue
            msg = obj.get("message") or {}
            role = msg.get("role")
            content = msg.get("content")

            # 提取 usage 信息 (仅 assistant 消息有 usage)
            if role == "assistant":
                usage = msg.get("usage")
                if usage and isinstance(usage, dict):
                    total_input_tokens += usage.get("input", 0)
                    total_output_tokens += usage.get("output", 0)
                    total_reasoning_tokens += usage.get("reasoningTokens", 0)

            if isinstance(content, str):
                text = content
                truncated = False
                if len(text) > _MAX_MSG_CHARS:
                    text = text[:_MAX_MSG_CHARS]
                    truncated = True
                messages.append({"role": role, "content": text, "truncated": truncated})
            elif isinstance(content, list):
                simplified = _simplify_workspace_message(role, content)
                if simplified:
                    messages.append(simplified)

    result = {
        "session": session,
        "source_file": str(jsonl_path),
        "model": model,
        "message_count": len(messages),
        "messages": messages,
    }

    # 添加 token 统计信息
    if total_input_tokens > 0 or total_output_tokens > 0:
        result["token_usage"] = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "reasoning_tokens": total_reasoning_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

    return result


def _workspace_task_dir(session: str, output_dir: str) -> Optional[Path]:
    """workspace 的 per_session[].session 就是 task 目录名, 定位 <output_dir>/origin/<session>/。"""
    if "/" in session or ".." in session:
        return None
    d = origin_dir(output_dir) / session
    return d if d.is_dir() else None


def _load_workspace_assistant(session: str, output_dir: str) -> Optional[dict]:
    """加载 workspace assistant 轨迹: <task>/agents/{assistant*,main}/sessions/<非trajectory>.jsonl。
    assistant 侧目录名可能是 assistant1 或 main(不同 harness 变体)。"""
    task_dir = _workspace_task_dir(session, output_dir)
    if not task_dir:
        return None
    for pattern in ("agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"):
        for jsonl in sorted(task_dir.glob(pattern)):
            if "trajectory" in jsonl.name:
                continue
            traj = _load_workspace_jsonl_trajectory(jsonl, session)
            if traj:
                return traj
    return None


def _extract_verdict_from_log(session: str, output_dir: str) -> Optional[dict]:
    """从 <task>/logs/<task>.log 的「首轮」evaluator 块解析裁决, 归一化成与 _extract_verdict 同款结构。
    首轮 = 编号最小的 turn(见 #4), 复用 traj_stats.extract_first_evaluator_obj。"""
    task_dir = _workspace_task_dir(session, output_dir)
    if not task_dir:
        return None
    log_path = task_dir / "logs" / (session + ".log")
    if not log_path.exists():
        return None
    obj = traj_stats.extract_first_evaluator_obj(str(log_path))
    if not isinstance(obj, dict):   # None(无裁决) 或 '__BADJSON__'(解析失败)
        return None
    gate_status = obj.get("gate_status") if isinstance(obj.get("gate_status"), dict) else {}
    rubric_checks = [
        {
            "kind": "gate" if rc.get("rubric_id") in gate_status else "reward",
            "criterion": rc.get("criterion"),
            "passed": rc.get("passed"),
            "evidence": rc.get("evidence"),
        }
        for rc in (obj.get("rubric_checks") or []) if isinstance(rc, dict)
    ]
    rubric_checks = _sort_rubric_gate_first(rubric_checks)
    return {
        "completion": obj.get("completion"),
        "reason": obj.get("reason"),
        "inclination": obj.get("inclination"),
        "rubric_checks": rubric_checks,
    }


_INCL_STR_RE = re.compile(r'"inclination"\s*:\s*"([^"]+)"')
_REASON_STR_RE = re.compile(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"')
_COMP_NUM_RE = re.compile(r'"completion"\s*:\s*(null|-?[0-9.]+)')


def _eval_traj_verdict_text(jsonl_path: Path) -> Optional[str]:
    """evaluator 轨迹里最后一条含 rubric_checks + inclination 的 assistant 文本(裁决原文)。"""
    if not jsonl_path or not jsonl_path.exists():
        return None
    last = None
    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
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
            c = msg.get("content")
            if isinstance(c, str):
                txt = c
            elif isinstance(c, list):
                txt = "\n".join(p.get("text", "") for p in c
                                if isinstance(p, dict) and p.get("type") == "text")
            else:
                txt = ""
            if "rubric_checks" in txt and "inclination" in txt:
                last = txt
    return last


def _verdict_from_eval_traj_text(txt: str) -> Optional[dict]:
    """把 evaluator 轨迹裁决原文构造成 verdict dict。
    先尝试整块 json.loads(可拿到逐条 rubric); 失败则用正则退化取 completion/inclination/reason
    (裁决 JSON 常含非法转义, 整块解析不可靠, 此时 rubric 列表留空, 用户仍可在 evaluator 轨迹里看原文)。"""
    if not txt:
        return None
    m = _VERDICT_FENCE_RE.search(txt)
    raw = m.group(1) if m else txt
    obj = None
    try:
        o = json.loads(raw)
        if isinstance(o, dict):
            obj = o
    except (json.JSONDecodeError, TypeError):
        obj = None
    if obj is not None:
        gate_status = obj.get("gate_status") if isinstance(obj.get("gate_status"), dict) else {}
        rubric_checks = _sort_rubric_gate_first([
            {
                "kind": "gate" if rc.get("rubric_id") in gate_status else "reward",
                "criterion": rc.get("criterion"),
                "passed": rc.get("passed"),
                "evidence": rc.get("evidence"),
            }
            for rc in (obj.get("rubric_checks") or []) if isinstance(rc, dict)
        ])
        return {
            "completion": obj.get("completion"),
            "reason": obj.get("reason"),
            "inclination": obj.get("inclination"),
            "rubric_checks": rubric_checks,
            "verdict_source": "eval_traj",
        }
    # 退化: 正则取标量
    cm = _COMP_NUM_RE.findall(txt)
    completion = None
    if cm and cm[-1] != "null":
        try:
            completion = float(cm[-1])
        except ValueError:
            completion = None
    im = _INCL_STR_RE.search(txt)
    rm = _REASON_STR_RE.search(txt)
    return {
        "completion": completion,
        "reason": rm.group(1) if rm else None,
        "inclination": im.group(1) if im else None,
        "rubric_checks": [],
        "verdict_source": "eval_traj_partial",
    }


def _extract_workspace_verdict(session: str, output_dir: str) -> Optional[dict]:
    """workspace 首轮裁决: 先读 log; log 无裁决时回退本地 evaluator 轨迹(见方案 B)。"""
    verdict = _extract_verdict_from_log(session, output_dir)
    if verdict is not None:
        return verdict
    task_dir = _workspace_task_dir(session, output_dir)
    if not task_dir:
        return None
    ev = traj_stats.find_evaluator_trajectory(str(task_dir))
    if not ev:
        return None
    return _verdict_from_eval_traj_text(_eval_traj_verdict_text(Path(ev)))


def _ensure_workspace_evaluator(task: dict, session: str) -> Optional[dict]:
    """加载 workspace evaluator 轨迹; 本地无则用 workspace_obs 按需临时下载该 task 的 evaluator jsonl。
    下载失败/无地址时返回 None(前端据此隐藏 evaluator 页), 不影响 assistant 与裁决展示。"""
    output_dir = task["output_dir"]
    task_dir = _workspace_task_dir(session, output_dir)
    if not task_dir:
        return None

    def _find_local():
        for jsonl in sorted(task_dir.glob("agents/evaluator/sessions/*.jsonl")):
            if "trajectory" in jsonl.name:
                continue
            return jsonl
        return None

    jsonl = _find_local()
    if not jsonl:
        # 按需下载: <workspace_obs>/<session>/ 只拉 evaluator 非 trajectory jsonl
        workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
        if not workspace_obs:
            return None
        cfg = load_config()
        task_obs = f"{workspace_obs}/{session}/"
        origin = str(origin_dir(output_dir))
        cmd = [
            cfg["obsutil_path"], "cp", task_obs, origin, "-r", "-f",
            "-include", "*evaluator*sessions*.jsonl",
            "-exclude", "*trajectory*",
        ] + _obs_cred_args_for_task(task)
        try:
            subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        except Exception:
            return None
        jsonl = _find_local()
    if not jsonl:
        return None
    return _load_workspace_jsonl_trajectory(jsonl, session)


# ── Hermes 来源(query1.json): 轨迹+评测同在一个文件, 无 agents/*/sessions ────────

def _load_hermes_profile_session(json_path: Path, session: str) -> Optional[dict]:
    """解析 Hermes 的 profiles/*/sessions/*.json(标准 OpenAI 格式: {messages:[{role,content,tool_calls}]}),
    裁剪为前端渲染结构。与 session_analysis 侧同格式, 故复用 _simplify_message。"""
    if not json_path or not json_path.exists():
        return None
    try:
        with open(json_path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    messages = data.get("messages", [])
    return {
        "session": session,
        "source_file": str(json_path),
        "model": data.get("model"),
        "message_count": len(messages),
        "messages": [_simplify_message(m) for m in messages],
    }


def _ensure_hermes_evaluator(task: dict, session: str) -> Optional[dict]:
    """加载 Hermes evaluator 完整轨迹: <task>/profiles/evaluator/sessions/*.json。
    本地无则用 workspace_obs 按需临时下载该 session 的 evaluator profile session(懒加载,
    默认批量下载已不含 evaluator 以省空间, 见 download_workspace_and_run.py 的过滤规则)。
    下载失败/无地址时返回 None(前端据此隐藏 evaluator 页), 不影响 assistant 与裁决展示。"""
    output_dir = task["output_dir"]
    task_dir = _workspace_task_dir(session, output_dir)
    if not task_dir:
        return None

    def _find_local():
        cands = sorted(task_dir.glob("profiles/evaluator/sessions/*.json"),
                       key=lambda p: p.stat().st_size, reverse=True)
        return cands[0] if cands else None

    json_path = _find_local()
    if not json_path:
        # 按需下载: <workspace_obs>/<session>/ 只拉 evaluator profile session json
        workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
        if not workspace_obs:
            return None
        cfg = load_config()
        task_obs = f"{workspace_obs}/{session}/"
        origin = str(origin_dir(output_dir))
        cmd = [
            cfg["obsutil_path"], "cp", task_obs, origin, "-r", "-f",
            "-include", "*profiles/evaluator/sessions/*.json",
        ] + _obs_cred_args_for_task(task)
        try:
            subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        except Exception:
            return None
        json_path = _find_local()
    if not json_path:
        return None
    return _load_hermes_profile_session(json_path, session)


def _find_query1_json(session: str, output_dir: str) -> Optional[Path]:
    """定位 Hermes 的 query1.json: <output_dir>/origin/<session>/logs/trajectories/*/query*.json。
    多时间戳目录时取最大者(与统计侧一致)。"""
    task_dir = _workspace_task_dir(session, output_dir)
    if not task_dir:
        return None
    cands = sorted(task_dir.glob("logs/trajectories/*/query*.json"),
                   key=lambda p: p.stat().st_size, reverse=True)
    return cands[0] if cands else None


def _render_query1_trajectory(query1_path: Path, session: str) -> Optional[dict]:
    """把 query1.json 的 turns[] 渲染成前端已认的 {role, content, ...} 消息流:
       每个 turn -> user(user_input) + assistant(agent_content + tool_calls)。"""
    try:
        with open(query1_path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    messages = []
    for t in (data.get("turns") or []):
        if not isinstance(t, dict):
            continue
        ui = (t.get("user_input") or "").strip()
        if ui:
            txt, trunc = ui, False
            if len(txt) > _MAX_MSG_CHARS:
                txt, trunc = txt[:_MAX_MSG_CHARS], True
            messages.append({"role": "user", "content": txt, "truncated": trunc})

        ac = t.get("agent_content") or ""
        tool_calls = []
        for tc in (t.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            tool_calls.append({
                "name": tc.get("tool"),
                # 复用 openclaw 的 arguments 字段, 前端已能渲染
                "arguments": tc.get("input"),
                "output": tc.get("output"),
            })
        txt, trunc = ac, False
        if len(txt) > _MAX_MSG_CHARS:
            txt, trunc = txt[:_MAX_MSG_CHARS], True
        msg = {"role": "assistant", "content": txt, "truncated": trunc}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if msg["content"] or tool_calls:
            messages.append(msg)

    return {
        "session": session,
        "source_file": str(query1_path),
        "model": data.get("agent_name"),
        "message_count": len(messages),
        "messages": messages,
    }


def _query1_verdict(query1_path: Path) -> Optional[dict]:
    """从 query1.json.evaluations[] 首轮(turn 号最小)构造 verdict, 归一化成 rubric 面板结构。"""
    try:
        with open(query1_path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    evals = [e for e in (data.get("evaluations") or []) if isinstance(e, dict)]
    if not evals:
        return None
    obj = min(evals, key=lambda e: e.get("turn", float("inf")))
    gate_status = obj.get("gate_status") if isinstance(obj.get("gate_status"), dict) else {}
    rubric_checks = _sort_rubric_gate_first([
        {
            "kind": "gate" if rc.get("rubric_id") in gate_status else "reward",
            "criterion": rc.get("criterion"),
            "passed": rc.get("passed"),
            "evidence": rc.get("evidence"),
        }
        for rc in (obj.get("rubric_checks") or []) if isinstance(rc, dict)
    ])
    return {
        "completion": obj.get("completion"),
        "reason": obj.get("reason"),
        "inclination": obj.get("inclination"),
        "rubric_checks": rubric_checks,
        "verdict_source": "query1",
    }


def _read_hermes_state_db(task_dir: Path) -> Optional[dict]:
    """从 Hermes 的 profiles/assistant1/state.db 读取 token 使用量。
    task_dir 是 <output_dir>/origin/<task_name>/。
    返回 {input_tokens, output_tokens, reasoning_tokens, total_tokens}。"""
    state_db = task_dir / "profiles" / "assistant1" / "state.db"
    if not state_db.exists():
        return None
    try:
        con = sqlite3.connect(str(state_db))
        con.row_factory = sqlite3.Row
        # 每个 Hermes task 只有一个 session, 取第一条即可
        row = con.execute(
            "SELECT input_tokens, output_tokens, reasoning_tokens "
            "FROM sessions LIMIT 1"
        ).fetchone()
        con.close()
        if not row:
            return None
        inp = row["input_tokens"] or 0
        out = row["output_tokens"] or 0
        reas = row["reasoning_tokens"] or 0
        return {
            "input_tokens": inp,
            "output_tokens": out,
            "reasoning_tokens": reas,
            "total_tokens": inp + out + reas,
        }
    except Exception:
        return None


@app.get("/api/tasks/{task_id}/session-detail/{session}")
def api_task_session_detail(task_id: str, session: str, eval_qc: Optional[str] = None,
                            load_evaluator: Optional[int] = 0):
    task = find_task(task_id)
    if not task:
        return JSONResponse({"found": False, "message": "任务不存在"}, status_code=404)

    if task.get("source_type") == "workspace":
        assistant = _load_workspace_assistant(session, task["output_dir"])
        if assistant:
            # openclaw: 有 agents/*/sessions 事件流
            result = {"found": True, **assistant}
            # A: 首轮裁决(log 优先, log 无则回退本地 evaluator 轨迹), 挂在 assistant 上供 verdict 面板复用
            result["verdict"] = _extract_workspace_verdict(session, task["output_dir"])
            # B: evaluator 完整轨迹, 仅在前端明确请求时按需拉取(懒加载)
            if load_evaluator:
                evaluator = _ensure_workspace_evaluator(task, session)
                if evaluator is not None:
                    # 把 log 裁决也挂到 evaluator 页, 复用 rubric 面板
                    evaluator["verdict"] = result["verdict"]
                result["evaluator"] = evaluator
            return result

        # Hermes: 无 sessions jsonl, 轨迹+评测都在 query1.json 里
        query1 = _find_query1_json(session, task["output_dir"])
        if not query1:
            return JSONResponse({"found": False, "message": "未找到该会话的 assistant 轨迹文件"}, status_code=404)
        traj = _render_query1_trajectory(query1, session)
        if not traj:
            return JSONResponse({"found": False, "message": "query1.json 解析失败"}, status_code=404)
        result = {"found": True, **traj}
        result["verdict"] = _query1_verdict(query1)
        # 从 state.db 读取 Hermes assistant 的 token 使用量
        task_dir = _workspace_task_dir(session, task["output_dir"])
        if task_dir:
            token_info = _read_hermes_state_db(task_dir)
            if token_info:
                result["token_usage"] = token_info
        # 裁决(rubric/completion)在 query1.json 里, 已挂到 verdict 面板。
        # evaluator 完整轨迹是独立的 profiles/evaluator/sessions/*.json, 默认不随批量下载,
        # 仅在前端明确请求(load_evaluator)时按需拉取(懒加载)。
        if load_evaluator:
            evaluator = _ensure_hermes_evaluator(task, session)
            if evaluator is not None:
                # 把 query1 裁决也挂到 evaluator 页, 复用 rubric 面板
                evaluator["verdict"] = result["verdict"]
            result["evaluator"] = evaluator
        else:
            result["evaluator"] = None
        return result

    assistant = _load_simplified_trajectory(session, task["output_dir"])
    if not assistant:
        return JSONResponse({"found": False, "message": "未找到该会话的原始轨迹文件"}, status_code=404)

    result = {"found": True, **assistant}

    if eval_qc:
        evaluator = _load_simplified_trajectory(eval_qc, task["output_dir"])
        result["evaluator"] = evaluator  # 找不到时为 None, 前端据此隐藏 evaluator 标签页

    return result


def _iter_tar_stream(root_dir: Path, arcname: str, chunk_size: int = 1024 * 1024):
    """把 root_dir 流式打包成 tar(不压缩)按块 yield, 内存占用恒定, 不受目录大小影响。
    origin 目录可达数 GB / 上万文件, 全读进内存或压缩都不可行, 故用非压缩流式 tar:
    tarfile 以 'w|' 流模式写入一个自定义的 fileobj, 每积累到一块就交出去。"""
    class _Buffer(io.RawIOBase):
        def __init__(self):
            self.chunks = []
        def write(self, b):
            self.chunks.append(bytes(b))
            return len(b)

    buf = _Buffer()
    tar = tarfile.open(fileobj=buf, mode="w|")  # 流式, 不压缩(上万小 json, gzip 只拖慢下载)
    try:
        for path in sorted(root_dir.rglob("*")):
            tar.add(path, arcname=os.path.join(arcname, path.relative_to(root_dir).as_posix()),
                    recursive=False)
            # 把 tarfile 已写入 buf 的数据攒够一块就交出去
            if sum(len(c) for c in buf.chunks) >= chunk_size:
                data = b"".join(buf.chunks)
                buf.chunks = []
                yield data
    finally:
        tar.close()  # 补齐 tar 尾部块
    if buf.chunks:
        yield b"".join(buf.chunks)


@app.get("/api/tasks/{task_id}/download-origin")
def api_download_origin(task_id: str):
    """把该任务本地已下载的原始轨迹目录(<output_dir>/origin/)流式打包成 tar 供下载。
    用于把本地这份原始数据整包分享给同事, 不重新访问 OBS。"""
    task = find_task(task_id)
    if not task:
        return JSONResponse({"success": False, "message": "任务不存在"}, status_code=404)

    od = origin_dir(task["output_dir"])
    if not od.is_dir() or not any(od.iterdir()):
        return JSONResponse(
            {"success": False, "message": "该任务本地暂无原始轨迹数据(可能尚未采集或已被清理)"},
            status_code=404,
        )

    # ASCII 兜底名(HTTP 头只能是 latin-1, 中文/特殊字符全替成 _; re.ASCII 让 \w 不匹配中文),
    # 真实中文任务名走 RFC 5987 的 filename* (UTF-8 编码), 现代浏览器优先用它。
    ascii_name = re.sub(r"[^\w.\-]", "_", task.get("name") or task_id, flags=re.ASCII).strip("_") or task_id
    utf8_name = re.sub(r"[\\/:*?\"<>|]", "_", task.get("name") or task_id)  # 仅去掉文件系统非法字符
    ascii_filename = f"{ascii_name}_origin.tar"
    utf8_filename = f"{utf8_name}_origin.tar"
    headers = {
        "Content-Disposition": (
            f"attachment; filename=\"{ascii_filename}\"; "
            f"filename*=UTF-8''{quote(utf8_filename)}"
        )
    }
    # tar 内顶层目录用 ASCII 名, 跨平台解包不出乱码
    return StreamingResponse(
        _iter_tar_stream(od, arcname=f"{ascii_name}_origin"),
        media_type="application/x-tar",
        headers=headers,
    )


@app.get("/api/job-status")
def api_job_status():
    with _job_lock:
        snap = dict(_job_state)
        snap["log_tail"] = list(_job_state["log_tail"])
        # 不暴露进程对象到API响应
        snap.pop("process", None)
        return snap


@app.post("/api/job-stop")
def api_job_stop():
    """终止当前正在运行的采集任务"""
    with _job_lock:
        if not _job_state["running"]:
            return {"success": False, "message": "当前没有正在运行的采集任务"}
        proc = _job_state["process"]
        task_id = _job_state["task_id"]

    if proc is None:
        return {"success": False, "message": "无法获取进程对象"}

    try:
        # 尝试优雅终止
        proc.terminate()
        # 等待最多5秒
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # 强制杀死
            proc.kill()
            proc.wait()

        with _job_lock:
            _job_state["running"] = False
            _job_state["started_at"] = None
            _job_state["last_error"] = "用户手动终止"
            _job_state["last_exit_code"] = -1
            _job_state["process"] = None

        return {"success": True, "message": f"已终止采集任务", "task_id": task_id}
    except Exception as e:
        return {"success": False, "message": f"终止失败: {str(e)}"}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/{full_path:path}")
def serve_index(full_path: str):
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    cfg = load_config()
    print(f"Starting server on http://0.0.0.0:{cfg['port']}")
    uvicorn.run(app, host="0.0.0.0", port=cfg["port"])
