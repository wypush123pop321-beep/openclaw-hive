# -*- coding: utf-8 -*-
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
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

# ── 采集队列 ──────────────────────────────────────────────────────────────────
# 连续登记/重新采集多个任务时, 按登记顺序排队, 由单一后台 worker 逐个下载(全局仍只有
# 一个采集进程在跑, 与 _job_state 的单任务模型一致)。队列元素: {"task_id", "ssh_passwords"}。
# ssh_passwords 只在内存里排队, 采集完即弃, 从不落盘(与既有一次性密码口径一致)。
# _queue_cond 复用 _job_lock 作底层锁, 入队 notify、worker 空队列时 wait。
_pipeline_queue = deque()
_queue_cond = threading.Condition(_job_lock)

_tasks_lock = threading.Lock()
_stats_write_lock = threading.Lock()  # 串行化 filter_stats.json 的「读-改-写」, 防按需回填互相覆盖

# ── 工具调用失败统计: 后台线程状态(按 task_id) ────────────────────────────────
# 「统计工具失败」是重操作(需下载全量 assistant 轨迹再分析), 走后台线程, 前端轮询下面状态。
_toolfail_lock = threading.Lock()
_toolfail_state = {}  # task_id -> {running, done, total, error, result, finished_at}

# ── 会话全量 workspace 下载: 后台准备 + 就绪后流式打包 ─────────────────────────
# 下载单个 session 的完整 workspace 可能较大, 先用后台线程 obsutil 拉到临时目录,
# 前端轮询到 ready 后走流式 tar 下载; 下载(或超时)后临时目录即清。
_wsdl_lock = threading.Lock()
_wsdl_state = {}            # f"{task_id}/{session}" -> {running, ready, error, staging, started_at}
_WS_DL_TTL = 3600           # 就绪后 1 小时未下载则清理临时目录

# ── 会话全量 workspace 缓存: 后台拉取到本地 origin, 长期保留 ────────────────────
# 「下载全量 workspace」是打包成 tar 给浏览器下载(临时目录即用即删); 「缓存全量
# workspace」则是把该会话全量文件落进 <output_dir>/origin/<session>/, 供后续分析直接用。
_wscache_lock = threading.Lock()
_wscache_state = {}         # f"{task_id}/{session}" -> {running, error, finished_at}


def _task_in_queue(task_id: str) -> bool:
    """调用方需持有 _job_lock。"""
    return any(it["task_id"] == task_id for it in _pipeline_queue)


def _enqueue_pipeline(task_id: str, ssh_passwords: Optional[dict] = None) -> dict:
    """把一个采集任务加入队列(按登记顺序). 返回:
      {"state": "running"}            该任务正在采集中(重复触发)
      {"state": "queued", "position": n}  已在队列中(重复) / 新入队, position 为队列中位次(1-based, 不含正在运行的那个)
    """
    with _job_lock:
        if _job_state["running"] and _job_state["task_id"] == task_id:
            return {"state": "running"}
        if _task_in_queue(task_id):
            pos = [it["task_id"] for it in _pipeline_queue].index(task_id) + 1
            return {"state": "queued", "position": pos, "duplicate": True}
        _pipeline_queue.append({"task_id": task_id, "ssh_passwords": ssh_passwords or {}})
        pos = len(_pipeline_queue)
        _queue_cond.notify()
        return {"state": "queued", "position": pos}


def _remove_from_queue(task_id: str) -> bool:
    """从队列中移除一个尚未开始的任务(正在运行的不受影响)。移除成功返回 True。"""
    with _job_lock:
        kept = [it for it in _pipeline_queue if it["task_id"] != task_id]
        removed = len(kept) < len(_pipeline_queue)
        if removed:
            _pipeline_queue.clear()
            _pipeline_queue.extend(kept)
        return removed


def _pipeline_worker():
    """单一后台 worker: 空队列时阻塞等待, 有任务则取队首逐个运行(严格 FIFO)。"""
    while True:
        with _job_lock:
            while not _pipeline_queue:
                _queue_cond.wait()
            item = _pipeline_queue.popleft()
        try:
            run_pipeline(item["task_id"], item["ssh_passwords"])
        except Exception as exc:   # 单个任务异常不拖垮 worker, 继续下一个
            print(f"[queue] run_pipeline 异常 task_id={item['task_id']}: {exc}")


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


def _sanitize_dir_name(name: str, max_bytes: int = 160) -> Optional[str]:
    """把任务名改写成可安全用作文件夹名的形式: 去掉文件系统非法字符(/\\:*?\"<>| 与控制字符)、
    首尾空白与点; 超长按 UTF-8 字节截断; 清洗后为空返回 None(调用方回退到 task_id)。"""
    s = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", (name or "").strip())
    s = s.rstrip(". ").strip("_").strip()
    if not s:
        return None
    while len(s.encode("utf-8")) > max_bytes and len(s) > 1:
        s = s[:-1]
    return s or None


def _task_dir_name(name: str, task_id: str, exclude_dir: Optional[str] = None,
                   tasks_list: Optional[list] = None) -> str:
    """为该任务确定 <output_base_dir>/tasks/ 下的文件夹名: 优先任务名(清洗后),
    与其它任务 / 磁盘上已存在的目录名冲突时加 task_id 后缀。
    exclude_dir: 该任务自己的旧目录名, 重命名时不再把它视作占用(tasks 与磁盘两边都排除,
                免得任务已按任务名命名时被自己卡住, 平白加 task_id 后缀)。
    tasks_list: 可传入外部可变的任务列表(迁移脚本在内存里边改边算, 保证 dry-run == 实跑)。"""
    base = _sanitize_dir_name(name) or f"task_{task_id}"
    used = set()
    for t in (tasks_list if tasks_list is not None else load_tasks()):
        used.add(Path(t["output_dir"]).name)
    if exclude_dir:
        used.discard(exclude_dir)
    tasks_root = Path(load_config()["output_base_dir"]) / "tasks"
    if tasks_root.is_dir():
        used |= {p.name for p in tasks_root.iterdir() if p.is_dir() or p.is_symlink()}
        if exclude_dir:
            used.discard(exclude_dir)
    return base if base not in used else f"{base}_{task_id}"


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
    # 单一采集 worker: 消费采集队列, 按登记顺序逐个下载
    threading.Thread(target=_pipeline_worker, daemon=True).start()
    # 后台异步回填旧任务的 token_stats / task_done_count，避免阻塞 startup
    threading.Thread(target=_backfill_token_stats, daemon=True).start()
    threading.Thread(target=_backfill_task_done, daemon=True).start()


def _acquire_per_task_for_backfill(task: dict) -> Optional[list]:
    """为回填 char_len_stats/token_stats 拿到 per_task(逐条含 char_len/total_tokens)。

    这两列(Estimated Tokens / Avg Char Len)只能由轨迹正文字符数算出, 唯一来源是本地 origin
    下的轨迹文件经 traj_stats.process_root() 现算。老任务(走整包下载路径)本地有完整轨迹, 走这里
    即可回填; fast 采集的任务本地无轨迹(只有 per-task logs/traj_stats_result.json, 其中无 char_len),
    回填拿不到数据 -> 返回 None, 两列保持留空, 待用户按需下载轨迹时由
    _ensure_workspace_session_files() 渐进补算。"""
    origin = origin_dir(task["output_dir"])
    if origin.is_dir() and any(origin.iterdir()):
        per_task = traj_stats.process_root(str(origin))
        if per_task:
            return per_task
    return None


def _backfill_token_stats():
    """启动时回填旧任务的 token_stats/char_len_stats（新任务在 pipeline 阶段已写入）。

    「Estimated Tokens(L1)」与「Avg Char Len(L1)」两列都取自 char_len_stats.L1, 很多旧任务
    缺这一档。本函数对缺 L1 档的任务, 用本地轨迹重新聚合 per_task(见
    _acquire_per_task_for_backfill), 计算各 tier 平均值后写回。fast 采集的任务本地无轨迹,
    拿不到 char_len -> 跳过, 两列留空, 待按需下载轨迹时渐进补算。
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
        ts0 = data.get("token_stats") or {}
        cs0 = data.get("char_len_stats") or {}
        if "L1" in cs0 and "L1" in ts0:
            continue                # L1 档 token/char 均已有, 跳过(两列已可展示)

        # 拿 per_task(优先复用采集侧 traj_stats_result.json, 见 _acquire_per_task_for_backfill)
        per_task = _acquire_per_task_for_backfill(t)
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
        # 工具调用失败统计(需先点「统计工具失败」生成; 未生成时缺省, 前端显示 —)
        if data.get("tool_fail_stats"):
            summary["tool_fail_stats"] = data["tool_fail_stats"]
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
    # 任务文件夹按任务名命名(清洗/去重), 便于人眼直接对应; 命名冲突时追加 task_id 后缀
    tasks = load_tasks()
    output_dir = str(Path(cfg["output_base_dir"]) / "tasks" / _task_dir_name(name, task_id))
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
    tasks.append(task)
    save_tasks(tasks)

    # 入队, 由单一 worker 按登记顺序逐个采集
    enq = _enqueue_pipeline(task_id, ssh_passwords)
    with _job_lock:
        busy = _job_state["running"]
    if not busy and enq.get("position") == 1:
        return {"success": True, "task": task, "started": True, "queued": True,
                "queue_position": 1, "message": "任务已创建，正在采集数据"}
    pos = enq.get("position", 1)
    return {"success": True, "task": task, "started": False, "queued": True,
            "queue_position": pos,
            "message": f"任务已创建，已加入采集队列（排在第 {pos} 位，将按登记顺序依次采集）"}


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

    enq = _enqueue_pipeline(task_id, ssh_passwords)
    if enq["state"] == "running":
        return {"success": False, "message": "该任务正在采集中"}
    with _job_lock:
        busy = _job_state["running"]
    pos = enq.get("position", 1)
    if enq.get("duplicate"):
        return {"success": False, "message": f"该任务已在采集队列中（第 {pos} 位）"}
    if not busy and pos == 1:
        return {"success": True, "started": True, "queue_position": 1, "message": "已开始采集"}
    return {"success": True, "started": False, "queue_position": pos,
            "message": f"已加入采集队列（第 {pos} 位）"}


@app.patch("/api/tasks/{task_id}")
def api_rename_task(task_id: str, body: dict):
    task = find_task(task_id)
    if not task:
        return JSONResponse({"success": False, "message": "任务不存在"}, status_code=404)
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"success": False, "message": "任务名称不能为空"}, status_code=400)
    # 采集中禁止重命名: 目录移动会打断正在写入的子进程
    with _job_lock:
        if _job_state["running"] and _job_state["task_id"] == task_id:
            return JSONResponse({"success": False, "message": "该任务正在采集中，暂不能重命名"}, status_code=409)

    old_od = Path(task["output_dir"])
    new_name = _task_dir_name(name, task_id, exclude_dir=old_od.name)
    new_od = old_od.parent / new_name
    # 目录确实存在且确实要改名: 原地 mv(同一文件系统, 便宜), 数据不复制
    if str(new_od) != str(old_od):
        if new_od.exists():
            return JSONResponse({"success": False, "message": f"目标文件夹已存在: {new_name}"}, status_code=409)
        if old_od.exists():
            old_od.rename(new_od)
    update_task(task_id, name=name, output_dir=str(new_od))
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

    _remove_from_queue(task_id)   # 若在采集队列里排队, 先出队再删数据
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


def _get_session_level(s: dict) -> str:
    """返回该会话的最高层级: L3 > L2 > L1.5 > L1 > L0。"""
    if s.get("has_eval") and s.get("completion") == 1:
        return "L3"
    if s.get("has_eval") and isinstance(s.get("completion"), (int, float)) and s["completion"] >= 0.5:
        return "L2"
    if s.get("has_eval"):
        return "L1.5"
    if s.get("passed_gate"):
        return "L1"
    return "L0"


@app.get("/api/tasks/{task_id}/sessions")
def api_task_sessions(
    task_id: str,
    page: int = 1,
    page_size: int = 20,
    has_eval: Optional[bool] = None,
    completion_filter: Optional[str] = None,  # "ge05" | "eq1" | "no_eval"
    level_filter: Optional[str] = None,       # "L0" / "L1" / "L1.5" / "L2" / "L3", 逗号分隔多选
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

    # 为每个会话计算层级并筛选
    levels = set()
    want_task_done = None   # None=不限制, True=只要 task_done, False=排除 task_done
    if level_filter:
        parts = [l.strip() for l in level_filter.split(",")]
        if "TASK_DONE" in parts:
            want_task_done = True
        levels = {l for l in parts if l in ("L0", "L1", "L1.5", "L2", "L3")}
        if levels and want_task_done is None:
            want_task_done = False   # 选了层级但没勾 TASK_DONE → 排除有 TASK_DONE 的
    for s in sessions:
        s["level"] = _get_session_level(s)
    if levels:
        sessions = [s for s in sessions if s["level"] in levels]
    if want_task_done is True:
        sessions = [s for s in sessions if s.get("task_done")]
    elif want_task_done is False:
        sessions = [s for s in sessions if not s.get("task_done")]

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

def _simplify_workspace_message(role: str, parts: list, msg: Optional[dict] = None) -> Optional[dict]:
    """把 workspace assistant/evaluator .jsonl 的一条 message 的 content 部件列表,
    映射成前端已认的结构 {role, content, reasoning_content, thinking_signatures, tool_calls, truncated}。
    部件类型: thinking / text / toolCall / (toolResult 侧的) text。
    thinking 部件可选携带 thinkingSignature(opaque token), 单独透传供前端高亮显示。
    msg: 完整 message 对象, 用于捞出 content 之外的 message 层元数据
         (toolResult 的 toolName / isError / details.exitCode 等)。"""
    texts, reasonings, tool_calls = [], [], []
    signatures = []
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
            sig = p.get("thinkingSignature")
            if sig:
                signatures.append(sig)
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
    if signatures:
        out["thinking_signatures"] = signatures
    if tool_calls:
        out["tool_calls"] = tool_calls

    # toolResult 侧的 message 层元数据: 工具名 / 是否报错 / 退出码 / 结构化 details
    if isinstance(msg, dict) and role == "toolResult":
        if msg.get("toolName"):
            out["tool_name"] = msg["toolName"]
        if msg.get("isError"):
            out["is_error"] = True
        details = msg.get("details")
        if isinstance(details, dict):
            # exec 类工具的退出码; 非 0 视作失败信号
            if details.get("exitCode") is not None:
                out["exit_code"] = details["exitCode"]
            det_str = json.dumps(details, ensure_ascii=False, indent=2)
            if len(det_str) > _MAX_MSG_CHARS:
                det_str = det_str[:_MAX_MSG_CHARS]
                out["details_truncated"] = True
            out["details"] = det_str
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
                simplified = _simplify_workspace_message(role, content, msg)
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


# 任务 log 查看: 单个日志文件最大回传字节数(过大只回传尾部, 日志尾部通常含裁决/结论)。
_LOG_MAX_BYTES = 2 * 1024 * 1024


def _workspace_log_rel(task_dir: Path, session: str) -> str:
    """该 session 主 log 的相对路径: 优先取 traj_stats_result.json 里的 log_file, 回退 logs/<session>.log。"""
    tsr = task_dir / "logs" / "traj_stats_result.json"
    if tsr.exists():
        try:
            with open(tsr, encoding="utf-8", errors="replace") as f:
                lf = (json.load(f).get("log_file") or "").strip()
            if lf and ".." not in lf and not lf.startswith("/"):
                return lf
        except (json.JSONDecodeError, OSError):
            pass
    return f"logs/{session}.log"


def _ensure_workspace_log(task: dict, session: str) -> Optional[Path]:
    """返回该 session 主 log 的本地路径; 本地无则用 workspace_obs 按需下载(懒加载)。
    快速路径采集只留 traj_stats_result.json, log 需按需拉取。返回 None 表示无法获取。"""
    output_dir = task["output_dir"]
    if "/" in session or ".." in session:
        return None
    task_dir = origin_dir(output_dir) / session
    rel = _workspace_log_rel(task_dir, session)
    local = task_dir / rel
    if local.exists():
        return local
    workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
    if not workspace_obs:
        return None
    cfg = load_config()
    task_obs = f"{workspace_obs}/{session}/{rel}"
    dest = str(task_dir / Path(rel).parent) + "/"
    cmd = [cfg["obsutil_path"], "cp", task_obs, dest, "-f"] + _obs_cred_args_for_task(task)
    try:
        subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    except Exception:
        return None
    return local if local.exists() else None


def _load_session_log_content(task: dict, session: str) -> Optional[dict]:
    """读取本地已缓存的 session 主 log 内容(纯本地读, 不触发按需下载)。
    _ensure_workspace_session_files 已拉下 *logs*.log, 此函数只做本地读取。"""
    if "/" in session or ".." in session:
        return None
    task_dir = origin_dir(task["output_dir"]) / session
    if not task_dir.is_dir():
        return None
    rel = _workspace_log_rel(task_dir, session)
    log_path = task_dir / rel
    if not log_path.exists():
        return None
    try:
        size = log_path.stat().st_size
        truncated = size > _LOG_MAX_BYTES
        with open(log_path, "rb") as f:
            if truncated:
                f.seek(size - _LOG_MAX_BYTES)
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        if truncated:
            nl = text.find("\n")
            if nl >= 0:
                text = text[nl + 1:]
        verdict = _extract_verdict_from_log(session, task["output_dir"])
        return {"filename": log_path.name, "size": size, "truncated": truncated, "log": text, "verdict": verdict}
    except OSError:
        return None


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


# 详情页按需下载单个 session 所需文件时的 include/exclude(evaluator 仍独立懒加载, 此处不下)。
# 与 download_workspace_and_run.py 的 INCLUDE_PATTERNS 对齐, 去掉 evaluator sessions。
_SESSION_INCLUDE = [
    "*assistant*sessions*.jsonl", "*agents/main/sessions/*.jsonl",
    "*logs*.log",
    "*logs/trajectories/*query*.json",
    "*profiles/assistant*/sessions/*.json", "*profiles/main/sessions/*.json",
    "*profiles/assistant*/state.db*",
]
_SESSION_EXCLUDE = ["*.trajectory.jsonl", "*_use.log", "*profiles/*/logs/*", "*_logs/*.log"]


def _workspace_session_has_traj(task_dir: Optional[Path]) -> bool:
    """本地是否已有该 session 的 assistant 轨迹(openclaw jsonl 或 Hermes query1/profiles)。"""
    if not task_dir or not task_dir.is_dir():
        return False
    for pattern in ("agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"):
        for p in task_dir.glob(pattern):
            if "trajectory" not in p.name:
                return True
    for pattern in ("logs/trajectories/*/query*.json",
                    "profiles/assistant*/sessions/*.json", "profiles/main/sessions/*.json"):
        if next(iter(task_dir.glob(pattern)), None) is not None:
            return True
    return False


def _recompute_char_len_stats(data: dict) -> Optional[dict]:
    """按 per_session 的 char_len 重新聚合各 tier 的平均字符数(供按需回填后刷新)。

    tier 口径与 pipeline 的 stats_from_per_task 完全一致(每档都是上一档子集):
      L0=全部, L1=passed_gate, L1.5=L1且有数值 completion, L2=L1.5且>=0.5,
      L3=L1.5且==1, T_DONE=task_done。只统计已有 char_len 的会话(用户查看过详情的),
      故 fast 采集任务是「渐进填充」——查看越多会话, 该均值覆盖面越大。全无 char_len 返回 None。
    """
    sessions = data.get("per_session") or []
    tiers = {"L0": [], "L1": [], "L1.5": [], "L2": [], "L3": [], "T_DONE": []}
    for s in sessions:
        cl = s.get("char_len")
        if not isinstance(cl, (int, float)):
            continue
        comp = s.get("completion")
        has_eval = bool(s.get("has_eval"))
        passed = bool(s.get("passed_gate"))
        tiers["L0"].append(cl)
        if s.get("task_done"):
            tiers["T_DONE"].append(cl)
        if passed:
            tiers["L1"].append(cl)
            if has_eval:
                tiers["L1.5"].append(cl)
                if isinstance(comp, (int, float)) and comp >= 0.5:
                    tiers["L2"].append(cl)
                if isinstance(comp, (int, float)) and comp == 1:
                    tiers["L3"].append(cl)
    out = {}
    for tier, vals in tiers.items():
        if vals:
            total = sum(vals)
            out[tier] = {"avg_char_len": round(total / len(vals)),
                         "sum_total": total, "count": len(vals)}
    return out or None


def _backfill_session_char_len(task: dict, session: str) -> None:
    """某 session 的轨迹文件已在本地时, 算出其 char_len 写回 filter_stats.json 并刷新 char_len_stats。

    快速路径采集时两列(Estimated Tokens/Avg Char Len)留空, 用户查看某会话详情触发轨迹下载后,
    在此就地补算该会话字符数并重聚合, 使两列渐进填上(与「按需算」设计一致)。
    只处理 workspace 来源; 会话已有 char_len 或本地无轨迹文件时为 no-op。
    """
    if task.get("source_type") != "workspace":
        return
    sp = stats_path(task["output_dir"])
    if not sp.exists():
        return
    origin = origin_dir(task["output_dir"])
    with _stats_write_lock:
        try:
            with open(sp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        sessions = data.get("per_session") or []
        entry = next((s for s in sessions if s.get("session") == session), None)
        if entry is None or isinstance(entry.get("char_len"), (int, float)):
            return  # 无此会话 或 已回填过

        # 定位轨迹文件: 优先用 per_session.trajectory(origin 相对路径), 兜底 glob 该 session 目录
        traj_rel = entry.get("trajectory")
        traj_path = None
        if traj_rel:
            cand = origin / traj_rel
            if cand.is_file():
                traj_path = cand
        if traj_path is None:
            sess_dir = origin / session
            for pat in ("agents/*/sessions/*.jsonl", "logs/trajectories/*/query*.json",
                        "profiles/*/sessions/*.json"):
                for p in sess_dir.glob(pat):
                    if "trajectory" not in p.name:
                        traj_path = p
                        break
                if traj_path:
                    break
        if traj_path is None:
            return  # 本地还没有该会话轨迹, 留待下次

        entry["char_len"] = traj_stats._char_len(str(traj_path))
        cs = _recompute_char_len_stats(data)
        if cs:
            data["char_len_stats"] = cs
        try:
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass


def _ensure_workspace_session_files(task: dict, session: str) -> None:
    """确保单个 session 的 assistant 轨迹文件在本地; 缺失则从 workspace_obs 按需下载。

    快速路径采集(见 download_workspace_and_run.py)只下 traj_stats_result.json, origin 下并无
    具体轨迹文件, 首次查看详情时在此按需拉取该 session 的必要文件(不含 evaluator, 它另有懒加载)。
    下载(或确认已存在)后就地补算该会话 char_len, 使 Estimated Tokens/Avg Char Len 两列渐进填上。
    """
    if "/" in session or ".." in session:
        return
    output_dir = task["output_dir"]
    task_dir = origin_dir(output_dir) / session
    if _workspace_session_has_traj(task_dir):
        # 本地已有轨迹(本会话之前下过): 仍尝试补算 char_len(可能上次没算/是老缓存)
        _backfill_session_char_len(task, session)
        return
    workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
    if not workspace_obs:
        return
    cfg = load_config()
    task_obs = f"{workspace_obs}/{session}/"
    origin = str(origin_dir(output_dir))
    cmd = [cfg["obsutil_path"], "cp", task_obs, origin, "-r", "-f"]
    for p in _SESSION_INCLUDE:
        cmd += ["-include", p]
    for p in _SESSION_EXCLUDE:
        cmd += ["-exclude", p]
    cmd += _obs_cred_args_for_task(task)
    try:
        subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180)
    except Exception:
        return
    # 下载成功: 就地补算该会话 char_len 并刷新 char_len_stats
    _backfill_session_char_len(task, session)


@app.get("/api/tasks/{task_id}/session-detail/{session}")
def api_task_session_detail(task_id: str, session: str, eval_qc: Optional[str] = None,
                            load_evaluator: Optional[int] = 0):
    task = find_task(task_id)
    if not task:
        return JSONResponse({"found": False, "message": "任务不存在"}, status_code=404)

    if task.get("source_type") == "workspace":
        # 快速路径采集下 origin 无具体轨迹, 首次查看时按需下载该 session 的 assistant 文件
        _ensure_workspace_session_files(task, session)
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
            # C: 主 log 内容(已在 _ensure_workspace_session_files 下到本地), 前端直接秒开
            result["task_log"] = _load_session_log_content(task, session)
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
        # C: 主 log 内容(已在 _ensure_workspace_session_files 下到本地)
        result["task_log"] = _load_session_log_content(task, session)
        return result

    assistant = _load_simplified_trajectory(session, task["output_dir"])
    if not assistant:
        return JSONResponse({"found": False, "message": "未找到该会话的原始轨迹文件"}, status_code=404)

    result = {"found": True, **assistant}

    if eval_qc:
        evaluator = _load_simplified_trajectory(eval_qc, task["output_dir"])
        result["evaluator"] = evaluator  # 找不到时为 None, 前端据此隐藏 evaluator 标签页

    return result


@app.get("/api/tasks/{task_id}/session-log/{session}")
def api_task_session_log(task_id: str, session: str):
    """返回该 session 的主 log 文本(与 assistant/evaluator 轨迹同级的「任务 Log」标签)。
    workspace 来源: 本地无则从 workspace_obs 按需下载。过大只回传尾部 _LOG_MAX_BYTES 字节。"""
    task = find_task(task_id)
    if not task:
        return JSONResponse({"found": False, "message": "任务不存在"}, status_code=404)
    if task.get("source_type") != "workspace":
        return JSONResponse({"found": False, "message": "该来源无独立 log 文件"}, status_code=404)

    log_path = _ensure_workspace_log(task, session)
    if not log_path or not log_path.exists():
        return JSONResponse({"found": False, "message": "未找到该会话的 log 文件(可能无地址或下载失败)"},
                            status_code=404)
    try:
        size = log_path.stat().st_size
        truncated = size > _LOG_MAX_BYTES
        with open(log_path, "rb") as f:
            if truncated:
                f.seek(size - _LOG_MAX_BYTES)
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        if truncated:
            # 从第一个换行切齐, 避免半个多字节字符/半行
            nl = text.find("\n")
            if nl >= 0:
                text = text[nl + 1:]
    except OSError:
        return JSONResponse({"found": False, "message": "读取 log 文件失败"}, status_code=500)
    verdict = _extract_verdict_from_log(session, task["output_dir"])
    return {
        "found": True,
        "filename": log_path.name,
        "size": size,
        "truncated": truncated,
        "log": text,
        "verdict": verdict,
    }


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


def _run_tool_failure_analysis(task: dict):
    """后台线程: 下载全量 assistant 轨迹 + 统计 tool_call 失败, 写回 filter_stats.json。
    进度/结果写入 _toolfail_state[task_id] 供前端轮询。"""
    import analyze_tool_failures  # 延迟导入, 避免与 server 循环导入
    tid = task["id"]

    def progress(done, total):
        with _toolfail_lock:
            st = _toolfail_state.get(tid)
            if st is not None:
                st["done"] = done
                st["total"] = total

    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        with _stats_write_lock:  # 与 char_len/token 回填串行, 防 filter_stats.json 互相覆盖
            result = analyze_tool_failures.analyze(task, progress=progress, now_iso=now_iso)
        with _toolfail_lock:
            _toolfail_state[tid] = {
                "running": False, "done": result.get("analyzed_sessions", 0),
                "total": result.get("analyzed_sessions", 0), "error": None,
                "result": result, "finished_at": now_iso,
            }
    except Exception as e:  # noqa: BLE001  失败也要落到状态里让前端看到
        with _toolfail_lock:
            st = _toolfail_state.get(tid, {})
            st.update({"running": False, "error": str(e), "result": None})
            _toolfail_state[tid] = st


@app.post("/api/tasks/{task_id}/analyze-tool-failures")
def api_analyze_tool_failures(task_id: str):
    """触发「统计工具失败」: 起后台线程下载全量 assistant 轨迹并统计失败次数。"""
    task = find_task(task_id)
    if not task:
        return JSONResponse({"success": False, "message": "任务不存在"}, status_code=404)
    if task.get("source_type") != "workspace":
        return JSONResponse(
            {"success": False, "message": "仅 workspace 来源的任务支持工具失败统计"},
            status_code=400,
        )
    with _toolfail_lock:
        st = _toolfail_state.get(task_id)
        if st and st.get("running"):
            return {"success": False, "message": "该任务正在统计工具失败，请稍候"}
        _toolfail_state[task_id] = {
            "running": True, "done": 0, "total": 0,
            "error": None, "result": None, "finished_at": None,
        }
    threading.Thread(target=_run_tool_failure_analysis, args=(task,), daemon=True).start()
    return {"success": True, "message": "已开始统计工具失败（后台下载轨迹并分析，请稍候）"}


@app.get("/api/tasks/{task_id}/tool-failure-status")
def api_tool_failure_status(task_id: str):
    """前端轮询「统计工具失败」进度/结果。无记录表示从未触发过。"""
    with _toolfail_lock:
        st = _toolfail_state.get(task_id)
        return {"found": st is not None, "status": dict(st) if st else None}


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


def _prepare_session_workspace(task: dict, session: str, key: str):
    """后台线程: 把 <workspace_obs>/<session>/ 全量拉到临时目录, 结果写入 _wsdl_state[key]。
    下载成功且 session 目录存在才标记 ready(临时目录即作 tar 源)。"""
    workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
    task_obs = f"{workspace_obs}/{session}/"
    cfg = load_config()
    # obsutil cp <prefix>/ <dest> -r 会落成 <dest>/<session>/... (prefix 末段多一层), 正好作 tar 顶层
    staging = Path(tempfile.mkdtemp(prefix=f"session_ws_{session[:24].replace('/', '_')}_"))

    def _done(**fields):
        with _wsdl_lock:
            _wsdl_state[key] = {"running": False, "ready": False, "error": None, "staging": None,
                                "started_at": time.time(), **fields}

    try:
        cmd = [cfg["obsutil_path"], "cp", task_obs, str(staging), "-r", "-f"]
        cmd += _obs_cred_args_for_task(task)
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=1800)
    except subprocess.TimeoutExpired:
        shutil.rmtree(staging, ignore_errors=True)
        _done(error="OBS 下载超时(可能 workspace 过大)")
        return
    if res.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        _done(error=f"OBS 下载失败: {(res.stderr or '')[-300:]}")
        return
    if not (staging / session).is_dir():
        shutil.rmtree(staging, ignore_errors=True)
        _done(error="OBS 上未找到该 session 的 workspace")
        return
    with _wsdl_lock:
        _wsdl_state[key] = {"running": False, "ready": True, "error": None,
                            "staging": str(staging), "started_at": time.time()}


@app.post("/api/tasks/{task_id}/session-workspace/{session}/prepare")
def api_session_workspace_prepare(task_id: str, session: str):
    """触发准备: 后台下载该 session 的全量 workspace 到临时目录, 供随后流式下载。"""
    task = find_task(task_id)
    if not task:
        return JSONResponse({"success": False, "message": "任务不存在"}, status_code=404)
    if task.get("source_type") != "workspace":
        return JSONResponse({"success": False, "message": "仅 workspace 来源的任务支持下载全量 workspace"},
                            status_code=400)
    if not session or "/" in session or ".." in session:
        return JSONResponse({"success": False, "message": "非法 session"}, status_code=400)
    if not (task.get("workspace_obs") or "").rstrip("/"):
        return JSONResponse({"success": False, "message": "该任务无 workspace_obs"}, status_code=400)

    key = f"{task_id}/{session}"
    with _wsdl_lock:
        st = _wsdl_state.get(key)
        if st and st.get("running"):
            return {"success": True, "running": True, "ready": False, "message": "该会话 workspace 正在准备中"}
        if st and st.get("ready"):
            return {"success": True, "running": False, "ready": True, "message": "该会话 workspace 已就绪"}
        _wsdl_state[key] = {"running": True, "ready": False, "error": None,
                            "staging": None, "started_at": time.time()}
    threading.Thread(target=_prepare_session_workspace, args=(task, session, key), daemon=True).start()
    return {"success": True, "running": True, "ready": False, "message": "已开始准备该会话的全量 workspace"}


@app.get("/api/tasks/{task_id}/session-workspace/{session}/status")
def api_session_workspace_status(task_id: str, session: str):
    """前端轮询: {found, running, ready, error}。就绪后超过 TTL 未下载则清掉临时目录。"""
    key = f"{task_id}/{session}"
    with _wsdl_lock:
        st = _wsdl_state.get(key)
        if not st:
            return {"found": False}
        if st.get("ready") and time.time() - st.get("started_at", 0) > _WS_DL_TTL:
            if st.get("staging"):
                shutil.rmtree(st.get("staging"), ignore_errors=True)
            _wsdl_state.pop(key, None)
            return {"found": False}
        return {"found": True, "running": st.get("running"), "ready": st.get("ready"),
                "error": st.get("error")}


@app.get("/api/tasks/{task_id}/session-workspace/{session}/download")
def api_session_workspace_download(task_id: str, session: str):
    """就绪后流式打包下载(取走即清临时目录, 不重复)。"""
    key = f"{task_id}/{session}"
    with _wsdl_lock:
        st = _wsdl_state.get(key)
        if not st or not st.get("ready") or not st.get("staging"):
            return JSONResponse({"success": False, "message": "workspace 尚未准备好, 请先触发准备"}, status_code=409)
        staging = Path(st["staging"])
        _wsdl_state.pop(key, None)

    ascii_name = re.sub(r"[^\w.\-]", "_", session, flags=re.ASCII).strip("_") or "session"
    utf8_name = re.sub(r"[\\/:*?\"<>|]", "_", session) or "session"
    headers = {
        "Content-Disposition": (
            f'attachment; filename="{ascii_name}_workspace.tar"; '
            f"filename*=UTF-8''{quote(utf8_name + '_workspace.tar')}"
        )
    }

    def _stream():
        try:
            yield from _iter_tar_stream(staging, arcname=ascii_name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    return StreamingResponse(_stream(), media_type="application/x-tar", headers=headers)


def _cache_session_workspace(task: dict, session: str, key: str):
    """后台线程: 把 <workspace_obs>/<session>/ 全量拉取到 <output_dir>/origin/<session>/。
    obsutil cp <prefix>/ <origin> -r 会按 prefix 末段建一层, <session>/ 正好落成 origin/<session>/,
    与既有本地目录结构一致; -f 增量覆盖刷新, 不删除本地已有的其它文件。"""
    workspace_obs = (task.get("workspace_obs") or "").rstrip("/")
    task_obs = f"{workspace_obs}/{session}/"
    cfg = load_config()
    origin = origin_dir(task["output_dir"])
    origin.mkdir(parents=True, exist_ok=True)
    cmd = [cfg["obsutil_path"], "cp", task_obs, str(origin), "-r", "-f"]
    cmd += _obs_cred_args_for_task(task)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=3600)
    except subprocess.TimeoutExpired:
        err = "OBS 下载超时(可能 workspace 过大)"
    else:
        err = None if res.returncode == 0 else f"OBS 下载失败: {(res.stderr or '')[-300:]}"
    if err is None and not (origin / session).is_dir():
        err = "OBS 上未找到该 session 的 workspace"
    with _wscache_lock:
        _wscache_state[key] = {"running": False, "error": err, "finished_at": time.time()}


@app.post("/api/tasks/{task_id}/session-workspace/{session}/cache")
def api_session_workspace_cache(task_id: str, session: str):
    """触发「缓存全量 workspace」: 后台把该会话全量文件拉取到本地 origin, 长期保留。"""
    task = find_task(task_id)
    if not task:
        return JSONResponse({"success": False, "message": "任务不存在"}, status_code=404)
    if task.get("source_type") != "workspace":
        return JSONResponse({"success": False, "message": "仅 workspace 来源的任务支持缓存全量 workspace"},
                            status_code=400)
    if not session or "/" in session or ".." in session:
        return JSONResponse({"success": False, "message": "非法 session"}, status_code=400)
    if not (task.get("workspace_obs") or "").rstrip("/"):
        return JSONResponse({"success": False, "message": "该任务无 workspace_obs"}, status_code=400)

    key = f"{task_id}/{session}"
    with _wscache_lock:
        st = _wscache_state.get(key)
        if st and st.get("running"):
            return {"success": True, "running": True, "message": "该会话正在缓存全量 workspace"}
        _wscache_state[key] = {"running": True, "error": None, "finished_at": None}
    threading.Thread(target=_cache_session_workspace, args=(task, session, key), daemon=True).start()
    return {"success": True, "running": True, "message": "已开始缓存该会话的全量 workspace 到本地"}


@app.get("/api/tasks/{task_id}/session-workspace/{session}/cache-status")
def api_session_workspace_cache_status(task_id: str, session: str):
    """前端轮询缓存进度: {found, running, error}。"""
    key = f"{task_id}/{session}"
    with _wscache_lock:
        st = _wscache_state.get(key)
        if not st:
            return {"found": False}
        return {"found": True, "running": st.get("running"), "error": st.get("error")}


@app.get("/api/job-status")
def api_job_status():
    with _job_lock:
        snap = dict(_job_state)
        snap["log_tail"] = list(_job_state["log_tail"])
        queued_ids = [it["task_id"] for it in _pipeline_queue]
        # 不暴露进程对象到API响应
        snap.pop("process", None)
    # 队列任务名映射放到锁外, 避免在 _job_lock 内做文件 I/O
    id2name = {t["id"]: t.get("name") for t in load_tasks()}
    snap["queue"] = [{"task_id": tid, "name": id2name.get(tid, tid)} for tid in queued_ids]
    snap["queue_len"] = len(queued_ids)
    return snap


@app.post("/api/tasks/{task_id}/dequeue")
def api_task_dequeue(task_id: str):
    """把一个尚在排队(未开始)的任务移出采集队列。正在运行的请用「终止采集」。"""
    with _job_lock:
        if _job_state["running"] and _job_state["task_id"] == task_id:
            return {"success": False, "message": "该任务正在采集中，请用「终止采集」"}
    if _remove_from_queue(task_id):
        return {"success": True, "message": "已移出采集队列"}
    return {"success": False, "message": "该任务不在采集队列中"}


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
