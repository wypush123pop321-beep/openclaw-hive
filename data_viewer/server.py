# -*- coding: utf-8 -*-
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
PIPELINE_SCRIPT = HERE.parent / "traj_pipeline" / "download_and_run.py"
STATIC_DIR = HERE / "static"
CONFIG_FILE = HERE / "config.json"
STATS_FILE_NAME = "filter_stats.json"

DEFAULT_CONFIG = {
    "assistant_obs": "obs://rl-agentdata/zhengnianzu/test/session_analysis/env-claude-99oR/key-6fda/ex-260714192731/",
    "evaluator_obs": "obs://rl-agentdata/zhengnianzu/test/session_analysis/env-claude-99oR/key-b771/ex-260716211014/",
    "output_dir": str(HERE / "pipeline_output"),
    "obsutil_path": "/home/ma-user/obsutil/obsutil",
    "refresh_interval_minutes": 30,
    "port": 8080,
}

_job_lock = threading.Lock()
_job_state = {
    "running": False,
    "started_at": None,
    "progress": "",
    "log_tail": [],
    "last_run_time": None,
    "last_duration_seconds": None,
    "last_error": None,
    "last_exit_code": None,
}
_LOG_TAIL_MAX = 40


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return DEFAULT_CONFIG.copy()


def stats_path() -> Path:
    return Path(load_config()["output_dir"]) / STATS_FILE_NAME


def origin_dir() -> Path:
    return Path(load_config()["output_dir"]) / "origin"


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


def find_session_json(session: str) -> Optional[Path]:
    """在 <output_dir>/origin/*/<session>/ 下查找该 session 的最新原始轨迹 json。
    assistant 目录名每次下载会变(取决于 obs 路径末段), 所以用通配符搜, 不写死目录名。"""
    if "/" in session or ".." in session:
        return None
    for candidate in origin_dir().glob(f"*/{session}"):
        if candidate.is_dir():
            found = _latest_json(candidate)
            if found:
                return found
    return None


_MAX_MSG_CHARS = 6000


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


def _stream_subprocess(cmd):
    """按字符读取子进程输出, 遇 \\r/\\n 断行(与 download_and_run.py 里 obsutil 的读取方式一致),
    这样下载进度这种用 \\r 原地刷新的行也能被实时捕获, 而不必等进程退出才能拿到全部输出。"""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
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
    return proc.returncode, "\n".join(all_output)


def run_pipeline():
    with _job_lock:
        if _job_state["running"]:
            return
        _job_state["running"] = True
        _job_state["started_at"] = datetime.now().isoformat()
        _job_state["progress"] = "正在启动..."
        _job_state["log_tail"] = []
        _job_state["last_error"] = None

    t0 = time.time()
    try:
        cfg = load_config()
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            str(PIPELINE_SCRIPT),
            cfg["assistant_obs"],
            cfg["evaluator_obs"],
            out_dir,
            "--obsutil", cfg["obsutil_path"],
        ]
        exit_code, full_output = _stream_subprocess(cmd)
        with _job_lock:
            _job_state["last_run_time"] = datetime.now().isoformat()
            _job_state["last_duration_seconds"] = round(time.time() - t0, 1)
            _job_state["last_exit_code"] = exit_code
            if exit_code != 0:
                _job_state["last_error"] = full_output[-3000:].strip()
    except Exception as exc:
        with _job_lock:
            _job_state["last_run_time"] = datetime.now().isoformat()
            _job_state["last_duration_seconds"] = round(time.time() - t0, 1)
            _job_state["last_error"] = str(exc)
    finally:
        with _job_lock:
            _job_state["running"] = False
            _job_state["started_at"] = None


def _scheduler_loop():
    while True:
        cfg = load_config()
        interval = max(1, cfg.get("refresh_interval_minutes", 30)) * 60
        time.sleep(interval)
        run_pipeline()


app = FastAPI(title="Trajectory Viewer")


@app.on_event("startup")
def _startup():
    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()


@app.get("/api/stats")
def api_stats():
    p = stats_path()
    if not p.exists():
        return JSONResponse({"available": False, "message": "尚无数据，请先运行流水线"})
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    summary = {k: v for k, v in data.items() if k != "per_session"}
    summary["available"] = True
    summary["session_total"] = len(data.get("per_session", []))
    return summary


@app.get("/api/sessions")
def api_sessions(
    page: int = 1,
    page_size: int = 20,
    has_eval: Optional[bool] = None,
    completion_filter: Optional[str] = None,  # "ge05" | "eq1" | "no_eval"
):
    p = stats_path()
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


@app.get("/api/session-detail/{session}")
def api_session_detail(session: str):
    """按需拉取单个 session 的原始轨迹(裁剪后), 供前端点开会话详情时懒加载, 避免一次性把全部轨迹(单条最大 700KB+)传给前端。"""
    json_path = find_session_json(session)
    if not json_path:
        return JSONResponse({"found": False, "message": "未找到该会话的原始轨迹文件"}, status_code=404)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    return {
        "found": True,
        "session": session,
        "source_file": str(json_path),
        "model": data.get("model"),
        "message_count": len(messages),
        "messages": [_simplify_message(m) for m in messages],
    }


@app.post("/api/trigger")
def api_trigger():
    with _job_lock:
        if _job_state["running"]:
            return {"success": False, "message": "流水线正在运行中"}
    threading.Thread(target=run_pipeline, daemon=True).start()
    return {"success": True, "message": "流水线已启动"}


@app.get("/api/job-status")
def api_job_status():
    with _job_lock:
        snap = dict(_job_state)
        snap["log_tail"] = list(_job_state["log_tail"])
        return snap


@app.get("/api/config")
def api_config():
    return load_config()


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/{full_path:path}")
def serve_index(full_path: str):
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    cfg = load_config()
    print(f"Starting server on http://0.0.0.0:{cfg['port']}")
    uvicorn.run(app, host="0.0.0.0", port=cfg["port"])
