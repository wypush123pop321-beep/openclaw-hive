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
  python download_workspace_and_run.py <workspace_obs> <out_dir> \\
         [--obsutil PATH] [--max-tasks N] [--concurrency N] [--hermes]

示例:
  python download_workspace_and_run.py \
    "obs://rl-agentdata/openclaw_trajs/traj_glm_oc_0719_1228/" \
    output --obsutil ~/bin/obsutil

  python download_workspace_and_run.py \
    "obs://rl-agentdata/openclaw_trajs/traj_glm_hermes_0719_1525/" \
    output --obsutil ~/bin/obsutil --hermes --concurrency 16
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

# Hermes 专用模式（assistant 侧: assistant* 或 main，对应 _is_assistant_agent_dir()）。
# 与通用模式的区别:
#   - 去掉 openclaw 的 .jsonl 格式(*assistant*sessions*.jsonl 等)
#   - 保留 profiles/assistant* 和 profiles/main 的 sessions
#   - 保留 profiles/assistant*/state.db*（token 用量）
#   - evaluator sessions 不走 bulk，由详情页按需懒加载(_ensure_hermes_evaluator)
# 主 log(logs/<task>.log): Hermes 也有, 分数走 query1.json, 但「【Task_Done】」标记只在主 log
#   正文里(check_task_done_in_logs_dir 扫 <task>.log), 故必须下载, 否则 task_done_count 恒为 0。
#   顺带让详情页「任务 Log」标签秒开(本地已有, 不必按需拉取)。
HERMES_INCLUDE = [
    "*logs/trajectories/*query*.json",            # 分数 + 裁决 + 聚合轨迹
    "*logs*.log",                                  # 主 log(<task>.log): TASK_DONE 标记 + 详情页任务 Log
    "*profiles/assistant*/sessions/*.json",        # assistant sessions
    "*profiles/main/sessions/*.json",              # main sessions（备选 agent 名）
    "*profiles/assistant*/state.db*",              # token 用量 DB
]
# 宽泛的 *logs*.log include 会连带匹配噪音日志, 逐条 exclude 挡掉(统计/详情都不读, 与通用模式一致):
#   - *_use.log           : api_use.log / evaluator_use.log
#   - *profiles/*/logs/*  : profiles 内 agent.log/errors.log
#   - *_logs/*.log        : npm 调试日志
# Hermes 没有 *.trajectory.jsonl，保留该 exclude 仅作安全冗余
HERMES_EXCLUDE = ["*.trajectory.jsonl", "*_use.log", "*profiles/*/logs/*", "*_logs/*.log"]


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


def list_task_dirs(obsutil, workspace_obs, obs_cred_args=None):
    """用 obsutil ls -d 枚举 workspace 下的直接子目录(task), 处理 Next marker 翻页。

    obs_cred_args: 可选 ["-i", ak, "-k", sk, "-e", endpoint], 用于覆盖 obsutil 全局默认凭证
    (访问另一个账号/桶时用, 不传则走全局默认)。
    返回 task 的 obs URL 列表(每个以 / 结尾)。
    """
    workspace_obs = workspace_obs if workspace_obs.endswith("/") else workspace_obs + "/"
    tasks = []
    marker = None
    while True:
        cmd = [obsutil, "ls", workspace_obs, "-d", "-limit", "1000"] + (obs_cred_args or [])
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


def download_task(obsutil, task_obs, dest_dir,
                  include_patterns=None, exclude_patterns=None, obs_cred_args=None):
    """按需下载单个 task 的必要文件到 dest_dir/<leaf>/ (include/exclude 过滤)。

    include_patterns/exclude_patterns: 若为 None, 使用模块级别的 INCLUDE_PATTERNS/EXCLUDE_PATTERNS。
    obs_cred_args: 可选 ["-i", ak, "-k", sk, "-e", endpoint], 覆盖 obsutil 全局默认凭证。
    """
    if include_patterns is None:
        include_patterns = INCLUDE_PATTERNS
    if exclude_patterns is None:
        exclude_patterns = EXCLUDE_PATTERNS

    os.makedirs(dest_dir, exist_ok=True)
    task_obs = task_obs if task_obs.endswith("/") else task_obs + "/"
    cmd = [obsutil, "cp", task_obs, dest_dir, "-r", "-f"] + (obs_cred_args or [])
    for p in include_patterns:
        cmd += ["-include", p]
    for p in exclude_patterns:
        cmd += ["-exclude", p]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"obsutil cp 失败(退出码 {res.returncode}) {task_obs}: "
                           f"{(res.stdout or '')[-400:]}")


def harness_tsr_to_entries(d):
    """把单个 task 的采集侧 logs/traj_stats_result.json 映射成 traj_stats 内部 per_task entry。

    采集 harness 在每个 task 目录下产出 <task>/logs/traj_stats_result.json, 结构是单 task 对象:
      {task, config_file, log_file, harness_home, task_level, best_completion,
       agents:[{agent, trajectory, tool_calls, assistant_rounds, plain_rounds,
                has_ge3_toolcalls, has_plain_round, has_eval, evaluator_completion,
                verdict_source, level}]}
    注意该文件**不含 char_len / token** 字段(那两个要读轨迹正文才算), 故这里也不产出这些键,
    使 stats_from_per_task 里 char_len_stats/token_stats 归 None(快速路径两列先留空, 详情按需回填)。

    对 agents 里每个 agent 产出一条 entry; 无法解析(缺 agents)时返回 []。
    """
    task = d.get("task")
    agents = d.get("agents")
    if not task or not isinstance(agents, list):
        return []
    harness_home = d.get("harness_home") or ""
    harness = "openclaw" if ".openclaw" in harness_home else "hermes"
    entries = []
    for a in agents:
        if not isinstance(a, dict):
            continue
        tc = a.get("tool_calls") or 0
        pr = a.get("plain_rounds") or 0
        has_ge3 = bool(a.get("has_ge3_toolcalls"))
        has_plain = bool(a.get("has_plain_round"))
        # L1 门槛: openclaw = ≥3工具调用 且 有纯轮; hermes = 有产出(纯轮>0)。与 process_root 口径一致。
        passed = (has_ge3 and has_plain) if harness == "openclaw" else has_plain
        # trajectory 绝对容器路径(如 <harness_home>/agents/main/sessions/<uuid>.jsonl)
        # → 映射成 origin 相对路径 <task>/agents/main/sessions/<uuid>.jsonl, 供详情页按需下载定位。
        traj_abs = a.get("trajectory") or ""
        try:
            rel = os.path.relpath(traj_abs, harness_home) if (traj_abs and harness_home) else traj_abs
        except ValueError:
            rel = traj_abs
        traj_rel = os.path.join(task, rel) if rel else None
        entries.append({
            "task": task,
            "trajectory": traj_rel,
            "tool_calls": tc,
            "assistant_rounds": a.get("assistant_rounds") or 0,
            "plain_rounds": pr,
            "has_ge3_toolcalls": has_ge3,
            "has_plain_round": has_plain,
            "passed_gate": passed,
            "has_eval": bool(a.get("has_eval")),
            "evaluator_completion": a.get("evaluator_completion"),
            "verdict_source": a.get("verdict_source"),
            "harness": harness,
            # 不写 char_len / total_tokens: 快速路径两列留空, 详情按需回填
        })
    return entries


def _fetch_task_done_marker(obsutil, task_obs, origin, leaf, log_file,
                            obs_cred_args=None):
    """快速路径补齐 TASK_DONE: traj_stats_result.json 不含 task_done 字段,
    该标记只在主 .log 正文里, 故这里额外下载主 log 并扫「【Task_Done】」标记。

    log_file 来自 traj_stats_result.json 的 log_file 字段(相对 task 目录, 如 logs/xxx.log);
    缺失时回退按 check_task_done_in_logs_dir 的候选名(<leaf>.log / harness_automation.log)逐个尝试。
    命中任一即 True; 全部下载失败/无标记返回 False。
    """
    candidates = []
    if log_file:
        candidates.append(log_file.replace("\\", "/").lstrip("/"))
    for name in (leaf + ".log", "harness_automation.log"):
        rel = "logs/" + name
        if rel not in candidates:
            candidates.append(rel)
    for rel in candidates:
        dest = os.path.join(origin, leaf, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        obj_url = task_obs + rel
        cmd = [obsutil, "cp", obj_url, dest, "-f"] + (obs_cred_args or [])
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=300)
        except Exception:
            continue
        if res.returncode != 0 or not os.path.isfile(dest):
            continue                  # 该候选名不存在, 试下一个
        # 成功下到主 log 即为定论: 一个 task 只有一份主 log, 有没有标记看这份即可,
        # 不再试其它候选名(避免绝大多数「未完成」任务白白多下 1~2 个 log 文件)。
        return traj_stats.has_task_done_marker(dest)
    return False


def _fetch_one_task_stats(obsutil, task_obs, origin, obs_cred_args=None,
                          with_task_done=True):
    """下载单个 task 的 logs/traj_stats_result.json 到 origin/<leaf>/logs/traj_stats_result.json,
    解析并经 harness_tsr_to_entries 展平。文件不存在/失败返回 []。

    with_task_done=True 时额外下载主 log 扫「【Task_Done】」标记, 回填每条 entry 的 task_done
    (traj_stats_result.json 本身不含该字段)。"""
    task_obs = task_obs if task_obs.endswith("/") else task_obs + "/"
    leaf = task_obs.rstrip("/").split("/")[-1]
    dest = os.path.join(origin, leaf, "logs", "traj_stats_result.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    obj_url = task_obs + "logs/traj_stats_result.json"
    cmd = [obsutil, "cp", obj_url, dest, "-f"] + (obs_cred_args or [])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
    except Exception:
        return []
    if res.returncode != 0 or not os.path.isfile(dest):
        return []                     # 该 task 无此文件(老 workspace), 正常, 不刷错误
    try:
        with open(dest, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    entries = harness_tsr_to_entries(data)
    if entries and with_task_done:
        done = _fetch_task_done_marker(obsutil, task_obs, origin, leaf,
                                       data.get("log_file"), obs_cred_args)
        for e in entries:
            e["task_done"] = done
    return entries


def fetch_per_task_stats_files(obsutil, workspace_obs, origin, obs_cred_args=None,
                               concurrency=8, with_task_done=True):
    """快速路径: 枚举 workspace 下各 task, 并发下载每个 <task>/logs/traj_stats_result.json
    (每份约 1KB), 展平成 per_task 列表返回。

    with_task_done=True 时每个 task 额外下载主 log 扫「【Task_Done】」标记(带宽略增, 但仍远小于
    整包轨迹); False 则跳过, TASK_DONE 计数恒为 0(与旧 fast 行为一致)。

    一个都没拿到(老 workspace 或无此文件)返回 None, 调用方回退逐 task 全量下载老逻辑。
    """
    tasks = list_task_dirs(obsutil, workspace_obs, obs_cred_args=obs_cred_args)
    if not tasks:
        return None
    total = len(tasks)
    step = "下 stats + 扫主 log(TASK_DONE)" if with_task_done else "下 stats"
    print(f"      [fast] 共 {total} 个 task, 并发 {concurrency} {step}, 逐个上报进度 ...",
          flush=True)
    per_task = []
    got = 0
    done_n = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(_fetch_one_task_stats, obsutil, t, origin, obs_cred_args,
                          with_task_done): t
                for t in tasks}
        for fut in as_completed(futs):
            done_n += 1
            entries = fut.result()
            if entries:
                got += 1
                per_task.extend(entries)
            # 逐 task 上报: 每完成一个都打一行, 让前端/终端能看到进度, 不再长时间静默。
            # 附实时速率(task/s, 按已用时均值) + 并发数 + 估算剩余时间(ETA)。
            if done_n % 5 == 0 or done_n == total:
                elapsed = time.time() - t_start
                rate = done_n / elapsed if elapsed > 0 else 0.0
                eta = (total - done_n) / rate if rate > 0 else 0.0
                print(f"      [fast] 进度 {done_n}/{total} (命中 {got}) | "
                      f"{rate:.1f} task/s @ 并发 {concurrency} | ETA {eta/60:.1f} min",
                      flush=True)
    if got == 0:
        return None
    print(f"      [fast] {got}/{total} 个 task 命中 logs/traj_stats_result.json", flush=True)
    return per_task


def _avg_tokens(entries, tier_key, tier_entries):
    """计算某一档次的平均 token 总长度，同时返回原始 sum + count 以便跨任务聚合。

    entries: 完整的 per_task 原始行列表
    tier_entries: 该档次在 entries 中的下标列表
    返回 {avg_total_tokens, sum_total, count}，
    若该档次无 token 数据则返回 None。
    """
    sum_total = 0
    count = 0
    for idx in tier_entries:
        row = entries[idx]
        total = row.get("total_tokens")
        if total is None:
            continue
        sum_total += total
        count += 1
    if count == 0:
        return None
    return {
        "avg_total_tokens": round(sum_total / count),
        "sum_total": sum_total,
        "count": count,
    }


def _avg_char_len(entries, tier_entries):
    """计算某一档次的平均轨迹字符数，同 _avg_tokens 结构，供跨任务聚合。"""
    sum_total = 0
    count = 0
    for idx in tier_entries:
        row = entries[idx]
        total = row.get("char_len")
        if total is None:
            continue
        sum_total += total
        count += 1
    if count == 0:
        return None
    return {
        "avg_char_len": round(sum_total / count),
        "sum_total": sum_total,
        "count": count,
    }


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
    return stats_from_per_task(per_task, source_type="workspace")


def stats_from_per_task(per_task, source_type="workspace"):
    """把 traj_stats 的 per_task(process_root 的返回, 或 traj_stats_result.json 的 details)
    按 workspace 门槛口径聚合成 filter_stats.json 结构。

    抽出此函数是为了让「快速路径」(直接吃 workspace 根目录已有的 traj_stats_result.json.details)
    与「老路径」(本地 process_root)共用同一套 tier 聚合逻辑, 保证两条路口径一致。
    source_type 仅用于 note/source_type 标注, 不影响聚合。
    """
    per_session = []
    kept = with_eval = ge05 = eq1 = 0
    dropped = 0
    task_done_count = 0
    # 记录每个 tier 在 per_task 中的下标（用于 token 统计）
    l0_idx, l1_idx, l15_idx, l2_idx, l3_idx, td_idx = [], [], [], [], [], []
    for i, row in enumerate(per_task):
        comp = row.get("evaluator_completion")
        has_score = isinstance(comp, (int, float))         # L1.5: 有首轮数值分(不含 null)
        # L1 门槛: 直接读 traj_stats 算好的 passed_gate(openclaw/Hermes 同式: ≥3工具调用+纯轮),
        # 不在此重算, 保证两套 harness 口径统一。
        passed = bool(row.get("passed_gate"))
        task_done = bool(row.get("task_done"))
        if task_done:
            task_done_count += 1
            td_idx.append(i)
        if passed:
            kept += 1
            l1_idx.append(i)
            if has_score:                                  # L1.5: 门槛内且有数值分
                with_eval += 1
                l15_idx.append(i)
            if has_score and comp >= 0.5:                  # L2
                ge05 += 1
                l2_idx.append(i)
            if has_score and comp == 1:                    # L3
                eq1 += 1
                l3_idx.append(i)
        else:
            dropped += 1
        l0_idx.append(i)                                   # L0: 所有 task
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
            "task_done": task_done,           # log 是否含「【Task_Done】」标记
        })
        # 透传 Hermes token 数据到 per_session（供 api 按任务查询）
        if "total_tokens" in row:
            per_session[-1]["input_tokens"] = row["input_tokens"]
            per_session[-1]["output_tokens"] = row["output_tokens"]
            per_session[-1]["reasoning_tokens"] = row["reasoning_tokens"]
            per_session[-1]["total_tokens"] = row["total_tokens"]
        if "char_len" in row:
            per_session[-1]["char_len"] = row["char_len"]

    token_stats = {}
    char_len_stats = {}
    for tier, label in [("L0", l0_idx), ("L1", l1_idx), ("T_DONE", td_idx),
                        ("L1.5", l15_idx), ("L2", l2_idx), ("L3", l3_idx)]:
        avg = _avg_tokens(per_task, tier, label)
        if avg is not None:
            token_stats[tier] = avg
        char_avg = _avg_char_len(per_task, label)
        if char_avg is not None:
            char_len_stats[tier] = char_avg

    return {
        "filtered_count":    kept,            # L1
        "with_eval_count":   with_eval,       # L1.5
        "completion_ge_0.5": ge05,            # L2
        "completion_eq_1":   eq1,             # L3
        "dropped_count":     dropped,         # 未过 L1 门槛的轨迹(L0 - L1)
        "task_done_count":   task_done_count, # 主 log 含「【Task_Done】」标记的 task 数(Hermes 未下载主 log 时恒为 0)
        "note": ("来源=workspace(原始轨迹按需下载); L0=总轨迹数, "
                 "L1=ge3_and_plain_round(≥3工具调用+有纯轮), "
                 "L1.5=L1内有 turn=1 数值 completion(不含 null), "
                 "L2/L3=L1.5内 turn=1 completion>=0.5 / ==1"),
        "source_type": source_type,
        "per_session": per_session,
        "token_stats": token_stats if token_stats else None,
        "char_len_stats": char_len_stats if char_len_stats else None,
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
    ap.add_argument("--hermes", action="store_true",
                    help="Hermes 模式: 使用 Hermes 专用 include/exclude 模式(去掉 openclaw 专用文件与无用日志)")
    ap.add_argument("--obs-ak", default=None, help="OBS Access Key ID(可选; 三个 --obs-* 参数要么都给要么都不给, "
                    "用于覆盖 obsutil 全局默认凭证, 访问另一个账号/桶时用)")
    ap.add_argument("--obs-sk", default=None, help="OBS Secret Access Key(可选, 见 --obs-ak)")
    ap.add_argument("--obs-endpoint", default=None, help="OBS endpoint(可选, 如 obs.cn-east-4.myhuaweicloud.com, 见 --obs-ak)")
    ap.add_argument("--no-fast", action="store_true",
                    help="禁用快速路径(即使 workspace 根目录有 traj_stats_result.json 也强制逐 task 下载重算)")
    ap.add_argument("--fast-no-task-done", action="store_true",
                    help="快速路径下不额外下载主 log 扫「【Task_Done】」标记(更快, 但 TASK_DONE 计数恒为 0)")
    a = ap.parse_args()

    if not os.path.exists(a.obsutil):
        ap.error(f"obsutil 不存在: {a.obsutil}")

    obs_cred_vals = (a.obs_ak, a.obs_sk, a.obs_endpoint)
    if any(obs_cred_vals) and not all(obs_cred_vals):
        ap.error("--obs-ak / --obs-sk / --obs-endpoint 要么都给, 要么都不给")
    obs_cred_args = ["-i", a.obs_ak, "-k", a.obs_sk, "-e", a.obs_endpoint] if all(obs_cred_vals) else []

    origin = os.path.join(a.out_dir, "origin")
    os.makedirs(origin, exist_ok=True)

    # ── 快速路径: 各 task 目录下已有采集侧预生成的 logs/traj_stats_result.json ──
    #   只下这些每份 ~1KB 的小文件, 展平成 per_task 出 filter_stats.json, 不再逐 task 下载
    #   原始轨迹(具体 assistant 轨迹改由详情页按需懒加载)。char_len/token 两列先留空, 详情按需回填。
    #   一个都没有(老 workspace)则回退下面的逐 task 全量下载老逻辑。
    if not a.no_fast:
        _td = " (含主 log TASK_DONE 扫描)" if not a.fast_no_task_done else ""
        print(f"[fast] 探测各 task 的 logs/traj_stats_result.json{_td} ...", flush=True)
        details = fetch_per_task_stats_files(a.obsutil, a.workspace_obs, origin,
                                             obs_cred_args=obs_cred_args,
                                             concurrency=a.concurrency,
                                             with_task_done=not a.fast_no_task_done)
        if details is not None:
            print(f"      命中! 复用采集侧统计({len(details)} 条轨迹), 跳过逐 task 全量下载", flush=True)
            stats = stats_from_per_task(details, source_type="workspace")
            stats["fast_path"] = True     # 标记: 本次走快速路径(原始轨迹未整包下载, 详情按需拉)
            out_path = os.path.join(a.out_dir, "filter_stats.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            total_traj = stats["filtered_count"] + stats["dropped_count"]
            print(f"[done] (快速路径) 结果 -> {out_path}", flush=True)
            print(f"      总轨迹数(L0)               : {total_traj}", flush=True)
            print(f"      合格轨迹 ≥3工具+纯轮(L1)   : {stats['filtered_count']}", flush=True)
            print(f"      L1内有 evaluator 裁决(L1.5): {stats['with_eval_count']}", flush=True)
            print(f"      completion >= 0.5 (L2)    : {stats['completion_ge_0.5']}", flush=True)
            print(f"      completion == 1  (L3)     : {stats['completion_eq_1']}", flush=True)
            print(f"      含「【Task_Done】」标记        : {stats['task_done_count']}", flush=True)
            return
        print(f"      未命中(老 workspace 或无此文件), 回退逐 task 下载", flush=True)

    print(f"[1] 枚举 task 子目录 <- {a.workspace_obs}", flush=True)
    tasks = list_task_dirs(a.obsutil, a.workspace_obs, obs_cred_args=obs_cred_args)
    if a.max_tasks and a.max_tasks > 0:
        tasks = tasks[:a.max_tasks]
    total = len(tasks)
    print(f"      共 {total} 个 task", flush=True)
    if total == 0:
        sys.exit(f"[error] workspace 下未枚举到任何 task 子目录: {a.workspace_obs}")

    concurrency = max(1, a.concurrency)
    include_pats = HERMES_INCLUDE if a.hermes else INCLUDE_PATTERNS
    exclude_pats = HERMES_EXCLUDE if a.hermes else EXCLUDE_PATTERNS
    mode_label = "Hermes" if a.hermes else "通用"
    print(f"[2] 按需下载(模式={mode_label}; 每 task 仅必要文件; 并发 {concurrency})", flush=True)
    t0 = time.time()
    failed = []
    done = 0
    lock = threading.Lock()
    last_report = [0.0]   # 用列表以便闭包内可改

    def worker(task_obs):
        """下载单个 task; 返回 (leaf, error_or_None)。异常不外抛, 单 task 失败不拖垮整批。"""
        leaf = obs_leaf(task_obs)
        try:
            download_task(a.obsutil, task_obs, origin,
                          include_patterns=include_pats, exclude_patterns=exclude_pats,
                          obs_cred_args=obs_cred_args)
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
    print(f"      含「【Task_Done】」标记        : {stats['task_done_count']}", flush=True)


if __name__ == "__main__":
    main()
