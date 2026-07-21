# -*- coding: utf-8 -*-
"""
轨迹统计脚本。

对给定根目录（默认 E:\\轨迹\\0716）下的每个任务子目录进行统计：

  - assistant 轨迹：<task>/agents/assistant*/sessions/ 下文件名不含 "trajectory" 的 .jsonl 文件
  - 任务记录：<task>/logs/<task>.log，从中提取第一轮 evaluator 的 "completion" 分数

统计指标（逐层递进）：
  1. total_tasks                         扫描到的任务子目录数
  2. assistant_trajectories             找到的 assistant 轨迹数
  3. ge3_toolcalls                      至少有三次工具调用的 assistant 轨迹数
  4. ge3_and_plain_round                在满足(3)的前提下，输出过不带工具调用的
                                        assistant 轮（只有 reasoning / content）的轨迹数
  5. with_evaluator_score               满足(4)且有 evaluator 第一轮打分的轨迹数
  6. score_ge_0_5                       满足(5)且打分 >= 0.5 的轨迹数
  7. score_eq_1                         满足(5)且打分 == 1 的轨迹数

用法:
    python traj_stats.py [根目录] [-o 输出json路径]
"""

import argparse
import json
import os
import re
import sys


def _is_assistant_agent_dir(name):
    """assistant 侧的 agent 目录名: 有的 harness 叫 assistant1, 有的叫 main。
    排除 evaluator(评测侧)。"""
    return (name.startswith("assistant") or name == "main") and name != "evaluator"


def find_assistant_trajectories(task_dir):
    """返回该任务目录下的 assistant 轨迹文件路径（文件名不含 trajectory 的 .jsonl）。

    若同一任务存在多个 assistant 轨迹，只取最后一个（按文件修改时间最新者）。
    assistant 侧目录名可能是 assistant1 或 main（不同 harness 变体）。
    """
    candidates = []
    agents_dir = os.path.join(task_dir, "agents")
    if not os.path.isdir(agents_dir):
        return []
    for name in sorted(os.listdir(agents_dir)):
        if not _is_assistant_agent_dir(name):
            continue
        sessions_dir = os.path.join(agents_dir, name, "sessions")
        if not os.path.isdir(sessions_dir):
            continue
        for fn in sorted(os.listdir(sessions_dir)):
            if not fn.endswith(".jsonl"):
                continue
            if "trajectory" in fn:
                continue
            candidates.append(os.path.join(sessions_dir, fn))

    if not candidates:
        return []
    # 多个轨迹时只取最后一个：以文件大小最大者为准
    latest = max(candidates, key=lambda p: os.path.getsize(p))
    return [latest]


def analyze_trajectory(path):
    """分析一条 assistant 轨迹。

    返回 dict: {tool_calls, plain_rounds, assistant_rounds}
      tool_calls       : 全轨迹中 toolCall 的总次数
      plain_rounds     : 不带工具调用的 assistant 轮数（只有 thinking / text）
      assistant_rounds : assistant 消息轮数
    """
    tool_calls = 0
    plain_rounds = 0
    assistant_rounds = 0

    with open(path, encoding="utf-8", errors="replace") as f:
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

            assistant_rounds += 1
            content = msg.get("content")
            # content 可能是字符串（纯文本，无工具调用）或部件列表
            if isinstance(content, str):
                parts_types = ["text"] if content else []
            elif isinstance(content, list):
                parts_types = [p.get("type") for p in content if isinstance(p, dict)]
            else:
                parts_types = []

            n_tc = parts_types.count("toolCall")
            tool_calls += n_tc
            if n_tc == 0:
                # 只有 reasoning(thinking) 和 content(text)，没有 toolCall
                plain_rounds += 1

    return {
        "tool_calls": tool_calls,
        "plain_rounds": plain_rounds,
        "assistant_rounds": assistant_rounds,
    }


# evaluator 输出标记, 捕获 turn 编号。注意: 有些任务的评测并非从 turn=1 开始
# (前面的 turn 可能未触发 evaluator 或被 reset), 首轮可能是 turn=2/3。
_EVAL_MARKER = re.compile(r"\[Evaluator\]\s+turn=(\d+)\s+agent=\S+.*输出")


def _parse_json_block_after(lines, idx):
    """从 lines[idx] 之后第一个以 '{' 开头的行起, 用大括号计数取出完整 JSON 块并解析。
    返回 dict; 找不到块返回 None; 块内 JSON 非法返回字符串标记 '__BADJSON__'。"""
    j = idx + 1
    while j < len(lines) and lines[j].strip() != "{":
        j += 1
    if j >= len(lines):
        return None
    depth = 0
    buf = []
    for k in range(j, len(lines)):
        buf.append(lines[k])
        depth += lines[k].count("{") - lines[k].count("}")
        if depth <= 0:
            break
    try:
        return json.loads("".join(buf))
    except json.JSONDecodeError:
        return "__BADJSON__"


def extract_first_evaluator_obj(log_path):
    """从任务 log 中提取「首轮」evaluator 裁决的完整 dict。

    首轮 = 编号最小的 [Evaluator] turn=N 块(而非死守 turn=1), 以正确覆盖评测从
    turn=2/3 才开始的任务。找不到任何裁决块返回 None; 块存在但 JSON 非法返回
    字符串 '__BADJSON__'(调用方据此判断「有裁决但无法解析分数」)。
    """
    if not os.path.isfile(log_path):
        return None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    marks = []
    for i, l in enumerate(lines):
        m = _EVAL_MARKER.search(l)
        if m:
            marks.append((int(m.group(1)), i))
    if not marks:
        return None
    marks.sort(key=lambda x: (x[0], x[1]))   # 先按 turn 号, 再按出现行号
    _, idx = marks[0]
    return _parse_json_block_after(lines, idx)


def extract_first_evaluator_verdict(log_path):
    """返回 (has_verdict, completion)：
      has_verdict : 是否存在 evaluator 裁决块(哪怕 completion 为 null / JSON 非法)
      completion  : float 分数; 块内无数值(或为 null / 非法)时为 None
    """
    obj = extract_first_evaluator_obj(log_path)
    if obj is None:
        return False, None
    if obj == "__BADJSON__":
        return True, None
    comp = obj.get("completion")
    if isinstance(comp, (int, float)):
        return True, float(comp)
    return True, None


def extract_first_evaluator_completion(log_path):
    """兼容旧接口：只返回首轮 evaluator 的 completion 分数（无则 None）。"""
    _, comp = extract_first_evaluator_verdict(log_path)
    return comp


def find_evaluator_trajectory(task_dir):
    """该任务 evaluator 的非 trajectory session jsonl。

    有些任务的评测裁决没有回写进主 log(log 里 evals=0), 只落在 evaluator 会话轨迹里,
    此时需要从这里兜底解析裁决。
    """
    sessions = os.path.join(task_dir, "agents", "evaluator", "sessions")
    if not os.path.isdir(sessions):
        return None
    cands = [os.path.join(sessions, fn) for fn in sorted(os.listdir(sessions))
             if fn.endswith(".jsonl") and "trajectory" not in fn]
    if not cands:
        return None
    # 多个时取最大者(与 assistant 侧一致)
    return max(cands, key=lambda p: os.path.getsize(p))


_EVAL_COMP_RE = re.compile(r'"completion"\s*:\s*(null|-?[0-9.]+)')


def extract_eval_traj_verdict(jsonl_path):
    """从 evaluator 轨迹 jsonl 兜底解析首轮裁决(log 无裁决时用)。

    取最后一条同时含 rubric_checks 与 inclination 的 assistant 消息(即裁决原文),
    用正则抓其中的 completion(裁决 JSON 常跨多行且含非法转义, 整块 json.loads 不可靠,
    故用正则; 提示词里的 completion 出现在 user 消息, 已被「只看 assistant 消息」排除)。
    返回 (has_verdict, completion)。
    """
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return False, None
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
    if last is None:
        return False, None
    ms = _EVAL_COMP_RE.findall(last)
    if not ms:
        return True, None
    v = ms[-1]
    return True, (None if v == "null" else float(v))


# ── Hermes harness: 轨迹在 profiles/*/sessions/*.json 或 logs/trajectories/*/query1.json ──
#   探测: task 无 assistant sessions jsonl, 但有以下之一:
#     - profiles/*/sessions/session_*.json  (完整 event-stream, 优先使用)
#     - logs/trajectories/*/query1.json     (聚合 turns[], 兜底)
#   profiles session 结构: {messages: [{role, content}, ...], model, ...} 标准 event-stream
#   query1.json 结构: {turns: [{user_input, agent_content, tool_calls[], ...}], evaluations: [...]}

def find_hermes_sessions(task_dir):
    """返回 Hermes 的 profiles/*/sessions/session_*.json 路径列表。
    只统计 assistant 侧的轨迹（profiles/assistant*/sessions 或 profiles/main/sessions），
    排除 evaluator 侧（evaluator 是评测轨迹，不是被评测的轨迹）。
    多个时返回全部（一个 task 可能有多次重试），按文件名排序。"""
    import glob
    cands = []
    profiles_dir = os.path.join(task_dir, "profiles")
    if not os.path.isdir(profiles_dir):
        return []
    for agent_name in os.listdir(profiles_dir):
        # 只取 assistant 侧：assistant1 / assistant2 / main，排除 evaluator
        if not _is_assistant_agent_dir(agent_name):
            continue
        pattern = os.path.join(profiles_dir, agent_name, "sessions", "session_*.json")
        cands.extend(glob.glob(pattern))
    return sorted(cands) if cands else []


def analyze_hermes_session(path):
    """分析 Hermes profiles session json (标准 event-stream)。

    返回 dict: {tool_calls, plain_rounds, assistant_rounds}
      tool_calls       : assistant 消息中 toolCall 部件的总数
      plain_rounds     : 不带 toolCall 的 assistant 消息数（纯文本/thinking）
      assistant_rounds : assistant 消息总数
    """
    tool_calls = 0
    plain_rounds = 0
    assistant_rounds = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"tool_calls": 0, "plain_rounds": 0, "assistant_rounds": 0}

    for msg in (data.get("messages") or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        assistant_rounds += 1
        content = msg.get("content")
        # content 可能是字符串（纯文本）或部件列表
        if isinstance(content, str):
            parts_types = ["text"] if content else []
        elif isinstance(content, list):
            parts_types = [p.get("type") for p in content if isinstance(p, dict)]
        else:
            parts_types = []

        n_tc = parts_types.count("toolCall")
        tool_calls += n_tc
        if n_tc == 0:
            # 只有 text/thinking，没有 toolCall
            plain_rounds += 1

    return {
        "tool_calls": tool_calls,
        "plain_rounds": plain_rounds,
        "assistant_rounds": assistant_rounds,
    }


def find_query1_json(task_dir):
    """Hermes: 返回该任务的 query1.json 路径(logs/trajectories/<ts>/query1.json)。
    多个时间戳目录时取最大者(与 assistant 侧「取最新」口径一致)。无则返回 None。"""
    import glob
    cands = glob.glob(os.path.join(task_dir, "logs", "trajectories", "*", "query*.json"))
    if not cands:
        return None
    return max(cands, key=lambda p: os.path.getsize(p))


def analyze_query1(path):
    """分析 Hermes query1.json 的 turns[]。

    返回 dict: {tool_calls, plain_rounds, assistant_rounds}
      tool_calls       : 所有 turn 的 tool_calls 总数
      plain_rounds     : 「有实质产出」的 turn 数 —— agent_content 非空
                         (Hermes 的 turn 粒度比 openclaw event-stream 粗, 一个 turn 可能既有
                         工具调用又有 agent_content 收尾; 只要产出了 agent_content 就算有效,
                         不要求 tool_calls 必须为空, 以适配「边搜边答」模式)
      assistant_rounds : turn 总数
    """
    tool_calls = 0
    plain_rounds = 0
    turns_n = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"tool_calls": 0, "plain_rounds": 0, "assistant_rounds": 0}

    for t in (data.get("turns") or []):
        if not isinstance(t, dict):
            continue
        turns_n += 1
        tcs = t.get("tool_calls") or []
        n_tc = len(tcs) if isinstance(tcs, list) else 0
        tool_calls += n_tc
        # 只要有 agent_content 就算有产出 (不再要求 tool_calls 必须为空)
        if (t.get("agent_content") or "").strip():
            plain_rounds += 1

    return {
        "tool_calls": tool_calls,
        "plain_rounds": plain_rounds,
        "assistant_rounds": turns_n,
    }


def extract_query1_verdict(path):
    """从 query1.json.evaluations[] 取「首轮」裁决。

    首轮 = turn 号最小的那条 evaluation(评测可能从 turn=2/3 才开始, 与 openclaw 同理,
    故不写死 evaluations[0])。返回 (has_eval, completion):
      has_eval   : 是否存在裁决块(哪怕 completion 为 null)
      completion : float 分数; 无数值(或 null)时为 None
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, None

    evals = [e for e in (data.get("evaluations") or []) if isinstance(e, dict)]
    if not evals:
        return False, None
    # 按 turn 号取最小; 缺 turn 字段的排到最后, 保持稳定
    first = min(evals, key=lambda e: e.get("turn", float("inf")))
    comp = first.get("completion")
    if isinstance(comp, (int, float)):
        return True, float(comp)
    return True, None


def resolve_first_verdict(task_dir, task):
    """统一取「首轮裁决」: 先读 log(编号最小 turn), log 无数值分时回退 evaluator 轨迹。

    返回 (has_eval, completion, source):
      source ∈ {"log", "eval_traj", None}
    """
    log_path = os.path.join(task_dir, "logs", task + ".log")
    has_eval, score = extract_first_evaluator_verdict(log_path)
    source = "log" if has_eval else None
    if score is None:                      # log 无数值分 → 回退 evaluator 轨迹
        ev = find_evaluator_trajectory(task_dir)
        if ev:
            ev_has, ev_score = extract_eval_traj_verdict(ev)
            if ev_score is not None:
                return True, ev_score, "eval_traj"
            if ev_has and not has_eval:
                return True, None, "eval_traj"
    return has_eval, score, source


def process_root(root):
    per_task = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        task = entry.name
        task_dir = entry.path

        # 格式探测: 有 assistant sessions jsonl -> openclaw; 否则有 query1.json -> Hermes
        traj_paths = find_assistant_trajectories(task_dir)
        if traj_paths:
            has_eval, score, verdict_source = resolve_first_verdict(task_dir, task)
            for tp in traj_paths:
                info = analyze_trajectory(tp)
                gate = info["tool_calls"] >= 3 and info["plain_rounds"] > 0
                per_task.append({
                    "task": task,
                    "trajectory": os.path.relpath(tp, root),
                    "tool_calls": info["tool_calls"],
                    "assistant_rounds": info["assistant_rounds"],
                    "plain_rounds": info["plain_rounds"],
                    "has_ge3_toolcalls": info["tool_calls"] >= 3,
                    "has_plain_round": info["plain_rounds"] > 0,
                    "passed_gate": gate,           # L1 门槛(≥3工具调用+纯轮)
                    "has_eval": has_eval,
                    "evaluator_completion": score,
                    "verdict_source": verdict_source,
                    "harness": "openclaw",
                })
            continue

        # Hermes: 优先用 profiles/*/sessions/*.json (完整轨迹), 回退 query1.json (聚合视图)
        hermes_sessions = find_hermes_sessions(task_dir)
        if hermes_sessions:
            # 有 profiles sessions: 每个 session 独立统计（一个 task 可能有多次重试）
            for session_path in hermes_sessions:
                info = analyze_hermes_session(session_path)
                has_eval, score = extract_query1_verdict(find_query1_json(task_dir) or "")
                # Hermes L1 门槛: 有产出即可(plain_rounds > 0)
                gate = info["plain_rounds"] > 0
                per_task.append({
                    "task": task,
                    "trajectory": os.path.relpath(session_path, root),
                    "tool_calls": info["tool_calls"],
                    "assistant_rounds": info["assistant_rounds"],
                    "plain_rounds": info["plain_rounds"],
                    "has_ge3_toolcalls": info["tool_calls"] >= 3,
                    "has_plain_round": info["plain_rounds"] > 0,
                    "passed_gate": gate,
                    "has_eval": has_eval,
                    "evaluator_completion": score,
                    "verdict_source": "query1" if has_eval else None,
                    "harness": "hermes",
                })
            continue

        # Hermes 回退: 只有 query1.json，没有 profiles sessions
        query1 = find_query1_json(task_dir)
        if not query1:
            continue                              # 两种格式都没有, 跳过该目录
        info = analyze_query1(query1)
        has_eval, score = extract_query1_verdict(query1)
        # Hermes L1 门槛: 有产出即可(plain_rounds > 0), 不强制要求 ≥3 工具调用
        # (因 Hermes turn 粒度粗, 很多单轮对话任务 tool_calls=0 但有完整答复)
        gate = info["plain_rounds"] > 0
        per_task.append({
            "task": task,
            "trajectory": os.path.relpath(query1, root),
            "tool_calls": info["tool_calls"],
            "assistant_rounds": info["assistant_rounds"],
            "plain_rounds": info["plain_rounds"],
            "has_ge3_toolcalls": info["tool_calls"] >= 3,
            "has_plain_round": info["plain_rounds"] > 0,
            "passed_gate": gate,                  # L1 门槛: Hermes 只看有无产出, 不看工具调用数
            "has_eval": has_eval,
            "evaluator_completion": score,
            "verdict_source": "query1" if has_eval else None,
            "harness": "hermes",
        })
    return per_task


def summarize(per_task):
    total_tasks = len({t["task"] for t in per_task})

    ge3 = [t for t in per_task if t["has_ge3_toolcalls"]]
    ge3_plain = [t for t in ge3 if t["has_plain_round"]]
    with_score = [t for t in ge3_plain if t["evaluator_completion"] is not None]
    score_ge_05 = [t for t in with_score if t["evaluator_completion"] >= 0.5]
    score_eq_1 = [t for t in with_score if t["evaluator_completion"] == 1.0]

    return {
        "total_tasks": total_tasks,
        "assistant_trajectories": len(per_task),
        "ge3_toolcalls": len(ge3),
        "ge3_toolcalls_and_plain_round": len(ge3_plain),
        "ge3_plain_and_has_evaluator_score": len(with_score),
        "score_ge_0.5": len(score_ge_05),
        "score_eq_1": len(score_eq_1),
    }


def main():
    parser = argparse.ArgumentParser(description="轨迹统计")
    parser.add_argument("root", nargs="?", default=r"E:\轨迹\0716",
                        help="根目录，默认 E:\\轨迹\\0716")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 json 文件路径（默认 <根目录>/traj_stats_result.json）")
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print("根目录不存在: %s" % root, file=sys.stderr)
        sys.exit(1)

    per_task = process_root(root)
    summary = summarize(per_task)

    output = args.output or os.path.join(root, "traj_stats_result.json")
    result = {
        "root": os.path.abspath(root),
        "summary": summary,
        "details": per_task,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("统计完成，结果写入: %s" % output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
