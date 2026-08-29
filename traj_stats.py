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
import sqlite3
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


def _char_len(path):
    """返回轨迹文件的原始字符数(读取为文本, 非法字节替换)。文件不存在/读取失败返回 0。"""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return len(f.read())
    except OSError:
        return 0


def has_task_done_marker(log_path):
    """检查 task 主 log 是否包含「【Task_Done】」标记(assistant 回答末尾输出的任务完成信号)。
    log 不存在时视为 False。"""
    if not os.path.isfile(log_path):
        return False
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            return "【Task_Done】" in f.read()
    except OSError:
        return False


def check_task_done_in_logs_dir(logs_dir, task_name):
    """在 logs 目录下检查多个候选 log 文件是否含「【Task_Done】」标记。

    有些 harness 将主 log 命名为 <task>.log，有些命名为 harness_automation.log。
    依次尝试所有候选文件名，有任一匹配即返回 True。"""
    candidates = [
        task_name + ".log",
        "harness_automation.log",
    ]
    return any(has_task_done_marker(os.path.join(logs_dir, name)) for name in candidates)


def analyze_trajectory(path):
    """分析一条 assistant 轨迹并提取 token 用量。

    返回 dict: {tool_calls, plain_rounds, assistant_rounds, input_tokens, output_tokens, reasoning_tokens, total_tokens}
      tool_calls       : 全轨迹中 toolCall 的总次数
      plain_rounds     : 不带工具调用的 assistant 轮数（只有 thinking / text）
      assistant_rounds : assistant 消息轮数
      *_tokens         : 各 assistant 消息 usage 字段累加（无 usage 时 = 0）
    """
    tool_calls = 0
    plain_rounds = 0
    assistant_rounds = 0
    input_tk = output_tk = reasoning_tk = 0

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
            # token 用量: 每条 assistant 消息可能有 usage
            usage = msg.get("usage")
            if isinstance(usage, dict):
                input_tk += usage.get("input") or 0
                output_tk += usage.get("output") or 0
                reasoning_tk += usage.get("reasoningTokens") or 0

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
        "input_tokens": input_tk,
        "output_tokens": output_tk,
        "reasoning_tokens": reasoning_tk,
        "total_tokens": input_tk + output_tk + reasoning_tk,
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


# ── 工具调用失败统计(显式错误标记口径) ─────────────────────────────────────────
# 只认「结构化的显式错误标记」, 不做关键字扫描(正文出现 error/异常 不计), 以避免误报:
#   - openclaw: assistant .jsonl 里 role=="toolResult" 的 message, isError==True 或
#               details.exitCode 为非 0 整数, 即该次工具调用失败。
#   - Hermes  : query1.json 的 turns[].tool_calls[].output(JSON 对象)命中 ok:false /
#               truthy error / success:false / exit_code|returncode 非 0 整数, 即失败。
# 无显式标记的工具(如 exec 纯文本 stdout)其失败不计入 —— 符合「仅显式标记」口径。

def _toolcall_output_is_error(output) -> bool:
    """Hermes 单次 tool_call 的 output 是否为显式失败。output 多为 JSON 字符串。"""
    obj = output
    if isinstance(output, str):
        s = output.strip()
        if not s or s[:1] not in "{[":
            return False
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return False
    if not isinstance(obj, dict):
        return False
    if obj.get("ok") is False or obj.get("success") is False:
        return True
    if obj.get("error"):
        return True
    for k in ("exit_code", "returncode"):
        v = obj.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v != 0:
            return True
    return False


def _toolresult_msg_is_error(msg) -> bool:
    """openclaw 单条 role=='toolResult' 的 message 是否为显式失败。"""
    if not isinstance(msg, dict):
        return False
    if msg.get("isError"):
        return True
    details = msg.get("details")
    if isinstance(details, dict):
        code = details.get("exitCode")
        if isinstance(code, int) and not isinstance(code, bool) and code != 0:
            return True
    return False


# ── 工具误用 / 环境不符统计(宽口径) ──────────────────────────────────────────────
# 任务实际跑在 Linux openclaw 环境。以下任一信号出现即视为一次「误用」, 命中任一即算:
#   1. 调用了本环境不存在的工具: harness 回一条 role==toolResult 文本 "Tool X not found"。
#      宽口径 —— 任意 X 都算(不分大小写), 因为每条都等于模型调了 openclaw 没有的工具:
#        · Claude-Code/Windows 工具名: WebSearch / PowerShell / Glob / WebFetch /
#          TaskCreate / EnterPlanMode / Agent / Skill / TodoWrite ...(模型以为在
#          Claude-Code/Windows 环境);
#        · 被误当工具直接调的 shell 命令: grep / find / cat / ls / curl / stat ...
#          (正确应走 exec; 模型常先误调 not found、再改用 exec 重试);
#        · 大小写/拼写变体与幻觉: webfetch / web_search / exce / filesever ...。
#      这类调用 openclaw 不记成 toolCall 部件, 只在下一条 toolResult 留文本, 故检测走
#      toolResult 文本(见 _tool_not_found_count), 而非 toolCall.name —— 后者在
#      openclaw 词表(snake_case: read/exec/write...)下永远匹配不到。
#   2. 入参含 file_path 键(Windows 版 write/read 用 file_path 作路径入参, Linux 版
#      用 path)。
#   3. 任一入参值为 Windows 风格路径: 盘符冒号斜杠开头(C:\\Users\\... 这种,
#      Linux 是 /home/...)。
# 规则 2/3 只深入「入参本身」, 不进入 content 正文载荷(HTML/脚本/JSON 文本里出现的
# C:\\ 或 file_path 不代表调用发生在 Windows, 避免误报)。不区分调用成败。
# WIN_TOOL_NAMES 保留给 toolCall.name 路径(_is_windows_tool_call); openclaw 实际靠
# 规则 1 的 toolResult 扫描 + 规则 2 的 file_path 键命中。

WIN_TOOL_NAMES = frozenset({
    "powershell", "glob", "websearch", "webfetch", "taskcreate",
})

# 规则 1(宽口径): toolResult 文本里 "Tool X not found" 的次数, 任意 X 均计。
# X 允许 字母/数字/下划线/点/连字符(覆盖 web_search、cmd.exe、web-search 等写法)。
_TOOL_NOT_FOUND_RE = re.compile(r"Tool\s+([A-Za-z0-9_.\-]+)\s+not found")


def _tool_not_found_count(text) -> int:
    """toolResult 文本里 'Tool X not found' 的出现次数(宽口径: 任意 X 均计, 不分大小写,
    含 Claude-Code/Windows 工具名、被误当工具的 shell 命令、拼写幻觉)。非字符串返回 0。"""
    if not isinstance(text, str) or "not found" not in text:
        return 0
    return len(_TOOL_NOT_FOUND_RE.findall(text))
_WIN_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_MAX_JSON_PARSE_LEN = 8192  # 超过此长度的字符串(如整段 HTML content)不做 JSON 解析


def _is_win_path_value(value) -> bool:
    """字符串值是否为 Windows 风格路径(盘符冒号斜杠开头)。非字符串返回 False。"""
    if not isinstance(value, str):
        return False
    return bool(_WIN_PATH_RE.match(value.strip()))


def _win_signal_in_args(args) -> bool:
    """递归判断工具入参是否含 Windows 信号(file_path 键 / Windows 风格路径值)。

    file_path 键与路径值都只在「非 content 载荷」分支里找: content 是正文大字符串,
    其中的 C:\\ 与 file_path 均为内容而非 Windows 环境的证据。"""
    if isinstance(args, dict):
        for k, v in args.items():
            if k == "file_path":
                return True
            if k == "content":
                continue  # 正文载荷不参与判定
            if _win_signal_in_args(v):
                return True
    elif isinstance(args, str):
        s = args.strip()
        if s[:1] in "{[" and len(s) <= _MAX_JSON_PARSE_LEN:
            try:
                if _win_signal_in_args(json.loads(s)):
                    return True
            except json.JSONDecodeError:
                pass
        if _is_win_path_value(s):
            return True
    elif isinstance(args, (list, tuple)):
        return any(_win_signal_in_args(x) for x in args)
    return False


def _is_windows_tool_call(name, args) -> bool:
    """单次工具调用是否为 Windows 环境: 工具名命中或入参命中显式信号。"""
    if name and name.strip().lower() in WIN_TOOL_NAMES:
        return True
    return _win_signal_in_args(args)


# ── 分规则命中: 把「工具误用」宽口径拆成 3 条规则, 供分列统计 ────────────────────
# rule 1 = 调用本环境不存在的工具(toolResult 里 "Tool X not found" / toolCall 名命中
#          WIN_TOOL_NAMES); rule 2 = 入参含 file_path 键; rule 3 = 入参为 Windows 风格
#          路径值。一次 toolCall 可同时命中 2/3(各计一次)。分支与 _win_signal_in_args
#          完全一致(同样跳过 content 载荷、file_path 命中即不再深入其值), 只是把短路
#          bool 改成收集规则集合, 保证与旧「是否命中」口径逐调用一致。
def _win_signal_rules_in_args(args, rules=None):
    """递归收集入参命中的规则集合 ⊆ {2, 3}。"""
    if rules is None:
        rules = set()
    if isinstance(args, dict):
        for k, v in args.items():
            if k == "file_path":
                rules.add(2)
                continue  # 与 _win_signal_in_args 一致: 命中 file_path 键即不再深入其值
            if k == "content":
                continue  # 正文载荷不参与判定
            _win_signal_rules_in_args(v, rules)
    elif isinstance(args, str):
        s = args.strip()
        if s[:1] in "{[" and len(s) <= _MAX_JSON_PARSE_LEN:
            try:
                _win_signal_rules_in_args(json.loads(s), rules)
            except json.JSONDecodeError:
                pass
        if _is_win_path_value(s):
            rules.add(3)
    elif isinstance(args, (list, tuple)):
        for x in args:
            _win_signal_rules_in_args(x, rules)
    return rules


def _windows_call_rules(name, args):
    """单次 toolCall 命中的规则集合 ⊆ {1, 2, 3}(工具名命中归入规则 1)。"""
    rules = _win_signal_rules_in_args(args)
    if name and name.strip().lower() in WIN_TOOL_NAMES:
        rules.add(1)
    return rules


def count_tool_rule_stats_openclaw(jsonl_path):
    """单遍遍历 openclaw assistant .jsonl, 返回
    (tool_calls, tool_fails, win_tool_calls, r1_calls, r2_calls, r3_calls)。

    tool_calls = assistant 消息里 toolCall 部件总数(再加规则 1 检出的误用调用, 见下);
    tool_fails = 显式失败的 toolResult 数(口径与 count_tool_failures_openclaw 一致);
    win_tool_calls = 命中「工具误用/环境不符」信号的调用数(宽口径, 一次调用命中多条只计一次);
    r1/r2/r3_calls = 各规则单独命中的调用数(一次调用同时命中 2/3 时 r2、r3 各计一次,
        故 r1+r2+r3 ≥ win_tool_calls)。规则 1 = 调用本环境不存在的工具(toolResult 里
        "Tool X not found" 次数 + toolCall 名命中 WIN_TOOL_NAMES); 规则 2 = file_path
        入参键; 规则 3 = Windows 风格路径入参值。字段名沿用 win_* 仅为兼容。
    """
    tool_calls = 0
    tool_fails = 0
    win_calls = 0
    r1 = r2 = r3 = 0
    try:
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
                role = msg.get("role")
                if role == "assistant":
                    content = msg.get("content")
                    if isinstance(content, list):
                        for p in content:
                            if not isinstance(p, dict) or p.get("type") != "toolCall":
                                continue
                            tool_calls += 1
                            rules = _windows_call_rules(
                                p.get("name") or p.get("toolName") or "",
                                p.get("arguments") if "arguments" in p else p.get("input"),
                            )
                            if rules:
                                win_calls += 1
                                if 1 in rules:
                                    r1 += 1
                                if 2 in rules:
                                    r2 += 1
                                if 3 in rules:
                                    r3 += 1
                elif role == "toolResult":
                    if _toolresult_msg_is_error(msg):
                        tool_fails += 1
                    # 规则 1: "Tool X not found" 里 PascalCase 的 X = 一次误用 Windows/
                    # Claude-Code 工具的调用。这类调用无对应 toolCall 部件, 故这里同时补进
                    # tool_calls(分母)与 win_calls, 保证 win_rate ≤ 1。
                    rc = msg.get("content")
                    texts = ([p.get("text") for p in rc
                              if isinstance(p, dict) and p.get("type") == "text"]
                             if isinstance(rc, list)
                             else [rc] if isinstance(rc, str) else [])
                    for txt in texts:
                        n = _tool_not_found_count(txt)
                        if n:
                            win_calls += n
                            tool_calls += n
                            r1 += n
    except OSError:
        return (0, 0, 0, 0, 0, 0)
    return (tool_calls, tool_fails, win_calls, r1, r2, r3)


def count_tool_stats_openclaw(jsonl_path):
    """(向后兼容包装) 返回 (tool_calls, tool_fails, win_tool_calls); 分规则见
    count_tool_rule_stats_openclaw。"""
    c, f, w, _r1, _r2, _r3 = count_tool_rule_stats_openclaw(jsonl_path)
    return (c, f, w)


def count_tool_rule_stats_query1(query1_path):
    """单遍遍历 Hermes query1.json 的 turns[].tool_calls[], 返回
    (tool_calls, tool_fails, win_tool_calls, r1_calls, r2_calls, r3_calls)。口径同
    count_tool_rule_stats_openclaw; Hermes 侧规则 2/3 依赖结构化入参, 规则 1 为 best-effort
    (扫 output/agent_content 里的 "Tool X not found")。"""
    tool_calls = 0
    tool_fails = 0
    win_calls = 0
    r1 = r2 = r3 = 0
    try:
        with open(query1_path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return (0, 0, 0, 0, 0, 0)
    for t in (data.get("turns") or []):
        if not isinstance(t, dict):
            continue
        for tc in (t.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            tool_calls += 1
            if _toolcall_output_is_error(tc.get("output")):
                tool_fails += 1
            rules = _windows_call_rules(
                tc.get("tool") or tc.get("name") or "",
                tc.get("input") if "input" in tc else tc.get("arguments"),
            )
            if rules:
                win_calls += 1
            # 规则 1(best-effort): 误用工具的 "Tool X not found" 若落在本次 tool 的
            # output 里, 也计一次 Windows 调用(Hermes 侧多为 JSON 字符串 output)。
            out = tc.get("output")
            out_nf = _tool_not_found_count(out) if isinstance(out, str) else 0
            win_calls += out_nf  # 与旧口径逐字一致: 每个 "Tool X not found" 各计一次
            name_hit = 1 if 1 in rules else 0
            r1 += name_hit + out_nf
            if 2 in rules:
                r2 += 1
            if 3 in rules:
                r3 += 1
        # 规则 1(best-effort): 有的 not-found 报错落在回合正文 agent_content 里。
        ac = t.get("agent_content")
        if isinstance(ac, str):
            n = _tool_not_found_count(ac)
            if n:
                win_calls += n
                tool_calls += n
                r1 += n
    return (tool_calls, tool_fails, win_calls, r1, r2, r3)


def count_tool_stats_query1(query1_path):
    """(向后兼容包装) 返回 (tool_calls, tool_fails, win_tool_calls); 分规则见
    count_tool_rule_stats_query1。"""
    c, f, w, _r1, _r2, _r3 = count_tool_rule_stats_query1(query1_path)
    return (c, f, w)


def count_tool_failures_openclaw(jsonl_path):
    """(向后兼容包装) 遍历 openclaw assistant .jsonl, 返回 (tool_calls, tool_fails)。"""
    c, f, _w = count_tool_stats_openclaw(jsonl_path)
    return (c, f)


def count_tool_failures_query1(query1_path):
    """(向后兼容包装) 遍历 Hermes query1.json, 返回 (tool_calls, tool_fails)。"""
    c, f, _w = count_tool_stats_query1(query1_path)
    return (c, f)


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


def read_hermes_state_db(task_dir):
    """读取 Hermes 的 profiles/assistant1/state.db token 使用量。

    返回 {input_tokens, output_tokens, reasoning_tokens, total_tokens}；
    state.db 不存在或读取失败返回 None。"""
    db_path = os.path.join(task_dir, "profiles", "assistant1", "state.db")
    if not os.path.isfile(db_path):
        return None
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
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


def process_root(root):
    per_task = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        task = entry.name
        task_dir = entry.path
        task_done = check_task_done_in_logs_dir(os.path.join(task_dir, "logs"), task)

        # 格式探测: 有 assistant sessions jsonl -> openclaw; 否则有 query1.json -> Hermes
        traj_paths = find_assistant_trajectories(task_dir)
        if traj_paths:
            has_eval, score, verdict_source = resolve_first_verdict(task_dir, task)
            for tp in traj_paths:
                info = analyze_trajectory(tp)
                gate = info["tool_calls"] >= 3 and info["plain_rounds"] > 0
                entry = {
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
                    "char_len": _char_len(tp),
                    "task_done": task_done,
                }
                # 透传 token 数据（OC 从 .jsonl usage 字段提取）
                if info.get("total_tokens"):
                    entry["input_tokens"] = info["input_tokens"]
                    entry["output_tokens"] = info["output_tokens"]
                    entry["reasoning_tokens"] = info["reasoning_tokens"]
                    entry["total_tokens"] = info["total_tokens"]
                per_task.append(entry)
            continue

        # Hermes: 优先用 profiles/*/sessions/*.json (完整轨迹), 回退 query1.json (聚合视图)
        hermes_sessions = find_hermes_sessions(task_dir)
        token_info = read_hermes_state_db(task_dir)  # Hermes task 共享同一份 state.db
        if hermes_sessions:
            # 有 profiles sessions: 每个 session 独立统计（一个 task 可能有多次重试）
            for session_path in hermes_sessions:
                info = analyze_hermes_session(session_path)
                has_eval, score = extract_query1_verdict(find_query1_json(task_dir) or "")
                # Hermes L1 门槛: 有产出即可(plain_rounds > 0)
                gate = info["plain_rounds"] > 0
                entry = {
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
                    "char_len": _char_len(session_path),
                    "task_done": task_done,
                }
                if token_info:
                    entry["input_tokens"] = token_info["input_tokens"]
                    entry["output_tokens"] = token_info["output_tokens"]
                    entry["reasoning_tokens"] = token_info["reasoning_tokens"]
                    entry["total_tokens"] = token_info["total_tokens"]
                per_task.append(entry)
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
        entry = {
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
            "char_len": _char_len(query1),
            "task_done": task_done,
        }
        if token_info:
            entry["input_tokens"] = token_info["input_tokens"]
            entry["output_tokens"] = token_info["output_tokens"]
            entry["reasoning_tokens"] = token_info["reasoning_tokens"]
            entry["total_tokens"] = token_info["total_tokens"]
        per_task.append(entry)
    return per_task


def summarize(per_task):
    total_tasks = len({t["task"] for t in per_task})

    ge3 = [t for t in per_task if t["has_ge3_toolcalls"]]
    ge3_plain = [t for t in ge3 if t["has_plain_round"]]
    with_score = [t for t in ge3_plain if t["evaluator_completion"] is not None]
    score_ge_05 = [t for t in with_score if t["evaluator_completion"] >= 0.5]
    score_eq_1 = [t for t in with_score if t["evaluator_completion"] == 1.0]
    task_done_count = len({t["task"] for t in per_task if t.get("task_done")})

    return {
        "total_tasks": total_tasks,
        "assistant_trajectories": len(per_task),
        "ge3_toolcalls": len(ge3),
        "ge3_toolcalls_and_plain_round": len(ge3_plain),
        "ge3_plain_and_has_evaluator_score": len(with_score),
        "score_ge_0.5": len(score_ge_05),
        "score_eq_1": len(score_eq_1),
        "task_done_count": task_done_count,
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
