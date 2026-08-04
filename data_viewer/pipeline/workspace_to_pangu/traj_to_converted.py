# -*- coding: utf-8 -*-
"""
把 openclaw 的 *.trajectory.jsonl 直接转换成 panguml 训练数据格式。

动机: 每次调用的 model.completed.messagesSnapshot 才是模型当时真实看到的完整上下文
(含 system prompt、thinking、tool_use/tool_result 原始 block), 比 compact .jsonl 更完整;
但 messagesSnapshot 本身在记录层会被砍(整段按字节数被替换成 stub, 或数组按条数砍到 64
条+1条截断标记), 所以按调用顺序增量合并、用 timestamp 对齐去重, 只有在某次调用彻底拿不到
messagesSnapshot 且后续调用也从未把它补回来时, 才退化到用同一次调用的 trace.artifacts
(assistantTexts) 重建一条降级 assistant 消息。

一旦 messagesSnapshot 里出现过 role == "compactionSummary"(openclaw 自身的上下文折叠标记),
就把这条轨迹在该轮次处切开, 写成多个 session(累积前缀模式): 每次新的 compaction 边界写一行
(该行是从头到这个边界为止的完整对话), 边界之前的轮次里每条 assistant 消息 weight=0, 边界及
之后新产生的 assistant 消息 weight=1; 最后再写一行覆盖到文件末尾的完整版本。如果整条轨迹都
没发生过 compaction, 只输出一行, 不带 weight 字段。

三路输出(仿 _split_incomplete_tail: 只有以"有真实文本内容且不带 pending tool_calls"的
assistant 轮结尾才算完整轨迹, 否则末尾是卡在工具调用/无文本的半截轨迹):
  - 没发生过 compaction 且结尾完整                -> --out(主输出)
  - 没发生过 compaction 但结尾不完整              -> --truncated-out(默认 <out 去扩展名>_truncated.jsonl)
  - 发生过 compaction(不论结尾是否完整, 完整时才写, 结尾不完整就静默裁掉不完整的尾巴,
    不再单独进 truncated) -> --fold-out(默认 <out 去扩展名>_fold.jsonl)

只处理 agents/assistant*/sessions/(assistant1/assistant2/...)和 agents/main/sessions/ 下
真正执行任务的轨迹; agents/evaluator/sessions/ 下 evaluator 自己的裁决轨迹不是训练数据,
直接在文件发现阶段过滤掉。

每一路输出都是单个聚合了所有任务轨迹的 .jsonl 文件(仿 testcase/converted.jsonl 的 shape),
不是按输入目录结构散落的一堆文件。

本文件不依赖任何其他本地脚本(split_trajectory_by_call.py / convert_log.py 等), 所需的按调用
分组、内容抽取、消息拼接、完整性判定、工具格式转换等逻辑全部内联在下面, 可以单独拷走使用。

用法:
  python traj_to_converted.py --traj-in "<xxx>.trajectory.jsonl 或目录>" --out "<输出的聚合 .jsonl 文件>" \
      [--fold-out "<...>"] [--truncated-out "<...>"]
"""
import os
import io
import re
import json
import hashlib
import argparse
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# A. 按调用分组 / 文件发现
# ═══════════════════════════════════════════════════════════════════════════════
def group_calls(path):
    """把按事件类型逐行记录的 *.trajectory.jsonl 按 session.started 分界, 合并成
    「一次调用一个 dict」的列表; 一次完整的 LLM 调用固定是 session.started ->
    trace.metadata -> context.compiled -> prompt.submitted -> model.completed ->
    trace.artifacts -> session.ended 这 7 个事件依次出现一轮(因异常可能缺失)。"""
    calls = []
    current = None

    def new_group(obj):
        return {
            "call_index":   len(calls) + 1,
            "trace_id":     obj.get("traceId"),
            "session_id":   obj.get("sessionId"),
            "session_key":  obj.get("sessionKey"),
            "run_id":       obj.get("runId"),
            "provider":     obj.get("provider"),
            "model_id":     obj.get("modelId"),
            "model_api":    obj.get("modelApi"),
            "workspace_dir": obj.get("workspaceDir"),
            "events": {},
        }

    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = obj.get("type")
            if etype == "session.started" and current is not None:
                calls.append(current)
                current = None
            if current is None:
                current = new_group(obj)
            current["events"][etype] = {
                "ts":        obj.get("ts"),
                "seq":       obj.get("seq"),
                "sourceSeq": obj.get("sourceSeq"),
                "data":      obj.get("data"),
            }

    if current is not None:
        calls.append(current)
    return calls


def iter_trajectory_files(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for n in names:
            if n.endswith(".trajectory.jsonl"):
                files.append(os.path.join(root, n))
    return sorted(files)


# ═══════════════════════════════════════════════════════════════════════════════
# B. 内容抽取 / 消息拼接 / 完整性判定 / 工具格式转换(仿 convert_log.py 的同名函数)
# ═══════════════════════════════════════════════════════════════════════════════
def _text_from(content) -> str:
    """Extract plain text from a string, a dict, or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") not in ("tool_use", "tool_result", "thinking", "image"):
                    if "text" in item:
                        parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(content)


def _thinking_from(content) -> str:
    """Extract thinking/reasoning text from content blocks."""
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "thinking":
            reflect = item.get("reflect")
            reflect_text = reflect.get("text") if (
                isinstance(reflect, dict) and reflect.get("status") == "done"
            ) else None
            if isinstance(reflect_text, str) and reflect_text.strip():
                t = reflect_text
            else:
                t = item.get("thinking") or item.get("text", "")
            if t:
                parts.append(t)
    return "\n".join(parts)


def _signature_from(content) -> str:
    """Extract the 'signature' field from thinking content blocks (extended-thinking
    verification token). Kept as-is, never concatenated with reasoning text."""
    if not isinstance(content, list):
        return ""
    sigs = [item.get("signature") for item in content
            if isinstance(item, dict) and item.get("type") == "thinking" and item.get("signature")]
    return "\n".join(sigs)


def _tool_uses_from(content):
    """Return list of tool_use blocks from an assistant message content."""
    if not isinstance(content, list):
        return []
    return [item for item in content
            if isinstance(item, dict) and item.get("type") == "tool_use"]


def _sanitize_messages(messages: list) -> list:
    """Ensure all messages have content as string (never null) and
    assistant messages always have reasoning_content field."""
    for msg in messages:
        if msg.get("content") is None:
            msg["content"] = ""
        if msg.get("role") == "assistant" and "reasoning_content" not in msg:
            msg["reasoning_content"] = ""
        if msg.get("role") == "assistant" and msg.get("reasoning_content") is None:
            msg["reasoning_content"] = ""
    return messages


def _append_user(out: list, text: str):
    if not text.strip():
        return
    if out and out[-1]["role"] == "user":
        out[-1]["content"] = out[-1]["content"] + "\n" + text
    else:
        out.append({"role": "user", "content": text})


def _tool_calls_from_uses(tool_uses: list) -> list:
    """Convert Anthropic tool_use blocks to tool_calls format."""
    result = []
    for idx, item in enumerate(tool_uses):
        tool_id = item.get("id", f"call_{idx:03d}")
        result.append({
            "id": tool_id,
            "type": "function",
            "function": {
                "name": item.get("name", ""),
                "arguments": json.dumps(item.get("input", {}), ensure_ascii=False),
            },
        })
    return result


def _append_assistant(out: list, text: str, reasoning: str, tool_calls: list = None, signature: str = ""):
    tool_calls = tool_calls or []
    if not text.strip() and not reasoning.strip() and not tool_calls:
        return
    if out and out[-1]["role"] == "assistant":
        if text.strip():
            out[-1]["content"] = out[-1]["content"] + "\n" + text
        if reasoning.strip():
            out[-1]["reasoning_content"] = out[-1]["reasoning_content"] + "\n" + reasoning
        if tool_calls:
            out[-1].setdefault("tool_calls", []).extend(tool_calls)
        if signature:
            if out[-1].get("signature"):
                out[-1]["signature"] = out[-1]["signature"] + "\n" + signature
            else:
                out[-1]["signature"] = signature
    else:
        entry = {"role": "assistant", "reasoning_content": reasoning, "content": text}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if signature:
            entry["signature"] = signature
        out.append(entry)


def _append_tool(out: list, text: str, tool_call_id: str = ""):
    """Tool messages are always appended separately (consecutive tool entries are OK)."""
    entry = {"role": "tool", "content": text}
    if tool_call_id:
        entry["tool_call_id"] = tool_call_id
    out.append(entry)


def _split_incomplete_tail(out_messages: list) -> tuple:
    """
    Split off a trailing 'incomplete' portion of the trajectory.

    A trajectory is only considered complete if it ends on an assistant turn
    that has real text content AND no pending tool call. If the newest turn
    has no text, or ends with a tool call (with or without accompanying
    text), it stops mid-task waiting on a tool result rather than on a
    finished answer, so it (and anything after the last safe turn) is split
    off into a separate 'truncated' list.
    """
    last_asst_idx = -1
    has_assistant = False
    for idx, m in enumerate(out_messages):
        if m["role"] == "assistant":
            has_assistant = True
            if m.get("content", "").strip() and not m.get("tool_calls"):
                last_asst_idx = idx
    if last_asst_idx == len(out_messages) - 1:
        return out_messages, []
    if last_asst_idx < 0:
        if not has_assistant:
            return out_messages, []
        return [], out_messages
    return out_messages[: last_asst_idx + 1], out_messages[last_asst_idx + 1 :]


def convert_tools(tools: list) -> list:
    """
    Input:  [{"name": "...", "description": "...", "input_schema": {...}}, ...]
    Output: [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}, ...]
    """
    result = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and "function" in t:
            result.append(t)
            continue
        fn = {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", t.get("parameters", {})),
        }
        result.append({"type": "function", "function": fn})
    return result


def calc_stats(messages: list, response: dict) -> dict:
    tool_num      = sum(1 for m in messages if m["role"] == "tool")
    assistant_num = sum(1 for m in messages if m["role"] == "assistant")
    user_num      = sum(1 for m in messages if m["role"] == "user")
    rounds        = assistant_num  # 统一使用 assistant 轮数
    total_chars   = sum(len(m.get("content") or "") for m in messages)

    usage  = response.get("usage", {}) if isinstance(response, dict) else {}
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    if not tokens:
        tokens = total_chars // 4   # rough estimate

    return {
        "tool_num":      tool_num,
        "assistant_num": assistant_num,
        "user_num":      user_num,
        "rounds":        rounds,
        "tokens":        tokens,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 1: 按调用增量合并出完整、去重的原始消息序列
# ═══════════════════════════════════════════════════════════════════════════════
def _entry_ts(m):
    return m.get("timestamp")


def _fix_truncated_stubs(obj):
    """messagesSnapshot 里任意字段(text/thinking/reflect.text/tool 的 arguments 等)都可能因为
    trajectory-field-size-limit 被整体替换成
    {"truncated":true,"reason":...,"originalChars"/"originalBytes":...,"limitChars"/"limitBytes":...}
    这种 stub(类型从字符串/结构变成一个标记 dict), 上面这些抽取函数都假定这些字段本来的类型
    (字符串等), 这里递归地把这种 stub 提前换成一段可读占位文本, 避免下游按原类型处理时报错。"""
    if isinstance(obj, dict):
        if obj.get("truncated") is True and "reason" in obj:
            orig = obj.get("originalChars", obj.get("originalBytes"))
            limit = obj.get("limitChars", obj.get("limitBytes"))
            return f"[[trajectory truncated: reason={obj.get('reason')}, original={orig}, limit={limit}]]"
        return {k: _fix_truncated_stubs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_truncated_stubs(v) for v in obj]
    return obj


def _compact_log_path(traj_path):
    """.trajectory.jsonl 同目录下, 同 UUID 但不带 .trajectory 后缀的"逐条消息"日志文件路径。
    这个文件按单条消息单独记录, 不受 .trajectory.jsonl 的 trajectory-event-size-limit
    (整个 model.completed.data 按字节数被替换成截断 stub)影响, 可以在 messagesSnapshot
    整段丢失时兜底找回原始内容。"""
    if traj_path and traj_path.endswith(".trajectory.jsonl"):
        return traj_path[: -len(".trajectory.jsonl")] + ".jsonl"
    return None


def _load_compact_log_messages(compact_path):
    """读取 compact 日志, 只保留 type=="message" 的行, 抽出内层 message 字典(role/content/
    timestamp/toolCallId 等)。这里的 content block 形状(thinking/text/toolCall)和
    messagesSnapshot 里的条目完全一致, 可以直接复用同一套下游抽取逻辑(_normalize_tool_call_blocks
    等), 不需要单独写一套解析。"""
    out = []
    if not compact_path or not os.path.isfile(compact_path):
        return out
    with io.open(compact_path, "r", encoding="utf-8") as f:
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
            msg = obj.get("message")
            if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant", "toolResult"):
                continue
            out.append(_fix_truncated_stubs(msg))
    return out


def merge_calls(calls, path=None):
    """
    返回 (full_msgs, boundaries, degraded_calls, compact_recovered, system_text, tools_raw)。

    full_msgs:   合并去重后的原始消息(role 只会是 user/assistant/toolResult, 按时间顺序)。
    boundaries:  检测到"新的" compactionSummary 内容时, 记录下的边界轮次号(1-indexed, 去重)。
    degraded_calls: 有多少次调用最终还是靠 trace.artifacts 兜底重建的(messagesSnapshot 彻底
                    拿不到, 同目录的 compact 日志也没找到/没覆盖这段缺口的兜底方案)。
    compact_recovered: 有多少次调用的缺口是靠同目录的 compact 日志(不带 .trajectory 后缀,
                    同 UUID)完整补回来的, 没有信息损失。
    """
    full_msgs = []
    last_ts = None
    seen_compaction_sig = None
    boundaries = []
    pending_gaps = []          # 暂存 messagesSnapshot 拿不到的调用, 等后续调用是否把内容补回来
    degraded_calls = 0
    compact_recovered = 0
    system_text = None
    tools_raw = None
    compact_cache = {}

    def compact_entries():
        if "v" not in compact_cache:
            compact_cache["v"] = _load_compact_log_messages(_compact_log_path(path))
        return compact_cache["v"]

    def flush_gaps_as_degraded():
        nonlocal degraded_calls, compact_recovered, last_ts
        if not pending_gaps:
            return

        # 优先用同目录的 compact 日志(不受 trajectory-event-size-limit 影响)找回这段缺口的
        # 完整内容(含 thinking/tool_calls/signature), 只有它也没有覆盖到这段缺口时,
        # 才退回旧的 trace.artifacts 纯文本降级重建。
        recovered = [
            m for m in compact_entries()
            if isinstance(m.get("timestamp"), (int, float))
            and (last_ts is None or m["timestamp"] > last_ts)
        ]
        if recovered:
            for m in recovered:
                full_msgs.append(m)
                ts = _entry_ts(m)
                if ts is not None:
                    last_ts = ts
            compact_recovered += len(pending_gaps)
            pending_gaps.clear()
            return

        for gap_call in pending_gaps:
            # messagesSnapshot 整段丢了的调用, 如果是发起新一轮 user 输入的调用,
            # prompt.submitted.data.prompt 里还留着这轮真实的 user 原文, 先把它补回来,
            # 否则这一轮用户说了什么会彻底丢失(trace.artifacts 里没有 user 侧的文本)。
            gap_ps = (gap_call["events"].get("prompt.submitted") or {}).get("data") or {}
            prompt_text = gap_ps.get("prompt")
            if isinstance(prompt_text, str) and prompt_text.strip():
                full_msgs.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                    "timestamp": last_ts,
                    "_degraded": True,
                })

            ta = (gap_call["events"].get("trace.artifacts") or {}).get("data") or {}
            texts = ta.get("assistantTexts")
            if not texts:
                continue  # finalStatus=="error" 之类, 本来就没产出, 不是截断丢的
            joined = "\n".join(t for t in texts if isinstance(t, str) and t.strip())
            if not joined.strip():
                continue
            full_msgs.append({
                "role": "assistant",
                "content": [{"type": "text", "text": joined}],
                "timestamp": last_ts,
                "_degraded": True,
            })
            degraded_calls += 1
        pending_gaps.clear()

    for call in calls:
        ps = (call["events"].get("prompt.submitted") or {}).get("data") or {}
        if not system_text and ps.get("systemPrompt"):
            system_text = ps.get("systemPrompt")
        cc = (call["events"].get("context.compiled") or {}).get("data") or {}
        if not tools_raw and cc.get("tools"):
            tools_raw = cc.get("tools")

        mc = (call["events"].get("model.completed") or {}).get("data") or {}
        snap = mc.get("messagesSnapshot")

        if not isinstance(snap, list) or not snap:
            # 整段被 trajectory-event-size-limit 截断, 拿不到 messagesSnapshot
            pending_gaps.append(call)
            continue

        compaction_entry = None
        real_entries = []
        for item in snap:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role is None:
                continue  # 数组截断标记 {"truncated":true,...}, 跳过
            if role == "compactionSummary":
                compaction_entry = item
                continue  # 不是真实消息, 只用来判定边界
            if role in ("user", "assistant", "toolResult"):
                real_entries.append(item)

        new_entries = real_entries if last_ts is None else [
            m for m in real_entries if (_entry_ts(m) or 0) > last_ts
        ]
        if new_entries:
            # 这次调用拿到了可用的 messagesSnapshot, 之前挂起的 gap 视为已被这份累积快照覆盖
            pending_gaps.clear()
        for m in new_entries:
            m = _fix_truncated_stubs(m)
            full_msgs.append(m)
            ts = _entry_ts(m)
            if ts is not None:
                last_ts = ts

        if compaction_entry is not None:
            sig = hashlib.md5(
                json.dumps(compaction_entry.get("content"), ensure_ascii=False, sort_keys=True)
                .encode("utf-8", "ignore")
            ).hexdigest()
            if sig != seen_compaction_sig:
                seen_compaction_sig = sig
                round_now = sum(1 for m in full_msgs if m.get("role") == "user")
                if round_now not in boundaries:
                    boundaries.append(round_now)

    # 文件末尾仍未被后续调用覆盖的 gap, 先尝试用 compact 日志补全, 补不到的才用 trace.artifacts 兜底
    flush_gaps_as_degraded()

    return full_msgs, boundaries, degraded_calls, compact_recovered, system_text or "", tools_raw or []


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 2: 原始消息 -> 目标格式, 按需打 weight
# ═══════════════════════════════════════════════════════════════════════════════
def _round_of_list(raw_msgs):
    rounds = []
    r = 0
    for m in raw_msgs:
        if m.get("role") == "user":
            r += 1
        rounds.append(r)
    return rounds


def _cutoff_index(round_of, boundary_round):
    """round_of 里最后一个 <= boundary_round 的位置(不含)的切片终点。"""
    end = 0
    for i, r in enumerate(round_of):
        if r <= boundary_round:
            end = i + 1
        else:
            break
    return end


def _normalize_tool_call_blocks(content):
    """openclaw 的 assistant content block 用的字段名和上面这些抽取函数假设的 Anthropic 形状
    对不上, 这里统一改写:
      - 工具调用块是 type=="toolCall"(id/name/arguments), 不是 type=="tool_use"(id/name/input)
        —— 实测这批数据里只出现 toolCall, 从未出现 tool_use。
      - thinking 块的签名字段是 thinkingSignature, 不是 _signature_from() 读取的 signature ——
        不改写的话签名会被静默丢弃(_thinking_from 已经兼容 item.get("thinking"), 只有
        signature 这一个字段名对不上)。"""
    if not isinstance(content, list):
        return content
    out = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "toolCall":
            item = {
                "type": "tool_use",
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "input": item.get("arguments") or {},
            }
        elif isinstance(item, dict) and item.get("type") == "thinking" and not item.get("signature"):
            if item.get("thinkingSignature"):
                item = {**item, "signature": item["thinkingSignature"]}
        out.append(item)
    return out


def build_segment_messages(raw_msgs, system_text, skip_upto_round):
    """
    把 raw_msgs(user/assistant/toolResult) 转成训练数据 messages 格式。
    skip_upto_round 为 None 时不打 weight 字段; 否则每条 assistant 消息都显式打
    weight = 0(所属轮 <= skip_upto_round) 或 1(所属轮 > skip_upto_round)。
    """
    out_messages = [{"role": "system", "content": _text_from(system_text)}]
    round_num = 0

    for m in raw_msgs:
        role = m.get("role")
        content = m.get("content")

        if role == "user":
            round_num += 1
            _append_user(out_messages, _text_from(content))

        elif role == "assistant":
            if not isinstance(content, list):
                content = []
            content = _normalize_tool_call_blocks(content)
            thinking_content = _thinking_from(content)
            signature = _signature_from(content)
            text_blocks = [item for item in content
                           if isinstance(item, dict) and item.get("type") == "text"]
            tool_calls = _tool_calls_from_uses(_tool_uses_from(content))

            reasoning = ""
            text = ""
            if thinking_content:
                reasoning = thinking_content
                if len(text_blocks) == 1:
                    text = text_blocks[0].get("text", "")
                elif len(text_blocks) >= 2:
                    text = "\n".join(tb.get("text", "") for tb in text_blocks)
            elif len(text_blocks) == 2:
                reasoning = text_blocks[0].get("text", "")
                text = text_blocks[1].get("text", "")
            elif len(text_blocks) == 1 and tool_calls:
                text = text_blocks[0].get("text", "")
            elif len(text_blocks) == 1:
                text = text_blocks[0].get("text", "")
            else:
                text = _text_from(content)

            if text.strip() or reasoning.strip() or tool_calls:
                _append_assistant(out_messages, text, reasoning, tool_calls, signature)
                if skip_upto_round is not None:
                    out_messages[-1]["weight"] = 0 if round_num <= skip_upto_round else 1

        elif role == "toolResult":
            tool_call_id = m.get("toolCallId", "")
            _append_tool(out_messages, _text_from(content), tool_call_id)

    return _sanitize_messages(out_messages)


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 3: 输出
# ═══════════════════════════════════════════════════════════════════════════════
def _write_line(fout, messages, tools_raw, source_path, owner, language, category,
                skip_rounds, degraded_calls, compaction_rounds, compact_recovered=0):
    stats = calc_stats(messages, {})
    time_str = datetime.now().strftime("%Y-%m-%d")
    output = {
        "version": "2.0.0",
        "messages": messages,
        "tools": convert_tools(_fix_truncated_stubs(tools_raw)),
        "meta_info": {
            "teacher": "unknown",
            "query_source": "synthesized",
            "response_generate_time": time_str,
            "response_update_time": time_str,
            "owner": owner,
            "language": language,
            "category": category,
            "rounds": stats["rounds"],
            "unique_info": {
                "path": source_path,
                "is_skill": True,
                "info": {
                    "complete_result": 1,
                    "correct_result": 1,
                    "number_of_turns": stats["rounds"],
                    "tokens": stats["tokens"],
                    "rounds": stats["rounds"],
                    "tool_num": stats["tool_num"],
                    "assistant_num": stats["assistant_num"],
                    "skip_rounds": skip_rounds,
                    "degraded_calls": degraded_calls,
                    "compact_recovered": compact_recovered,
                    "compaction_rounds": compaction_rounds,
                },
            },
        },
    }
    # 原始轨迹里偶尔会有截断截断在 UTF-16 代理对中间导致的孤立 surrogate 字符(如 \ude80 缺失配对的
    # \ud83d), json.dumps 能正常序列化, 但写文件时 utf-8 编码会报 UnicodeEncodeError; 这里统一替换掉,
    # 保证输出文件本身始终是合法 UTF-8(容忍这一个字符的信息损失, 因为它本来就已经是坏数据)。
    line = json.dumps(output, ensure_ascii=False)
    line = line.encode("utf-8", errors="replace").decode("utf-8")
    fout.write(line + "\n")
    return stats


def process_file(path, fout_main, fout_fold, fout_truncated, owner, language, category):
    """
    把单个 .trajectory.jsonl 转换出的 1..N 行, 按三分类写进对应的共享聚合输出文件句柄:
    完整(结尾是带真实文本、无 pending tool_calls 的 assistant 轮)且没发生过 compaction ->
    fout_main; 没发生过 compaction 但结尾不完整 -> fout_truncated(把裁掉的不完整尾巴重新拼
    回去、只再裁掉末尾非 assistant 的悬空消息); 发生过 compaction 的每一段 -> fout_fold, 段内
    如果结尾不完整就直接静默丢弃不完整的尾巴, 不再单独进 truncated。
    """
    calls = group_calls(path)
    full_msgs, boundaries, degraded_calls, compact_recovered, system_text, tools_raw = merge_calls(calls, path)

    if not full_msgs:
        print(f"[skip] {os.path.basename(path)}: 没有可用消息")
        return {"file": path, "segments": 0}

    round_of = _round_of_list(full_msgs)

    n_segments = 0
    n_main = n_fold = n_truncated = 0

    if not boundaries:
        messages = build_segment_messages(full_msgs, system_text, skip_upto_round=None)
        clean, tail = _split_incomplete_tail(messages)
        if tail:
            full_trimmed = clean + tail
            while full_trimmed and full_trimmed[-1]["role"] != "assistant":
                full_trimmed.pop()
            _write_line(fout_truncated, full_trimmed, tools_raw, path, owner, language, category,
                        skip_rounds=0, degraded_calls=degraded_calls, compaction_rounds=[],
                        compact_recovered=compact_recovered)
            n_truncated = 1
        else:
            _write_line(fout_main, messages, tools_raw, path, owner, language, category,
                        skip_rounds=0, degraded_calls=degraded_calls, compaction_rounds=[],
                        compact_recovered=compact_recovered)
            n_main = 1
        n_segments = 1
    else:
        skip_upto_round = 0
        for b in boundaries:
            end_idx = _cutoff_index(round_of, b)
            seg_raw = full_msgs[:end_idx]
            messages = build_segment_messages(seg_raw, system_text, skip_upto_round)
            clean, _tail = _split_incomplete_tail(messages)
            if clean:
                _write_line(fout_fold, clean, tools_raw, path, owner, language, category,
                            skip_rounds=skip_upto_round, degraded_calls=degraded_calls,
                            compaction_rounds=boundaries, compact_recovered=compact_recovered)
                n_fold += 1
                n_segments += 1
            skip_upto_round = b
        # 最后一行: 覆盖到文件末尾的完整版本
        messages = build_segment_messages(full_msgs, system_text, skip_upto_round)
        clean, _tail = _split_incomplete_tail(messages)
        if clean:
            _write_line(fout_fold, clean, tools_raw, path, owner, language, category,
                        skip_rounds=skip_upto_round, degraded_calls=degraded_calls,
                        compaction_rounds=boundaries, compact_recovered=compact_recovered)
            n_fold += 1
            n_segments += 1

    print(f"[{os.path.basename(path)}] {n_segments} 段(compaction 边界={boundaries}, "
          f"degraded_calls={degraded_calls}, compact_recovered={compact_recovered}, "
          f"main={n_main}, fold={n_fold}, truncated={n_truncated})")
    return {"file": path, "segments": n_segments, "boundaries": boundaries,
            "degraded_calls": degraded_calls, "compact_recovered": compact_recovered,
            "n_main": n_main, "n_fold": n_fold, "n_truncated": n_truncated}


_AGENT_DIR_RE = re.compile(r"/agents/([^/]+)/sessions/")


def _is_included_agent_traj(path):
    """只处理真正执行任务的轨迹: agents/assistant*/sessions/(assistant1/assistant2/...)和
    agents/main/sessions/。agents/evaluator/sessions/ 是 evaluator 自己对任务的裁决轨迹,
    不是训练数据, 排除掉; 其他未知 agent 目录也一并排除(不静默假设它们该被收录)。"""
    norm = path.replace("\\", "/")
    m = _AGENT_DIR_RE.search(norm)
    if not m:
        return False
    agent_name = m.group(1)
    return agent_name == "main" or agent_name.startswith("assistant")


def main():
    ap = argparse.ArgumentParser(
        description="把 openclaw *.trajectory.jsonl 转成训练数据格式, "
                    "按 openclaw 自身的上下文 compaction 拆分成多个 session 并打 weight, "
                    "所有任务聚合写进同一个输出文件")
    ap.add_argument("--traj-in", required=True, help="单个 .trajectory.jsonl 文件, 或包含若干该类文件的目录")
    ap.add_argument("--out", required=True, help="主输出: 无 compaction 且结尾完整的聚合 .jsonl 文件路径")
    ap.add_argument("--fold-out", default=None,
                    help="发生过 compaction 的轨迹聚合输出(默认 <out 去扩展名>_fold.jsonl)")
    ap.add_argument("--truncated-out", default=None,
                    help="无 compaction 但结尾不完整(卡在工具调用/无文本)的轨迹聚合输出"
                        "(默认 <out 去扩展名>_truncated.jsonl)")
    ap.add_argument("--owner", default="00935640")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--category", default="agent")
    a = ap.parse_args()

    if not os.path.exists(a.traj_in):
        ap.error(f"输入不存在: {a.traj_in}")

    out_base, out_ext = os.path.splitext(os.path.abspath(a.out))
    out_ext = out_ext or ".jsonl"
    fold_out = a.fold_out or (out_base + "_fold" + out_ext)
    truncated_out = a.truncated_out or (out_base + "_truncated" + out_ext)
    for p in (a.out, fold_out, truncated_out):
        os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)

    all_files = iter_trajectory_files(a.traj_in)
    files = [f for f in all_files if _is_included_agent_traj(f)]
    n_excluded = len(all_files) - len(files)
    if not files:
        ap.error(f"未找到 *.trajectory.jsonl 文件(仅收录 agents/assistant*/sessions 和 "
                f"agents/main/sessions 后): {a.traj_in}")

    with io.open(a.out, "w", encoding="utf-8") as fout_main, \
         io.open(fold_out, "w", encoding="utf-8") as fout_fold, \
         io.open(truncated_out, "w", encoding="utf-8") as fout_truncated:
        stats = [process_file(f, fout_main, fout_fold, fout_truncated, a.owner, a.language, a.category)
                  for f in files]

    total_segments = sum(s.get("segments", 0) for s in stats)
    total_main = sum(s.get("n_main", 0) for s in stats)
    total_fold = sum(s.get("n_fold", 0) for s in stats)
    total_truncated = sum(s.get("n_truncated", 0) for s in stats)
    with_compaction = sum(1 for s in stats if s.get("boundaries"))

    summary_path = a.out + ".stats.json"
    with io.open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "traj_in": a.traj_in, "out": a.out, "fold_out": fold_out, "truncated_out": truncated_out,
            "file_count": len(files), "excluded_non_task_agent_files": n_excluded,
            "total_segments": total_segments, "total_main": total_main,
            "total_fold": total_fold, "total_truncated": total_truncated,
            "files_with_compaction": with_compaction, "files": stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[done] {len(files)} 个文件(排除非 assistant*/main 的 agent 轨迹 {n_excluded} 个), "
          f"共 {total_segments} 段(main={total_main} -> {a.out}, fold={total_fold} -> {fold_out}, "
          f"truncated={total_truncated} -> {truncated_out}), 其中 {with_compaction} 个文件发生过 "
          f"compaction (统计: {summary_path})")


if __name__ == "__main__":
    main()
