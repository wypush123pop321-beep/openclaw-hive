#!/usr/bin/env python3
"""
Convert the latest JSON conversation log from source directory to a training data line.

Usage:
    python convert_log.py <source_dir> <target_file> [options]

Example:
    python convert_log.py "D:\\logs\\2026-03-20_02-37-08_180" "C:\\output\\data.json"
"""

import os
import re
import sys
import json
import glob
import argparse
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# 1. File discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_json(source_dir: str) -> str:
    """Return the JSON file whose filename timestamp is the latest."""
    files = glob.glob(os.path.join(source_dir, "*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {source_dir}")

    def _ts(path: str):
        name = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r'(\d{4}-\d{2}-\d{2})[_T](\d{2})[-:](\d{2})[-:](\d{2})', name)
        if m:
            try:
                return datetime(int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:]),
                                int(m.group(2)), int(m.group(3)), int(m.group(4)))
            except ValueError:
                pass
        return datetime.fromtimestamp(os.path.getmtime(path))

    files.sort(key=_ts, reverse=True)
    return files[0]


def find_all_json_sorted(source_dir: str) -> list:
    """Return all JSON files sorted by filename timestamp (oldest first)."""
    files = glob.glob(os.path.join(source_dir, "*.json"))
    if not files:
        return []

    def _ts(path: str):
        name = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r'(\d{4}-\d{2}-\d{2})[_T](\d{2})[-:](\d{2})[-:](\d{2})', name)
        if m:
            try:
                return datetime(int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:]),
                                int(m.group(2)), int(m.group(3)), int(m.group(4)))
            except ValueError:
                pass
        return datetime.fromtimestamp(os.path.getmtime(path))

    files.sort(key=_ts, reverse=False)  # oldest first
    return files


def has_tool_fold(data: dict) -> bool:
    """Check if the data contains 'tool output removed to free context'."""
    data_str = json.dumps(data, ensure_ascii=False)
    return "tool output removed to free context" in data_str


def count_tool_folds(data: dict) -> int:
    """Count the number of 'tool output removed to free context' occurrences in data."""
    data_str = json.dumps(data, ensure_ascii=False)
    return data_str.count("tool output removed to free context")


def count_rounds_in_messages(messages: list) -> int:
    """Count the number of assistant turns in messages."""
    return sum(1 for m in messages if m.get("role") == "assistant")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Content helpers
# ──────────────────────────────────────────────────────────────────────────────

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
                    # unknown block with a text field
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
            t = item.get("thinking") or item.get("text", "")
            if t:
                parts.append(t)
    return "\n".join(parts)


def _is_openai_format(data: dict) -> bool:
    """Detect if the log is in OpenAI-compatible format (vs Anthropic format).

    OpenAI format signals:
    - assistant messages have top-level 'reasoning_content' or 'tool_calls'
    - tool-role messages exist (instead of user messages with tool_result blocks)
    - response has only status_code (no content)
    """
    for msg in data.get("messages", []):
        if msg.get("role") == "assistant" and (
            "reasoning_content" in msg or "tool_calls" in msg
        ):
            return True
        if msg.get("role") == "tool":
            return True
    return False


def _tool_results_from(content):
    """Return list of tool_result blocks from a user message content."""
    if not isinstance(content, list):
        return []
    return [item for item in content
            if isinstance(item, dict) and item.get("type") == "tool_result"]


def _tool_result_text(item: dict) -> str:
    """Extract text from a tool_result block."""
    c = item.get("content", "")
    return _text_from(c)


def _tool_uses_from(content):
    """Return list of tool_use blocks from an assistant message content."""
    if not isinstance(content, list):
        return []
    return [item for item in content
            if isinstance(item, dict) and item.get("type") == "tool_use"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Message-list builders (no consecutive same role for user/assistant)
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_messages(messages: list) -> list:
    """Ensure all messages have content as string (never null) and
    assistant messages always have reasoning_content field."""
    for msg in messages:
        # content must be string, never null
        if msg.get("content") is None:
            msg["content"] = ""
        # assistant messages must have reasoning_content
        if msg.get("role") == "assistant" and "reasoning_content" not in msg:
            msg["reasoning_content"] = ""
        if msg.get("role") == "assistant" and msg.get("reasoning_content") is None:
            msg["reasoning_content"] = ""
    return messages


def _append_user(out: list, text: str):
    # 保留原始 user content，不处理 sender 和时间戳前缀
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


def _append_assistant(out: list, text: str, reasoning: str, tool_calls: list = None):
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
    else:
        entry = {"role": "assistant", "reasoning_content": reasoning, "content": text}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)


def _append_tool(out: list, text: str, tool_call_id: str = ""):
    """Tool messages are always appended separately (consecutive tool entries are OK)."""
    entry = {"role": "tool", "content": text}
    if tool_call_id:
        entry["tool_call_id"] = tool_call_id
    out.append(entry)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Convert tools: Anthropic → OpenAI format
# ──────────────────────────────────────────────────────────────────────────────

def convert_tools(tools: list) -> list:
    """
    Input:  [{"name": "...", "description": "...", "input_schema": {...}}, ...]
    Output: [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}, ...]
    """
    result = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # Already in OpenAI format?
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


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main conversion
# ──────────────────────────────────────────────────────────────────────────────

def convert_openai(data: dict, skip_fold_check: bool = False) -> tuple:
    """
    Convert an OpenAI-compatible API-request-log dict into (messages, tools, model_name, has_fold).

    The OpenAI-format log has:
        messages – list of turns (system/user/assistant/tool)
        model    – model name
        tools    – OpenAI-format tool list (already in target format)
        response – only status_code, no content

    Assistant messages have:
        content: null or str or list
        reasoning_content: str (may be absent)
        tool_calls: list of OpenAI-format tool calls
    Tool messages have:
        content: str
        tool_call_id: str
    """
    # Check for tool fold
    has_fold = False
    data_str = json.dumps(data, ensure_ascii=False)
    if "tool output removed to free context" in data_str:
        has_fold = True
        if not skip_fold_check:
            return [], [], [], None, True

    out_messages = []
    raw_tools = data.get("tools", [])
    model_name = data.get("model", "unknown")

    for turn in data.get("messages", []):
        role = turn.get("role", "")
        content = turn.get("content")
        reasoning_content = turn.get("reasoning_content")
        tool_calls = turn.get("tool_calls")

        if role == "system":
            system_text = _text_from(content)
            out_messages.append({"role": "system", "content": system_text})

        elif role == "user":
            # User content may be a string, null, or list of content blocks
            user_text = _text_from(content)
            _append_user(out_messages, user_text)

        elif role == "assistant":
            # content may be null → use ""
            text = _text_from(content)
            # reasoning_content may be absent → use ""
            reasoning = reasoning_content if reasoning_content else ""
            # tool_calls are already in OpenAI format
            tc = tool_calls if tool_calls else []
            _append_assistant(out_messages, text, reasoning, tc)

        elif role == "tool":
            tool_text = _text_from(content)
            tool_call_id = turn.get("tool_call_id", "")
            _append_tool(out_messages, tool_text, tool_call_id)

    # In OpenAI format, response has no content, so skip response processing.
    # Ensure the last message is always role: assistant
    while out_messages and out_messages[-1]["role"] != "assistant":
        out_messages.pop()

    return _sanitize_messages(out_messages), [], convert_tools(raw_tools), model_name, has_fold


def convert(data: dict, skip_fold_check: bool = False) -> tuple:
    """
    Convert a single API-request-log dict into (messages, tools, model_name, has_fold).

    Auto-detects whether the log is in Anthropic or OpenAI format and dispatches accordingly.
    """
    if _is_openai_format(data):
        return convert_openai(data, skip_fold_check=skip_fold_check)

    # ── Anthropic format below ─────────────────────────────────────────────

    """
    The Anthropic-format log dict has:
        system   – system prompt (str or list of content blocks)
        messages – list of prior turns
        response – the final model response (same shape as an assistant turn)
        model    – model name
        tools    – Anthropic-format tool list

    Returns:
        (messages, truncated, tools, model_name, has_fold)
        has_fold: True if data contains "tool output removed to free context"
    """
    # 检查整个对话数据是否包含 "tool output removed to free context"
    has_fold = False
    data_str = json.dumps(data, ensure_ascii=False)
    if "tool output removed to free context" in data_str:
        has_fold = True
        if not skip_fold_check:
            return [], [], [], None, True  # 返回空数据，标记为折叠

    out_messages = []
    raw_tools    = data.get("tools", [])
    model_name   = data.get("model", "unknown")

    # ── system ──────────────────────────────────────────────────────────────
    system_raw = data.get("system", "")
    system_text = _text_from(system_raw)
    out_messages.append({"role": "system", "content": system_text})

    # ── conversation turns ──────────────────────────────────────────────────
    all_turns = list(data.get("messages", []))

    for turn in all_turns:
        role    = turn.get("role", "")
        content = turn.get("content", "")

        if role == "user":
            tool_results = _tool_results_from(content)
            # Tool results: one output entry per result
            for tr in tool_results:
                tool_call_id = tr.get("tool_use_id", "")
                _append_tool(out_messages, _tool_result_text(tr), tool_call_id)
            # Any plain user text
            user_text_items = (
                [item for item in content
                 if isinstance(item, dict) and item.get("type") == "text"]
                if isinstance(content, list) else []
            )
            plain = _text_from(user_text_items) if user_text_items else (
                "" if tool_results else _text_from(content)
            )
            _append_user(out_messages, plain)

        elif role == "assistant":
            # Process assistant content with proper priority:
            # 1. thinking blocks -> reasoning_content (highest priority)
            # 2. text blocks handling
            # 3. tool_calls
            if not isinstance(content, list):
                content = []

            # Extract thinking blocks first (highest priority for reasoning)
            thinking_content = _thinking_from(content)

            # Extract text blocks
            text_blocks = [item for item in content
                           if isinstance(item, dict) and item.get("type") == "text"]

            # Extract tool calls
            tool_calls = _tool_calls_from_uses(_tool_uses_from(content))

            # Determine reasoning and text
            reasoning = ""
            text = ""

            if thinking_content:
                # Priority 1: thinking exists -> use as reasoning
                reasoning = thinking_content
                # Text blocks go to content
                if len(text_blocks) == 1:
                    text = text_blocks[0].get("text", "")
                elif len(text_blocks) >= 2:
                    # Multiple text blocks: concatenate or use first as reasoning too?
                    # Usually with thinking, text blocks are content
                    text = "\n".join(tb.get("text", "") for tb in text_blocks)
            elif len(text_blocks) == 2:
                # Priority 2: two text blocks -> first is reasoning, second is content
                reasoning = text_blocks[0].get("text", "")
                text = text_blocks[1].get("text", "")
            elif len(text_blocks) == 1 and tool_calls:
                # Priority 3: one text block with tool_calls -> text as reasoning
                reasoning = ""
                text = text_blocks[0].get("text", "")
            elif len(text_blocks) == 1:
                # Priority 4: one text block without tool_calls -> text as content
                text = text_blocks[0].get("text", "")
                reasoning = ""
            else:
                # Fallback: extract text from content
                text = _text_from(content)
                reasoning = ""

            _append_assistant(out_messages, text, reasoning, tool_calls)

    # ── Process final response as assistant turn ──────────────────────────────
    response = data.get("response", {})
    if response and response.get("role") == "assistant":
        resp_content = response.get("content", [])

        if isinstance(resp_content, list):
            # Extract thinking blocks first (highest priority for reasoning)
            thinking_content = _thinking_from(resp_content)

            # Extract text blocks
            text_blocks = [item for item in resp_content
                           if isinstance(item, dict) and item.get("type") == "text"]

            # Extract tool calls from response
            tool_calls = _tool_calls_from_uses(_tool_uses_from(resp_content))

            # Determine reasoning and text (same logic as assistant above)
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
                reasoning = ""
                text = text_blocks[0].get("text", "")
            elif len(text_blocks) == 1:
                text = text_blocks[0].get("text", "")
                reasoning = ""
            else:
                text = _text_from(resp_content)
                reasoning = ""

            _append_assistant(out_messages, text, reasoning, tool_calls)
        else:
            # Response content is not a list, treat as plain text
            text = _text_from(resp_content)
            _append_assistant(out_messages, text, "", [])

    # If the final response has no text, trim trailing messages after the last
    # assistant turn that has real content.
    response_text = _text_from(
        [item for item in (response.get("content", []) if isinstance(response, dict) else [])
         if isinstance(item, dict) and item.get("type") == "text"]
    ) if response else ""
    truncated = []
    if not response_text.strip():
        # find the last assistant entry with non-empty text content
        last_asst_idx = -1
        for idx, m in enumerate(out_messages):
            if m["role"] == "assistant" and m.get("content", "").strip():
                last_asst_idx = idx
        if last_asst_idx >= 0:
            truncated   = out_messages[last_asst_idx + 1:]
            out_messages = out_messages[: last_asst_idx + 1]

    # Ensure the last message is always role: assistant
    while out_messages and out_messages[-1]["role"] != "assistant":
        out_messages.pop()

    return _sanitize_messages(out_messages), truncated, convert_tools(raw_tools), model_name, has_fold


# ──────────────────────────────────────────────────────────────────────────────
# 6. Statistics
# ──────────────────────────────────────────────────────────────────────────────

def calc_stats(messages: list, response: dict) -> dict:
    tool_num      = sum(1 for m in messages if m["role"] == "tool")
    assistant_num = sum(1 for m in messages if m["role"] == "assistant")
    user_num      = sum(1 for m in messages if m["role"] == "user")
    rounds        = assistant_num  # 统一使用 assistant 轮数
    total_chars   = sum(len(m.get("content") or "") for m in messages)

    # Use actual token count from response.usage when available
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


# ──────────────────────────────────────────────────────────────────────────────
# 7. Timestamp helpers
# ──────────────────────────────────────────────────────────────────────────────

def ts_from_filename(path: str) -> str:
    """Extract date string (YYYY-MM-DD) from filename."""
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'(\d{4}-\d{2}-\d{2})[_T](\d{2})[-:](\d{2})[-:](\d{2})', name)
    if m:
        try:
            dt = datetime(int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:]),
                          int(m.group(2)), int(m.group(3)), int(m.group(4)))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return datetime.now().strftime("%Y-%m-%d")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Entry point
# ──────────────────────────────────────────────────────────────────────────────

def set_assistant_weight_zero(messages: list, up_to_round: int) -> list:
    """
    Set weight=0 for all assistant messages up to (and including) up_to_round.
    Returns a new messages list with weights set.
    """
    result = []
    assistant_count = 0
    for msg in messages:
        new_msg = dict(msg)
        if msg.get("role") == "assistant":
            assistant_count += 1
            if assistant_count <= up_to_round:
                new_msg["weight"] = 0
        result.append(new_msg)
    return result


def write_trajectory(messages: list, tools: list, model_name: str, time_str: str,
                     stats: dict, source_path: str, target_file: str,
                     owner: str, language: str, category: str):
    """Write a single trajectory to the target file."""
    output = {
        "version": "2.0.0",
        "messages": messages,
        "tools":    tools,
        "meta_info": {
            "teacher":                model_name,
            "query_source":           "synthesized",
            "response_generate_time": time_str,
            "response_update_time":   time_str,
            "owner":                  owner,
            "language":               language,
            "category":               category,
            "rounds":                 stats["rounds"],
            "unique_info": {
                "path":     source_path,
                "is_skill": True,
                "info": {
                    "complete_result":  1,
                    "correct_result":   1,
                    "number_of_turns":  stats["rounds"],
                    "tokens":           stats["tokens"],
                    "rounds":           stats["rounds"],
                    "tool_num":         stats["tool_num"],
                    "assistant_num":    stats["assistant_num"],
                },
            },
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def write_fold_trajectory(messages: list, tools: list, model_name: str, time_str: str,
                          stats: dict, source_path: str, fold_file: str,
                          owner: str, language: str, category: str, skip_rounds: int):
    """
    Write a trajectory with fold info to a separate fold file.

    Args:
        skip_rounds: Number of rounds from previous trajectories that should not be
                     trained again (weight=0). This is the accumulated rounds count.
    """
    output = {
        "version": "2.0.0",
        "messages": messages,
        "tools":    tools,
        "meta_info": {
            "teacher":                model_name,
            "query_source":           "synthesized",
            "response_generate_time": time_str,
            "response_update_time":   time_str,
            "owner":                  owner,
            "language":               language,
            "category":               category,
            "rounds":                 stats["rounds"],
            "unique_info": {
                "path":     source_path,
                "is_skill": True,
                "info": {
                    "complete_result":  1,
                    "correct_result":   1,
                    "number_of_turns":  stats["rounds"],
                    "tokens":           stats["tokens"],
                    "rounds":           stats["rounds"],
                    "tool_num":         stats["tool_num"],
                    "assistant_num":    stats["assistant_num"],
                    "skip_rounds":      skip_rounds,  # 前面已处理轨迹的总轮次，这些轮次weight=0
                },
            },
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(fold_file)), exist_ok=True)
    with open(fold_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def process_dir(subdir: str, target_file: str, truncated_file: str, fold_file: str,
                owner: str, language: str, category: str) -> bool:
    """Process the latest JSON in subdir. Returns True on success."""
    try:
        latest = find_latest_json(subdir)
    except FileNotFoundError:
        print(f"    [skip] no JSON files in {subdir}")
        return False

    print(f"[+] {subdir}")
    print(f"    latest file: {os.path.basename(latest)}")

    # Check if latest file has fold
    with open(latest, encoding="utf-8") as f:
        latest_data = json.load(f)

    if not has_tool_fold(latest_data):
        # No fold: process normally (original logic)
        messages, truncated, tools, model_name, _ = convert(latest_data, skip_fold_check=True)

        if model_name is None:
            print(f"    [skip] conversion failed")
            return False

        stats = calc_stats(messages, latest_data.get("response", {}))

        time_str = ts_from_filename(latest)

        output = {
            "version": "2.0.0",
            "messages": messages,
            "tools":    tools,
            "meta_info": {
                "teacher":                model_name,
                "query_source":           "synthesized",
                "response_generate_time": time_str,
                "response_update_time":   time_str,
                "owner":                  owner,
                "language":               language,
                "category":               category,
                "rounds":                 stats["rounds"],
                "unique_info": {
                    "path":     latest,
                    "is_skill": True,
                    "info": {
                        "complete_result":  1,
                        "correct_result":   1,
                        "number_of_turns":  stats["rounds"],
                        "tokens":           stats["tokens"],
                        "rounds":           stats["rounds"],
                        "tool_num":         stats["tool_num"],
                        "assistant_num":    stats["assistant_num"],
                    },
                },
            },
        }

        if truncated:
            full_output = dict(output)
            full_msgs = messages + truncated
            while full_msgs and full_msgs[-1]["role"] != "assistant":
                full_msgs.pop()
            full_output["messages"] = full_msgs
            os.makedirs(os.path.dirname(os.path.abspath(truncated_file)), exist_ok=True)
            with open(truncated_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(full_output, ensure_ascii=False) + "\n")
            print(f"    [!] incomplete -> {truncated_file}")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
            with open(target_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
            print(f"    [ok] -> {target_file}")

        print(f"    model={model_name}  assistant={stats['assistant_num']}  "
              f"tool={stats['tool_num']}  rounds={stats['rounds']}  tokens={stats['tokens']}")
        return True

    # Has fold: need to process all files in order
    print(f"    [fold detected] processing all files in order...")
    all_files = find_all_json_sorted(subdir)
    if not all_files:
        print(f"    [skip] no JSON files found")
        return False

    # skip_rounds: rounds from the last written trajectory (for setting weight=0)
    skip_rounds = 0
    # prev_fold_count: number of folds in previous file (to detect new folds)
    prev_fold_count = 0
    # Store last valid (non-fold) file info
    last_valid_messages = None
    last_valid_tools = None
    last_valid_model = None
    last_valid_stats = None
    last_valid_time_str = None
    last_valid_path = None
    total_written = 0

    for i, json_path in enumerate(all_files):
        is_last = (i == len(all_files) - 1)

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        current_fold_count = count_tool_folds(data)
        # Detect new fold: fold count increased compared to previous file
        has_new_fold = current_fold_count > prev_fold_count

        if has_new_fold:
            # This file has new fold (fold count increased)
            # First, write the stored valid file if exists
            if last_valid_messages is not None:
                # Write with current skip_rounds
                messages_with_weight = set_assistant_weight_zero(last_valid_messages, skip_rounds)
                write_fold_trajectory(
                    messages_with_weight, last_valid_tools, last_valid_model,
                    last_valid_time_str, last_valid_stats, last_valid_path,
                    fold_file, owner, language, category, skip_rounds
                )
                total_written += 1
                print(f"    [fold] written {os.path.basename(last_valid_path)} "
                      f"(skip_rounds={skip_rounds}, folds={prev_fold_count}->{current_fold_count}) -> {fold_file}")
                # Update skip_rounds to this trajectory's rounds
                skip_rounds = last_valid_stats["rounds"]
                # Clear stored valid file
                last_valid_messages = None
                last_valid_tools = None
                last_valid_model = None
                last_valid_stats = None
                last_valid_time_str = None
                last_valid_path = None

            # Update prev_fold_count
            prev_fold_count = current_fold_count

            # If this is the last file, write it
            if is_last:
                messages, truncated, tools, model_name, _ = convert(data, skip_fold_check=True)
                if model_name is None or not messages:
                    print(f"    [skip] last file conversion failed")
                    continue

                stats = calc_stats(messages, data.get("response", {}))

                time_str = ts_from_filename(json_path)
                # Ensure last message is assistant
                while messages and messages[-1]["role"] != "assistant":
                    messages.pop()

                messages_with_weight = set_assistant_weight_zero(messages, skip_rounds)
                write_fold_trajectory(
                    messages_with_weight, tools, model_name,
                    time_str, stats, json_path,
                    fold_file, owner, language, category, skip_rounds
                )
                total_written += 1
                print(f"    [fold] written last file {os.path.basename(json_path)} "
                      f"(skip_rounds={skip_rounds}) -> {fold_file}")
        else:
            # No new fold in this file - store it (overwrite previous)
            messages, truncated, tools, model_name, _ = convert(data, skip_fold_check=True)

            if model_name is None or not messages:
                prev_fold_count = current_fold_count
                continue

            stats = calc_stats(messages, data.get("response", {}))

            time_str = ts_from_filename(json_path)

            # Ensure last message is assistant
            while messages and messages[-1]["role"] != "assistant":
                messages.pop()

            # Store this valid file (will be written when next fold appears or at the end)
            last_valid_messages = messages
            last_valid_tools = tools
            last_valid_model = model_name
            last_valid_stats = stats
            last_valid_time_str = time_str
            last_valid_path = json_path

            # Update prev_fold_count
            prev_fold_count = current_fold_count

            # If this is the last file, write it
            if is_last:
                messages_with_weight = set_assistant_weight_zero(last_valid_messages, skip_rounds)
                write_fold_trajectory(
                    messages_with_weight, last_valid_tools, last_valid_model,
                    last_valid_time_str, last_valid_stats, last_valid_path,
                    fold_file, owner, language, category, skip_rounds
                )
                total_written += 1
                print(f"    [ok] written last file {os.path.basename(last_valid_path)} "
                      f"(skip_rounds={skip_rounds}) -> {fold_file}")

    if total_written > 0:
        print(f"    [done] {total_written} trajectory(ies) written to fold file")
        return True
    else:
        print(f"    [skip] no valid trajectories")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert conversation logs to training-data JSONL."
    )
    parser.add_argument("source_dir",   help="Base directory containing session subdirectories")
    parser.add_argument("target_file",  help="Output JSONL file for complete conversations")
    parser.add_argument("--session-list", default=None,
                        help="Text file listing session subdirectory names (one per line); "
                             "if omitted, all subdirectories of source_dir are processed")
    parser.add_argument("--owner",          default="00935640", help="Owner ID (default: 00935640)")
    parser.add_argument("--language",       default="zh",       help="Language code (default: zh)")
    parser.add_argument("--category",       default="agent",    help="Category (default: agent)")
    parser.add_argument("--truncated-file", default=None,
                        help="Output JSONL for incomplete conversations "
                             "(default: <target_file stem>_truncated.jsonl)")
    parser.add_argument("--fold-file", default=None,
                        help="Output JSONL for conversations with tool fold "
                             "(default: <target_file stem>_fold.jsonl)")
    args = parser.parse_args()

    if not args.truncated_file:
        base, ext = os.path.splitext(os.path.abspath(args.target_file))
        trunc_file = base + "_truncated" + (ext or ".jsonl")
    else:
        trunc_file = args.truncated_file

    if not args.fold_file:
        base, ext = os.path.splitext(os.path.abspath(args.target_file))
        fold_file = base + "_fold" + (ext or ".jsonl")
    else:
        fold_file = args.fold_file

    if not os.path.isdir(args.source_dir):
        print(f"[error] source directory not found: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    # Build session directory list
    if args.session_list:
        with open(args.session_list, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        entries = [os.path.join(args.source_dir, name) for name in names]
    else:
        try:
            entries = sorted(
                (e.path for e in os.scandir(args.source_dir) if e.is_dir()),
                key=lambda p: os.path.basename(p)
            )
        except FileNotFoundError:
            print(f"[error] source directory not found: {args.source_dir}", file=sys.stderr)
            sys.exit(1)

    if not entries:
        print(f"[error] no session directories found", file=sys.stderr)
        sys.exit(1)

    ok = skip = 0
    for subdir in entries:
        success = process_dir(subdir, args.target_file, trunc_file, fold_file,
                              args.owner, args.language, args.category)
        if success:
            ok += 1
        else:
            skip += 1

    print(f"\n[done] {ok} processed, {skip} skipped")


if __name__ == "__main__":
    main()