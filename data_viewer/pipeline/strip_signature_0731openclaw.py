#!/usr/bin/env python3
"""
从 0731openclaw 复原后的文件中移除 signature 等诊断字段，生成清洁版 JSONL。

如果 reflection_quality / length_diff / reasoning_content_reflected 有一条表示
invalid，则用 thinking_summary 替代 reasoning_content，并设 weight=0；
全合法则 weight=1。

用法:
    # 零参数运行（使用写死的默认路径）
    python cli/strip_signature_0731openclaw.py

    # 指定输入输出
    python cli/strip_signature_0731openclaw.py \
        --input pgml_and_pgml2_data/0731openclaw_cot_pgml2/converted_reflected.jsonl \
        --output pgml_and_pgml2_data/0731openclaw_cot_pgml2/converted_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = "pgml_and_pgml2_data/0731openclaw_cot_pgml2/converted_reflected.jsonl"
DEFAULT_OUTPUT = "pgml_and_pgml2_data/0731openclaw_cot_pgml2/converted_clean.jsonl"

# 需要移除的顶层字段
STRIP_TOP_FIELDS = {"_sample_idx"}

# 每个 message 中需要移除的诊断字段
STRIP_MSG_FIELDS = {
    "signature",
    "signature_info",
    "reasoning_content_reflected",
    "reflection_quality",
    "length_diff",
    "thinking_summary",
}


def _strip_signatures_recursive(obj: Any) -> Any:
    """递归移除所有嵌套层级中的 signature 字段。"""
    if isinstance(obj, dict):
        return {k: _strip_signatures_recursive(v) for k, v in obj.items() if k != "signature"}
    if isinstance(obj, list):
        return [_strip_signatures_recursive(item) for item in obj]
    return obj


def _is_reflection_valid(msg: dict[str, Any]) -> bool:
    """reflection 是否合法：三项都满足才算 valid。"""
    if not msg.get("reasoning_content_reflected"):
        return False
    if msg.get("reflection_quality", "") != "good":
        return False
    if msg.get("length_diff", "") != "valid":
        return False
    return True


def strip_sample(obj: dict[str, Any]) -> dict[str, Any]:
    """处理样本：评估每条消息的 reflection 质量，打 weight，fallback invalid 的 reasoning_content。"""
    # 处理 messages
    new_messages: list[dict[str, Any]] = []
    for msg in obj.get("messages") or []:
        new_msg = dict(msg)

        if msg.get("reasoning_content_reflected"):
            if _is_reflection_valid(msg):
                new_msg["weight"] = 1
            else:
                # fallback: 用 thinking_summary 作为 reasoning_content
                new_msg["reasoning_content"] = msg.get("thinking_summary", "")
                new_msg["weight"] = 0

        # 移除诊断字段
        for field in STRIP_MSG_FIELDS:
            new_msg.pop(field, None)

        new_messages.append(new_msg)

    # 移除顶层冗余字段
    clean = {k: v for k, v in obj.items() if k not in STRIP_TOP_FIELDS}
    clean["messages"] = new_messages

    # 递归清除嵌套 signature
    return _strip_signatures_recursive(clean)


def _count_signatures(obj: Any) -> int:
    """递归统计所有嵌套层级中的 signature 数量。"""
    if isinstance(obj, dict):
        count = 0
        for k, v in obj.items():
            if k == "signature":
                count += 1
            count += _count_signatures(v)
        return count
    if isinstance(obj, list):
        return sum(_count_signatures(item) for item in obj)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 0731openclaw 复原后文件中移除 signature 等冗余字段，打 weight",
    )
    parser.add_argument(
        "--input", type=Path,
        default=BASE_DIR / DEFAULT_INPUT,
        help=f"输入的反射后 JSONL 文件 (默认: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path,
        default=BASE_DIR / DEFAULT_OUTPUT,
        help=f"输出的清洁版 JSONL (默认: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    input_file = args.input if args.input.is_absolute() else BASE_DIR / args.input
    output_file = args.output if args.output.is_absolute() else BASE_DIR / args.output

    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    stripped_sigs = 0
    weight1 = 0
    weight0 = 0
    out_lines: list[str] = []

    for line in input_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        obj = json.loads(line)
        stripped_sigs += _count_signatures(obj)

        clean_obj = strip_sample(obj)

        # 统计 weight 分布
        for msg in clean_obj.get("messages") or []:
            w = msg.get("weight")
            if w == 1:
                weight1 += 1
            elif w == 0:
                weight0 += 1

        out_lines.append(json.dumps(clean_obj, ensure_ascii=False))

    output_file.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    total_weighted = weight1 + weight0
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print(f"处理: {total} 条样本, 移除 {stripped_sigs} 个 signature")
    print(f"weight 分布: weight=1: {weight1} ({weight1/total_weighted*100:.1f}%)  "
          f"weight=0: {weight0} ({weight0/total_weighted*100:.1f}%)")
    if stripped_sigs > 0:
        avg_sig_len = 200
        saving = stripped_sigs * avg_sig_len
        print(f"估算节省: ~{saving:,} 字符 ({saving / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
