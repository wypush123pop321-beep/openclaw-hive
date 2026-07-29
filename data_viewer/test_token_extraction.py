#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 token 提取功能"""
import json
from pathlib import Path

def extract_tokens_from_jsonl(jsonl_path: Path):
    """从 JSONL 文件中提取 token 使用信息"""
    if not jsonl_path.exists():
        return None

    total_input = 0
    total_output = 0
    total_reasoning = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get('type') != 'message':
                continue

            msg = obj.get('message', {})
            if msg.get('role') == 'assistant':
                usage = msg.get('usage')
                if usage and isinstance(usage, dict):
                    total_input += usage.get('input', 0)
                    total_output += usage.get('output', 0)
                    total_reasoning += usage.get('reasoningTokens', 0)

    if total_input > 0 or total_output > 0:
        return {
            'input_tokens': total_input,
            'output_tokens': total_output,
            'reasoning_tokens': total_reasoning,
            'total_tokens': total_input + total_output,
        }
    return None

# 测试
test_file = Path('pipeline_output/tasks/t_a3d113e2/origin/00002_投资助手_财报数值核对_5bf36fe3_q1/agents/assistant1/sessions/fe2487ac-3b0e-4d5b-bf2e-1b12b8345c10.jsonl')

if test_file.exists():
    result = extract_tokens_from_jsonl(test_file)
    if result:
        print('✓ Token usage extracted successfully:')
        print(f"  Input tokens: {result['input_tokens']:,}")
        print(f"  Output tokens: {result['output_tokens']:,}")
        print(f"  Reasoning tokens: {result['reasoning_tokens']:,}")
        print(f"  Total tokens: {result['total_tokens']:,}")
    else:
        print('✗ No token usage found')
else:
    print(f'✗ Test file not found: {test_file}')
