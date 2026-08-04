#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整 CoT 复原流水线：
  trajectory.jsonl / pgml2.jsonl → (转换) → reflect → retry → strip → clean.jsonl

用法:
    python run_cot_pipeline.py \\
        --input  <file.jsonl 或 file.trajectory.jsonl> \\
        --api-key  sk-xxx \\
        --base-url http://115.120.113.66:8082 \\
        --model    tokenfly-01/claude-opus-4.8 \\
        [--output-dir ./out]   # 默认与 --input 同目录
        [--workers 8]          # 默认 8
        [--max-retries 3]      # retry 阶段最多重试几次，默认 3

输出（全部在 --output-dir 下）：
    <stem>_converted.jsonl     仅 trajectory 输入时生成
    <stem>_reflected.jsonl     reflect + retry 后的结果
    <stem>_clean.jsonl         最终干净训练文件  ← 主要产物
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── 动态加载本目录下的三个脚本模块 ──────────────────────────────────────────
def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_api(mod, api_key: str, base_url: str, model: str):
    mod.API_KEY  = api_key
    mod.BASE_URL = base_url.rstrip("/")
    mod.MODEL    = model


# ── 格式检测 ─────────────────────────────────────────────────────────────────
def _is_trajectory(path: Path) -> bool:
    """检查文件是否为 trajectory.jsonl（含 session.started 事件）。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "session.started":
                return True
            # 只检查前 20 行
            break
    # 文件名包含 .trajectory 也算
    return ".trajectory" in path.name


# ── 步骤 0：trajectory → pgml2 转换 ─────────────────────────────────────────
def step_convert(traj_file: Path, out_dir: Path) -> Path:
    """调用 workspace_to_pangu/traj_to_converted.py，返回 converted.jsonl 路径。"""
    script = HERE / "workspace_to_pangu" / "traj_to_converted.py"
    if not script.exists():
        raise FileNotFoundError(
            f"找不到转换脚本: {script}\n"
            "请先解压 workspace_to_pangu.zip 到当前目录。"
        )

    # traj_to_converted 需要 agents/main/sessions/ 目录结构
    tmpdir = Path(tempfile.mkdtemp(prefix="cot_traj_"))
    sessions_dir = tmpdir / "agents" / "main" / "sessions"
    sessions_dir.mkdir(parents=True)
    shutil.copy2(traj_file, sessions_dir / traj_file.name)

    stem = traj_file.name.replace(".trajectory.jsonl", "").replace(".jsonl", "")
    converted = out_dir / f"{stem}_converted.jsonl"

    spec = importlib.util.spec_from_file_location("traj_to_converted", script)
    mod = importlib.util.module_from_spec(spec)
    # 脚本在 __main__ 保护下运行；直接调用其 public 函数
    spec.loader.exec_module(mod)

    # traj_to_converted 的入口是 main()，但需要 sys.argv
    old_argv = sys.argv[:]
    sys.argv = [
        str(script),
        "--traj-in", str(tmpdir),
        "--out",     str(converted),
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not converted.exists():
        raise RuntimeError(f"转换失败，未生成: {converted}")
    return converted


# ── 步骤 1：reflect ──────────────────────────────────────────────────────────
def step_reflect(
    input_file: Path,
    output_file: Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    workers: int,
) -> None:
    mod = _load("reflect_0731openclaw_cot")
    _set_api(mod, api_key, base_url, model)
    mod.run(
        input_file=input_file,
        output_file=output_file,
        workers=workers,
        limit=None,
        flush_interval=50,
        max_tokens=32768,
        stream=False,
    )


# ── 步骤 2：retry ────────────────────────────────────────────────────────────
def step_retry(
    io_file: Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    workers: int,
    max_content_retries: int,
) -> None:
    mod = _load("retry_suspicious_cot")
    _set_api(mod, api_key, base_url, model)
    mod.MAX_CONTENT_RETRIES = max_content_retries

    old_argv = sys.argv[:]
    sys.argv = [
        "retry_suspicious_cot.py",
        "--input",   str(io_file),
        "--output",  str(io_file),
        "--workers", str(workers),
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv


# ── 步骤 3：strip ────────────────────────────────────────────────────────────
def step_strip(input_file: Path, output_file: Path) -> None:
    mod = _load("strip_signature_0731openclaw")
    old_argv = sys.argv[:]
    sys.argv = [
        "strip_signature_0731openclaw.py",
        "--input",  str(input_file),
        "--output", str(output_file),
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv


# ── 主入口 ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="完整 CoT 复原流水线（trajectory / pgml2 → clean.jsonl）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",       required=True,  type=Path, help="输入文件（.trajectory.jsonl 或 pgml2 .jsonl）")
    parser.add_argument("--api-key",     required=True,  help="API key")
    parser.add_argument("--base-url",    required=True,  help="API base URL")
    parser.add_argument("--model",       required=True,  help="模型名称")
    parser.add_argument("--output-dir",  type=Path, default=None, help="输出目录（默认与输入文件同目录）")
    parser.add_argument("--workers",     type=int,  default=8,    help="并发线程数（默认 8）")
    parser.add_argument("--max-retries", type=int,  default=3,    help="retry 阶段内容重试次数（默认 3）")
    args = parser.parse_args()

    input_file: Path = args.input.resolve()
    if not input_file.exists():
        sys.exit(f"[ERROR] 输入文件不存在: {input_file}")

    out_dir: Path = (args.output_dir or input_file.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 去掉 .trajectory.jsonl / .jsonl 后缀作为 stem
    stem = input_file.name
    for suffix in (".trajectory.jsonl", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  输入:      {input_file}")
    print(f"  输出目录:  {out_dir}")
    print(f"  模型:      {args.model}")
    print(f"  并发:      {args.workers}")
    print(f"{'='*60}\n")

    # ── 步骤 0：格式检测 & 转换 ──────────────────────────────────────────────
    if _is_trajectory(input_file):
        print("[步骤 0] trajectory 格式 → 转换为 pgml2 …")
        pgml2_file = step_convert(input_file, out_dir)
        print(f"         → {pgml2_file.name}\n")
    else:
        pgml2_file = input_file
        print("[步骤 0] pgml2 格式，跳过转换\n")

    # ── 步骤 1：reflect ───────────────────────────────────────────────────────
    reflected_file = out_dir / f"{stem}_reflected.jsonl"
    print(f"[步骤 1] reflect → {reflected_file.name}")
    step_reflect(
        pgml2_file, reflected_file,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        workers=args.workers,
    )
    print()

    # ── 步骤 2：retry ─────────────────────────────────────────────────────────
    print(f"[步骤 2] retry  → {reflected_file.name}（原地覆盖）")
    step_retry(
        reflected_file,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        workers=args.workers,
        max_content_retries=args.max_retries,
    )
    print()

    # ── 步骤 3：strip ─────────────────────────────────────────────────────────
    clean_file = out_dir / f"{stem}_clean.jsonl"
    print(f"[步骤 3] strip  → {clean_file.name}")
    step_strip(reflected_file, clean_file)
    print()

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    # 统计 weight 分布
    w1 = w0 = 0
    with open(clean_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for msg in obj.get("messages", []):
                w = msg.get("weight")
                if w == 1:
                    w1 += 1
                elif w == 0:
                    w0 += 1

    total = w1 + w0
    print(f"{'='*60}")
    print(f"  完成！耗时 {elapsed:.0f}s")
    print(f"  最终文件: {clean_file}")
    print(f"  weight=1 (完整 CoT): {w1}/{total} ({w1/total*100:.1f}%)" if total else "  无带权重的消息")
    print(f"  weight=0 (thinking_summary 回退): {w0}/{total}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
