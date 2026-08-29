#!/usr/bin/env python3
"""对轨迹中的 thinkingSignature / signature 解码，校验解出的元数据是否含 "claude"。

规则：base64 解码 signature，得到 protobuf 内层 field[1] 元数据（含 model name）。
若解码后的可读文本中不含关键字 "claude"，判定该 signature 不符合规则
（说明并非真正的 Anthropic/claude 推理节点产出）。

用法:
    # 扫描指定 task 目录（origin 下所有 session），采样 N 个
    python check_signature_claude.py --task pipeline_output/tasks/<task_dir> --limit 50

    # 直接扫描单个轨迹 jsonl
    python check_signature_claude.py --file path/to/session.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KEYWORD = "claude"


def decode_info(sig: str) -> str | None:
    """base64 解码 signature，返回可读 ASCII（不可见字符用 . 替代）。

    解码失败（如含非 ASCII / 省略号截断等非法 base64）返回 None，
    与"解出正常文本但不含 claude"区分开。
    """
    try:
        raw = base64.b64decode(sig)
        return "".join(chr(b) if 32 <= b < 127 else "." for b in raw)
    except Exception:
        return None


def extract_model_hint(info: str) -> str:
    """从可读文本里截取 model name 附近的片段用于展示。"""
    low = info.lower()
    idx = low.find("claude")
    if idx < 0:
        # 尝试常见其它模型名
        for kw in ("glm", "gpt", "qwen", "gemini", "deepseek"):
            j = low.find(kw)
            if j >= 0:
                idx = j
                break
    if idx < 0:
        return info[-40:]
    return info[idx: idx + 40]


def iter_signatures(jsonl_path: Path):
    """遍历一个轨迹 jsonl，yield 每个 assistant thinking 的 signature 字符串。"""
    try:
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("type") != "message":
                    continue
                m = o.get("message") or {}
                if m.get("role") != "assistant":
                    continue
                for p in m.get("content") or []:
                    if not isinstance(p, dict):
                        continue
                    sig = p.get("thinkingSignature") or p.get("signature")
                    if sig:
                        yield sig
    except OSError:
        return


def check_file(jsonl_path: Path) -> dict:
    total = ok = mismatch = corrupt = 0
    bad_samples: list[str] = []
    for sig in iter_signatures(jsonl_path):
        total += 1
        info = decode_info(sig)
        if info is None:
            corrupt += 1  # base64 解码失败：签名损坏/截断
            continue
        if KEYWORD in info.lower():
            ok += 1
        else:
            mismatch += 1  # 解码成功但不含 claude：疑似非 claude 模型
            if len(bad_samples) < 3:
                bad_samples.append(extract_model_hint(info))
    return {
        "total": total,
        "ok": ok,
        "mismatch": mismatch,
        "corrupt": corrupt,
        "bad": mismatch + corrupt,
        "bad_samples": bad_samples,
    }


def find_traj_files(session_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pat in ("agents/*/sessions/*.jsonl", "agents/*/*/sessions/*.jsonl"):
        files.extend(session_dir.glob(pat))
    return [f for f in files if "trajectory" not in f.name]


def main() -> None:
    ap = argparse.ArgumentParser(description="校验 signature 解码后是否含 claude 关键字")
    ap.add_argument("--task", type=Path, help="task 目录（内含 origin/<session>/...）")
    ap.add_argument("--file", type=Path, help="直接指定单个轨迹 jsonl")
    ap.add_argument("--limit", type=int, default=50, help="task 模式下最多采样多少个 session")
    args = ap.parse_args()

    if args.file:
        r = check_file(args.file)
        verdict = "合规" if r["bad"] == 0 else "存在违规"
        print(f"[{verdict}] {args.file.name}: signature={r['total']} "
              f"合规={r['ok']} 模型不符={r['mismatch']} 签名损坏={r['corrupt']}")
        for s in r["bad_samples"]:
            print(f"    模型不符样例: {s!r}")
        return

    if not args.task:
        ap.error("需指定 --task 或 --file")

    task_dir = args.task if args.task.is_absolute() else BASE_DIR / args.task
    origin = task_dir / "origin"
    if not origin.is_dir():
        raise SystemExit(f"origin 目录不存在: {origin}")

    sessions = sorted([d for d in origin.iterdir() if d.is_dir()])[: args.limit]
    print(f"任务: {task_dir.name}  采样 {len(sessions)} 个 session\n")

    agg_total = agg_ok = agg_mismatch = agg_corrupt = 0
    sessions_with_sig = 0
    bad_sessions: list[tuple[str, dict]] = []

    for sess in sessions:
        trajs = find_traj_files(sess)
        if not trajs:
            continue
        r = check_file(trajs[0])
        if r["total"] == 0:
            continue
        sessions_with_sig += 1
        agg_total += r["total"]
        agg_ok += r["ok"]
        agg_mismatch += r["mismatch"]
        agg_corrupt += r["corrupt"]
        if r["bad"] > 0:
            bad_sessions.append((sess.name, r))

    print(f"含 signature 的 session: {sessions_with_sig}")
    print(f"signature 总数: {agg_total}  合规: {agg_ok}  "
          f"模型不符: {agg_mismatch}  签名损坏: {agg_corrupt}")
    if agg_total:
        print(f"合规率: {agg_ok / agg_total * 100:.1f}%")

    if bad_sessions:
        print(f"\n== 存在问题的 session（共 {len(bad_sessions)}）==")
        for name, r in bad_sessions[:20]:
            print(f"  {name}: 模型不符 {r['mismatch']} / 签名损坏 {r['corrupt']} "
                  f"(共 {r['total']} 条)  样例={r['bad_samples']}")
    else:
        print("\n本次采样未发现问题 signature。")


if __name__ == "__main__":
    main()
