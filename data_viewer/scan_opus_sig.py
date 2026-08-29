"""扫描 opus4.8 分组任务的被测 assistant 轨迹 signature 合规性。

口径与 viewer 完全一致: 只看 agents/assistant*/ 或 agents/main/ 的被测轨迹,
绝不碰 evaluator(裁判用别的模型, 签名本就不含 claude)。
"""
import json
import sys
from pathlib import Path

import server

TASKS = [
    "t_18e1be54", "t_cad9e497", "t_972054de", "t_4a554e5a", "t_13f146a8", "t_4b4cfbc0",
    "t_77aca9a2", "t_8698e68a", "t_9435d285", "t_5ab19a84", "t_3aeb8cfe", "t_f56ed76d",
    "t_5672f36d", "t_47f4c240", "t_2d06ac84", "t_34be5dee", "t_1f6a9981", "t_a7cab7a2",
]


def assistant_jsonl(task_dir: Path):
    """同后端 _load_workspace_assistant: assistant* 优先, 再 main, 排除 trajectory/evaluator。"""
    for pattern in ("agents/assistant*/sessions/*.jsonl", "agents/main/sessions/*.jsonl"):
        for jsonl in sorted(task_dir.glob(pattern)):
            if "trajectory" in jsonl.name:
                continue
            return jsonl
    return None


def iter_sigs(fp: Path):
    with open(fp, encoding="utf-8", errors="replace") as f:
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
                if isinstance(p, dict):
                    sig = p.get("thinkingSignature") or p.get("signature")
                    if sig:
                        yield sig


def main():
    print(f"{'任务':40} {'带sig会话':>8} {'sig总':>7} {'合规':>7} {'不符':>6} {'损坏':>6}", flush=True)
    mismatch_report = []
    for tid in TASKS:
        t = server.find_task(tid)
        name = (t.get("name") or "")[:38]
        origin = Path(t["output_dir"]) / "origin"
        if not origin.is_dir():
            print(f"{name:40} {'(无origin)':>8}", flush=True)
            continue
        nsess = tot = ok = mis = cor = 0
        mism = []
        for d in sorted(origin.iterdir()):
            if not d.is_dir():
                continue
            tdir = server._workspace_task_dir(d.name, t["output_dir"]) or d
            fp = assistant_jsonl(tdir)
            if not fp:
                continue
            s_tot = s_mis = 0
            for sig in iter_sigs(fp):
                tot += 1
                s_tot += 1
                v = server._check_signature_claude(sig)["verdict"]
                if v == "ok":
                    ok += 1
                elif v == "mismatch":
                    mis += 1
                    s_mis += 1
                else:
                    cor += 1
            if s_tot:
                nsess += 1
            if s_mis:
                mism.append((d.name, s_mis, s_tot))
        print(f"{name:40} {nsess:>8} {tot:>7} {ok:>7} {mis:>6} {cor:>6}", flush=True)
        if mism:
            mismatch_report.append((name, mism))

    print("\n===== 【模型不符】(真·不合规: 被测 assistant 轨迹解码不含 claude) =====", flush=True)
    if not mismatch_report:
        print("无 —— 所有 opus4.8 分组的被测 assistant 轨迹, signature 解码后均含 claude。", flush=True)
    else:
        for name, mism in mismatch_report:
            print(f"\n【{name}】 {len(mism)} 个 session 含不合规签名:", flush=True)
            for sname, sm, st in mism[:12]:
                print(f"    {sname[:52]}: 不符 {sm}/{st}", flush=True)
            if len(mism) > 12:
                print(f"    ... 还有 {len(mism) - 12} 个", flush=True)


if __name__ == "__main__":
    main()
