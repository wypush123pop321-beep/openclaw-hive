# -*- coding: utf-8 -*-
"""
把 traj_to_converted.py 转换出的轨迹(messages: system/user/assistant/tool, meta_info.unique_info.path
指向原始 .trajectory.jsonl)按路径直接关联到同一 task 下的 logs/<任务名>.log, 解析该 log 里逐轮
evaluator 裁决(completion + rubric_checks), 回填到 meta_info.unique_info.eval_info。

跟 fill_eval_info.py 的区别: 那个脚本是给 workspace(单独一批 profiles/assistant*/sessions 会话) 和
convert_log 转换后的轨迹做模糊的首轮 query + 内容指纹匹配, 因为两边目录结构不是一一对应的; 这里
traj_to_converted.py 输出的每一行都自带 meta_info.unique_info.path(原始 .trajectory.jsonl 的路径),
而这个路径本身就固定长在 <task_dir>/agents/<agent>/sessions/<file>.trajectory.jsonl 下, 从它往上数 4
层就是 task_dir, 其下 logs/<task_dir 的目录名>.log 就是对应的 evaluator 日志, 不需要做任何模糊匹配。

eval_info 是一个 list, 按 1..n_rounds 顺序对应轨迹里每一个 user 轮(每出现一次 role=="user" 算一轮);
第 i 个元素若 log 里有第 i 轮的裁决, 是 {"completion": <float>, "rubric_checks": <list/None>}, 否则是
null(不是 {"completion": None, "rubric_checks": None})。

输出是单个聚合了所有输入行的 .jsonl 文件, 不是按输入结构散落的一堆文件(即便 --traj-in 传入的
是一个目录、里面有多个 .jsonl 文件, 所有行也都会被写进同一个输出文件)。

本文件不依赖任何其他本地脚本(fill_eval_info.py / filter_by_workspace.py 等), 所需的 evaluator 裁决
解析逻辑已内联在下面, 可以单独拷走使用。

用法:
  python add_eval_info.py --traj-in "<traj_to_converted.py 的输出: 单个 .jsonl 或目录>" \
      --out "<输出的聚合 .jsonl 文件>"
"""
import os
import io
import re
import json
import argparse


# ── 阶段 0: 解析 log 里逐轮 evaluator 裁决(仿 filter_by_workspace.py / fill_eval_info.py) ──
_EVAL_MARK = re.compile(r"\[Evaluator\]\s+turn=(\d+)\s+agent=\S+.*输出")


def parse_all_verdicts_full(log_path):
    """解析 log 里所有轮次 evaluator 裁决, 返回按 turn 升序的
    [(turn, completion, inclination, rubric_checks)]。"""
    if not os.path.isfile(log_path):
        return []
    lines = io.open(log_path, encoding="utf-8", errors="replace").readlines()
    out = []
    for i, line in enumerate(lines):
        m = _EVAL_MARK.search(line)
        if not m:
            continue
        turn = int(m.group(1))
        j = i + 1
        while j < len(lines) and lines[j].strip() != "{":
            j += 1
        if j >= len(lines):
            out.append((turn, None, None, None))
            continue
        depth = 0
        buf = []
        for k in range(j, len(lines)):
            buf.append(lines[k])
            depth += lines[k].count("{") - lines[k].count("}")
            if depth <= 0:
                break
        try:
            obj = json.loads("".join(buf))
        except json.JSONDecodeError:
            out.append((turn, None, None, None))
            continue
        comp = obj.get("completion")
        comp = float(comp) if isinstance(comp, (int, float)) else None
        incl = obj.get("inclination")
        incl = incl if isinstance(incl, str) else None
        rubric_checks = obj.get("rubric_checks")
        out.append((turn, comp, incl, rubric_checks))
    out.sort(key=lambda x: x[0])
    return out


# ── 阶段 1: 从原始 .trajectory.jsonl 路径推出同一 task 下的 logs/<任务名>.log ──────────
def derive_log_path(traj_path):
    """traj_path 形如 <task_dir>/agents/<agent>/sessions/<file>.trajectory.jsonl,
    往上数 4 层(sessions -> agent -> agents -> task_dir)就是 task_dir。"""
    task_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(traj_path))))
    task_name = os.path.basename(task_dir)
    return os.path.join(task_dir, "logs", task_name + ".log")


def count_user_rounds(messages):
    return sum(1 for m in messages if m.get("role") == "user")


def build_eval_info(n_rounds, eval_by_turn):
    """按 1..n_rounds 顺序回填, 某轮没有裁决结果就是 null(不是 {"completion":None,...})。"""
    out = []
    for i in range(1, n_rounds + 1):
        if i in eval_by_turn:
            comp, rc = eval_by_turn[i]
            out.append({"completion": comp, "rubric_checks": rc})
        else:
            out.append(None)
    return out


# ── 阶段 2: 遍历 --traj-in, 按 meta_info.unique_info.path 直接关联 log 并回填 ──────────
def iter_jsonl_files(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for n in names:
            if n.endswith(".jsonl"):
                files.append(os.path.join(root, n))
    return sorted(files)


def process_file(path, fout, log_cache, indent=""):
    """把单个输入 .jsonl 文件的每一行(回填 eval_info 后)写进共享的聚合输出文件句柄 fout。"""
    n_total = n_filled = n_no_path = n_no_log = n_bad = 0
    with io.open(path, encoding="utf-8", errors="replace") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue

            messages = obj.get("messages", []) if isinstance(obj, dict) else []
            n_rounds = count_user_rounds(messages)
            src_path = ((obj.get("meta_info") or {}).get("unique_info") or {}).get("path")

            if not src_path:
                n_no_path += 1
                eval_info = [None] * n_rounds
            else:
                log_path = derive_log_path(src_path)
                if log_path not in log_cache:
                    verdicts = parse_all_verdicts_full(log_path)
                    log_cache[log_path] = {t: (comp, rc) for (t, comp, incl, rc) in verdicts}
                eval_by_turn = log_cache[log_path]
                if not eval_by_turn and not os.path.isfile(log_path):
                    n_no_log += 1
                else:
                    n_filled += 1
                eval_info = build_eval_info(n_rounds, eval_by_turn)

            obj.setdefault("meta_info", {}).setdefault("unique_info", {})["eval_info"] = eval_info
            out_line = json.dumps(obj, ensure_ascii=False)
            out_line = out_line.encode("utf-8", errors="replace").decode("utf-8")
            fout.write(out_line + "\n")

    print(f"{indent}[{os.path.basename(path)}] 共 {n_total} 条 | 关联到 log {n_filled} "
          f"| 缺 path {n_no_path} | log 文件不存在 {n_no_log} | 解析失败 {n_bad}")
    return {"file": path, "total": n_total, "filled": n_filled,
            "no_path": n_no_path, "no_log": n_no_log, "bad": n_bad}


def main():
    ap = argparse.ArgumentParser(
        description="按 meta_info.unique_info.path 直接关联同一 task 下的 logs/<任务名>.log, "
                    "把逐轮 evaluator 裁决(completion+rubric_checks)回填到 eval_info")
    ap.add_argument("--traj-in", required=True,
                    help="traj_to_converted.py 的输出: 单个 .jsonl 文件, 或包含若干 .jsonl 文件的目录")
    ap.add_argument("--out", required=True, help="输出的聚合 .jsonl 文件路径(单个文件, 不是目录)")
    a = ap.parse_args()

    if not os.path.exists(a.traj_in):
        ap.error(f"traj-in 不存在: {a.traj_in}")
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)

    files = iter_jsonl_files(a.traj_in)
    if not files:
        ap.error(f"traj-in 下未找到 .jsonl 文件: {a.traj_in}")
    print(f"[1] 处理 {len(files)} 个轨迹文件 -> {a.out}")

    log_cache = {}
    with io.open(a.out, "w", encoding="utf-8") as fout:
        stats = [process_file(f, fout, log_cache, indent="  ") for f in files]

    total = sum(s["total"] for s in stats)
    filled = sum(s["filled"] for s in stats)
    no_log = sum(s["no_log"] for s in stats)
    summary = {
        "traj_in": a.traj_in, "out": a.out,
        "log_files_seen": len(log_cache),
        "traj_total": total, "traj_filled": filled, "traj_no_log": no_log,
        "files": stats,
    }
    summary_path = a.out + ".stats.json"
    with io.open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] 共 {total} 条轨迹 | 关联到 log {filled} | log 文件缺失 {no_log} "
          f"| 涉及 {len(log_cache)} 个不同 log 文件 -> {a.out} (统计: {summary_path})")


if __name__ == "__main__":
    main()
