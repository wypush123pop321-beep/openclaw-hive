#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键跑「转换 -> 回填 eval_info -> 三层规则过滤」流程：

    traj_to_converted.py（原始 *.trajectory.jsonl 目录 -> converted.jsonl/_fold/_truncated）
    -> add_eval_info.py（按 meta_info.unique_info.path 关联 logs/<task>.log，回填逐轮
       evaluator 裁决到 meta_info.unique_info.eval_info）
    -> filter_trajectories.py（工具越界 / 参数不合法 / signature 不可信 三层规则过滤，
       全部不命中的轨迹落进 out-dir/filtered*/）

转换阶段产出 converted.jsonl(主) 和 converted_fold.jsonl(发生过 tool fold 的对话) 两路,
后两段(回填 eval_info + 三层规则过滤)对这两路分别各跑一遍, 互不混合、各自出结果
(fold 那路的中间/最终文件名都带 _fold 后缀, 见下面输出列表); converted_truncated.jsonl
(结尾不完整的对话)不参与后两段, 仅供人工核查。

三段都通过 subprocess 调用本目录下已有的 traj_to_converted.py / add_eval_info.py /
filter_trajectories.py，不重复实现逻辑，参数透传即可。

用法：
    python run_full_pipeline.py \
        --traj-in <原始轨迹根目录，含若干 task/agents/main/sessions/*.trajectory.jsonl> \
        --out-dir <输出根目录>

    # 已经有 traj_to_converted.py 转换好的 jsonl，只想跑 eval_info + 过滤：
    python run_full_pipeline.py --skip-convert \
        --converted-jsonl out\\converted.jsonl --out-dir out

    # 转换 + 过滤，不需要 eval_info（filter_trajectories 直接吃 converted(_fold).jsonl）：
    python run_full_pipeline.py --skip-eval --traj-in <...> --out-dir <...>

    # 只做转换 + 回填 eval_info，不跑过滤：
    python run_full_pipeline.py --skip-filter --traj-in <...> --out-dir <...>

    # 不想处理 fold 那路，只跑 converted.jsonl 主线：
    python run_full_pipeline.py --skip-fold --traj-in <...> --out-dir <...>

    # 批量模式：--batch-in 下每个子目录各是一批独立的原始轨迹任务（不要求属于同一批次），
    # 逐个跑完整流程，各自输出到 --out-dir\\<子目录名>\\ 下，并把所有子目录的 filter_stats.json
    # 汇总成 --out-dir\\batch_stats.json（总轨迹条数 + 规则通过条数）：
    python run_full_pipeline.py --batch-in <根目录，下面每个子目录是一批任务> --out-dir <输出根目录>

输出（落在 --out-dir 下，文件名与手动分步跑时一致；批量模式下每个子目录各有一份同样的产出，
外加一份跨子目录的汇总统计）：
    converted.jsonl                     转换阶段主输出（--skip-convert 时不生成）
    converted_fold.jsonl                转换阶段含 tool fold 的对话
    converted_truncated.jsonl           转换阶段判定为不完整的对话（不参与后两段）
    converted.jsonl.stats.json          转换阶段统计
    converted_with_eval.jsonl           主线回填 eval_info 后的结果（--skip-eval 时不生成）
    converted_fold_with_eval.jsonl      fold 那路回填 eval_info 后的结果（同上, --skip-fold 也不生成）
    filtered/<同名>.jsonl               主线过滤后最终保留的轨迹
    filtered/filter_stats.json          主线过滤统计
    filtered/rules/<规则名>/<同名>.jsonl 主线被过滤掉的轨迹, 按命中的规则分类(见 filter_trajectories.py)
    filtered_fold/<同名>.jsonl          fold 那路过滤后最终保留的轨迹
    filtered_fold/filter_stats.json     fold 那路过滤统计
    filtered_fold/rules/<规则名>/<同名>.jsonl  fold 那路被过滤掉的轨迹, 按命中的规则分类
    batch_stats.json                    仅批量模式：跨子目录汇总的轨迹总数/规则通过数
"""
import argparse
import io
import json
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJ_TO_CONVERTED = os.path.join(THIS_DIR, "traj_to_converted.py")
ADD_EVAL_INFO = os.path.join(THIS_DIR, "add_eval_info.py")
FILTER_TRAJECTORIES = os.path.join(THIS_DIR, "filter_trajectories.py")

# Windows 控制台默认编码常是 GBK，转发子进程的 utf-8 输出（含中文）时直接 write
# 会抛 UnicodeEncodeError，这里把标准流重配成 utf-8 + 容错替换。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def run_step(cmd, step_name):
    print("=" * 70)
    print(f"[{step_name}] {' '.join(cmd)}")
    print("=" * 70)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"[error] {step_name} 失败，退出码 {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)
    return proc.stdout


def do_convert(traj_in, converted_jsonl, args, step_label):
    cmd = [
        sys.executable, TRAJ_TO_CONVERTED,
        "--traj-in", traj_in,
        "--out", converted_jsonl,
        "--owner", args.owner,
        "--language", args.language,
        "--category", args.category,
    ]
    run_step(cmd, step_label)


def do_eval_info(converted_jsonl, with_eval_jsonl, step_label):
    cmd = [sys.executable, ADD_EVAL_INFO, "--traj-in", converted_jsonl, "--out", with_eval_jsonl]
    run_step(cmd, step_label)


def do_filter(traj_in, filtered_dir, step_label):
    cmd = [sys.executable, FILTER_TRAJECTORIES, "--traj-in", traj_in, "--out-dir", filtered_dir]
    run_step(cmd, step_label)


def process_track(label, converted_jsonl, out_dir, skip_eval, skip_filter, step_no, total_steps):
    """把一路 converted*.jsonl 依次跑 回填eval_info(可跳) -> 三层规则过滤(可跳)。
    label 为 "" 时是主线, 为 "fold" 时是 fold 那路(中间/最终文件名带 _fold 后缀)。
    返回 (with_eval_jsonl_or_None, filtered_dir_or_None, step_no)。"""
    suffix = f"_{label}" if label else ""
    tag = f"[{label}]" if label else "[main]"

    filter_input = converted_jsonl
    with_eval_jsonl = None
    if not skip_eval:
        with_eval_jsonl = os.path.join(out_dir, f"converted{suffix}_with_eval.jsonl")
        do_eval_info(converted_jsonl, with_eval_jsonl, f"{step_no}/{total_steps} {tag} 回填 add_eval_info")
        step_no += 1
        filter_input = with_eval_jsonl

    filtered_dir = None
    if not skip_filter:
        filtered_dir = os.path.join(out_dir, f"filtered{suffix}")
        do_filter(filter_input, filtered_dir, f"{step_no}/{total_steps} {tag} 三层规则过滤 filter_trajectories")
        step_no += 1

    return with_eval_jsonl, filtered_dir, step_no


# ── 单个输入(一个原始轨迹根目录/一份已转换好的 converted.jsonl)跑完整流程 ─────────
def run_pipeline_once(traj_in, converted_jsonl, out_dir, args, skip_convert, fold_jsonl_override=None):
    """转换(可跳) -> 主线/fold 两路各自 回填eval_info(可跳) -> 三层规则过滤(可跳)。
    转换失败(--skip-convert 但找不到 converted_jsonl)时打印错误并返回 None,
    调用方(单次模式/批量模式)自行决定是 sys.exit 还是跳过这一条继续下一个。"""
    os.makedirs(out_dir, exist_ok=True)
    base, ext = os.path.splitext(converted_jsonl)
    fold_jsonl = fold_jsonl_override or (base + "_fold" + (ext or ".jsonl"))

    per_track_steps = (0 if args.skip_eval else 1) + (0 if args.skip_filter else 1)
    do_fold = not args.skip_fold
    total_steps = (0 if skip_convert else 1) + per_track_steps * (2 if do_fold else 1)
    step_no = 1

    if not skip_convert:
        do_convert(traj_in, converted_jsonl, args, f"{step_no}/{total_steps} 转换 traj_to_converted")
        step_no += 1
    elif not os.path.isfile(converted_jsonl):
        print(f"[error] 找不到转换结果文件: {converted_jsonl}", file=sys.stderr)
        return None

    if do_fold and not os.path.isfile(fold_jsonl):
        print(f"[warn] fold 结果文件不存在，跳过 fold 那路: {fold_jsonl}", file=sys.stderr)
        do_fold = False
        total_steps = (0 if skip_convert else 1) + per_track_steps

    with_eval_jsonl, filtered_dir, step_no = process_track(
        "", converted_jsonl, out_dir, args.skip_eval, args.skip_filter, step_no, total_steps)

    fold_with_eval_jsonl = fold_filtered_dir = None
    if do_fold:
        fold_with_eval_jsonl, fold_filtered_dir, step_no = process_track(
            "fold", fold_jsonl, out_dir, args.skip_eval, args.skip_filter, step_no, total_steps)

    return {
        "traj_in": traj_in, "out_dir": out_dir,
        "converted_jsonl": converted_jsonl, "fold_jsonl": fold_jsonl if do_fold else None,
        "with_eval_jsonl": with_eval_jsonl, "filtered_dir": filtered_dir,
        "fold_with_eval_jsonl": fold_with_eval_jsonl, "fold_filtered_dir": fold_filtered_dir,
    }


def print_single_result(result):
    print("=" * 70)
    print("[done] 全部完成")
    print(f"  转换结果(主线)     : {result['converted_jsonl']}")
    if result["with_eval_jsonl"]:
        print(f"  回填eval_info(主线) : {result['with_eval_jsonl']}")
    if result["filtered_dir"]:
        print(f"  过滤后保留(主线)   : {result['filtered_dir']}")
        print(f"  过滤统计(主线)     : {os.path.join(result['filtered_dir'], 'filter_stats.json')}")
    if result["fold_jsonl"]:
        print(f"  转换结果(fold)     : {result['fold_jsonl']}")
        if result["fold_with_eval_jsonl"]:
            print(f"  回填eval_info(fold) : {result['fold_with_eval_jsonl']}")
        if result["fold_filtered_dir"]:
            print(f"  过滤后保留(fold)   : {result['fold_filtered_dir']}")
            print(f"  过滤统计(fold)     : {os.path.join(result['fold_filtered_dir'], 'filter_stats.json')}")
    else:
        print("  fold 那路          : 已跳过")


# ── 批量模式: --batch-in 下每个子目录各是一批独立任务, 逐个跑完整流程再汇总统计 ─────
def load_filter_total(filtered_dir):
    """读某一路 filter_trajectories.py 产出的 filter_stats.json, 取其中聚合的 total 字典
    (total/passed/no_tools_schema/bad_json/rules...); 这一路没跑过滤(filtered_dir 为 None)
    或统计文件不存在时返回 None, 调用方据此判断能不能把这批数据并入汇总。"""
    if not filtered_dir:
        return None
    stats_path = os.path.join(filtered_dir, "filter_stats.json")
    if not os.path.isfile(stats_path):
        return None
    with io.open(stats_path, encoding="utf-8") as f:
        return json.load(f)["total"]


def _print_stats_line(prefix, stats):
    r = stats["rules"]
    print(f"{prefix}共 {stats['total']} 条轨迹 | 通过 {stats['passed']} "
          f"| 无tools约束(规则1/2跳过) {stats['no_tools_schema']} | JSON解析失败 {stats['bad_json']}")
    print(f"{prefix}  规则1(工具越界) 命中 {r['rule1_tool_name']['hit_trajectories']} 条 / "
          f"{r['rule1_tool_name']['hit_occurrences']} 次")
    print(f"{prefix}  规则2(参数不合法) 命中 {r['rule2_tool_args']['hit_trajectories']} 条 / "
          f"{r['rule2_tool_args']['hit_occurrences']} 次")
    print(f"{prefix}  规则3(signature不可信) 命中 {r['rule3_signature']['hit_trajectories']} 条 / "
          f"{r['rule3_signature']['hit_occurrences']} 次")


def run_batch(args):
    # 复用 filter_trajectories.py 里已有的聚合结构(new_stats/merge_stats), 不用另起一套。
    from filter_trajectories import new_stats, merge_stats

    root = args.batch_in
    if not os.path.isdir(root):
        print(f"[error] --batch-in 不是目录: {root}", file=sys.stderr)
        sys.exit(1)
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if not names:
        print(f"[error] --batch-in 下没有子目录: {root}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[batch] 共 {len(names)} 个子目录任务 -> {args.out_dir}")
    agg_main = new_stats()
    agg_fold = new_stats()
    batches = []
    n_ok = 0
    for i, name in enumerate(names, 1):
        sub_in = os.path.join(root, name)
        sub_out = os.path.join(args.out_dir, name)
        print("#" * 70)
        print(f"[批次 {i}/{len(names)}] {name}  ({sub_in} -> {sub_out})")
        print("#" * 70)
        result = run_pipeline_once(sub_in, os.path.join(sub_out, "converted.jsonl"), sub_out, args,
                                    skip_convert=False)
        if result is None:
            print(f"[warn] 批次 {name} 转换失败，跳过统计", file=sys.stderr)
            batches.append({"name": name, "traj_in": sub_in, "out_dir": sub_out, "error": "convert_failed"})
            continue

        n_ok += 1
        main_total = load_filter_total(result["filtered_dir"])
        fold_total = load_filter_total(result["fold_filtered_dir"])
        if main_total:
            merge_stats(agg_main, main_total)
        if fold_total:
            merge_stats(agg_fold, fold_total)
        batches.append({"name": name, "traj_in": sub_in, "out_dir": sub_out,
                         "main": main_total, "fold": fold_total})

    combined = new_stats()
    merge_stats(combined, agg_main)
    merge_stats(combined, agg_fold)

    print("=" * 70)
    print(f"[batch done] {n_ok}/{len(names)} 个子目录处理完成")
    if args.skip_filter:
        print("  [warn] 加了 --skip-filter，没有 filter_stats.json 可汇总，以下总数/通过数全是 0")
    _print_stats_line("  主线合计   : ", agg_main)
    _print_stats_line("  fold合计   : ", agg_fold)
    _print_stats_line("  主线+fold  : ", combined)

    summary = {
        "batch_in": root, "out_dir": args.out_dir,
        "batches": batches,
        "total": {"main": agg_main, "fold": agg_fold, "combined": combined},
    }
    summary_path = os.path.join(args.out_dir, "batch_stats.json")
    with io.open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  批量统计已写入 {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="一键跑 转换(traj_to_converted) -> 回填eval_info(add_eval_info) "
                                              "-> 三层规则过滤(filter_trajectories) 流程, "
                                              "converted.jsonl 主线和 converted_fold.jsonl 各跑一遍; "
                                              "加 --batch-in 可批量跑一个目录下的多个子目录任务并汇总统计")
    ap.add_argument("--traj-in", default=None,
                     help="原始轨迹根目录（单个 .trajectory.jsonl 或其所在目录，跑转换阶段时必填；跟 --batch-in 互斥）")
    ap.add_argument("--batch-in", default=None,
                     help="批量模式：目录，其下每个子目录各是一批独立的原始轨迹任务(不要求属于同一批次)，"
                          "逐个跑完整流程，各自输出到 --out-dir/<子目录名>/ 下，并把各子目录 filtered(_fold)/"
                          "的 filter_stats.json 汇总成 --out-dir/batch_stats.json(总轨迹条数 + 规则通过条数)。"
                          "跟 --traj-in/--converted-jsonl/--fold-jsonl/--skip-convert 互斥。")
    ap.add_argument("--converted-jsonl", default=None,
                     help="转换阶段主线输出/复用的 jsonl 路径（默认 out-dir/converted.jsonl；"
                          "--skip-convert 时必须指向已存在的 jsonl）")
    ap.add_argument("--fold-jsonl", default=None,
                     help="转换阶段 fold 那路输出/复用的 jsonl 路径（默认跟 traj_to_converted.py 一致，"
                          "即 --converted-jsonl 去扩展名后加 _fold；--skip-convert 时若该文件不存在则"
                          "自动跳过 fold 那路，不会报错）")
    ap.add_argument("--out-dir", required=True, help="输出根目录（自动创建）")
    ap.add_argument("--owner", default="00935640", help="traj_to_converted --owner（默认 00935640）")
    ap.add_argument("--language", default="zh", help="traj_to_converted --language（默认 zh）")
    ap.add_argument("--category", default="agent", help="traj_to_converted --category（默认 agent）")
    ap.add_argument("--skip-convert", action="store_true", help="跳过转换阶段，直接用 --converted-jsonl(/--fold-jsonl)")
    ap.add_argument("--skip-eval", action="store_true",
                     help="跳过回填 eval_info 阶段，过滤阶段直接吃转换阶段的输出")
    ap.add_argument("--skip-filter", action="store_true", help="跳过三层规则过滤阶段，只做转换(+回填)")
    ap.add_argument("--skip-fold", action="store_true", help="不处理 converted_fold.jsonl，只跑主线")
    args = ap.parse_args()

    if args.batch_in:
        if args.traj_in or args.converted_jsonl or args.fold_jsonl or args.skip_convert:
            ap.error("--batch-in 跟 --traj-in/--converted-jsonl/--fold-jsonl/--skip-convert 不能同时使用")
        run_batch(args)
        return

    if not args.skip_convert and not args.traj_in:
        ap.error("跑转换阶段需要 --traj-in（或加 --skip-convert 只跑后续阶段，或用 --batch-in 批量模式）")
    if args.skip_convert and not args.converted_jsonl:
        ap.error("--skip-convert 时必须用 --converted-jsonl 指向已存在的转换结果")

    os.makedirs(args.out_dir, exist_ok=True)
    converted_jsonl = args.converted_jsonl or os.path.join(args.out_dir, "converted.jsonl")
    result = run_pipeline_once(args.traj_in, converted_jsonl, args.out_dir, args, args.skip_convert,
                                fold_jsonl_override=args.fold_jsonl)
    if result is None:
        sys.exit(1)
    print_single_result(result)


if __name__ == "__main__":
    main()
