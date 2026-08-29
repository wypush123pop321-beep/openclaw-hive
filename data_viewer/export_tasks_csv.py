#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把「采集任务」的每个任务(基础字段 + 前端表格里的计算列)导出成 CSV。

计算列口径与前端 index.html / server.py._task_summary 完全一致：
  L0     = filtered_count + dropped_count   (总轨迹)
  L1     = filtered_count                   (可用轨迹)
  T_DONE = task_done_count                  (含【Task_Done】标记)
  L1.5   = with_eval_count                  (有裁决)
  L2     = completion_ge_0.5                (completion≥0.5)
  L3     = completion_eq_1                  (completion=1)
数据源: tasks.json + 每个任务 output_dir 下的 filter_stats.json。
"""
import csv
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(BASE, "tasks.json")
OUT = os.path.join(os.path.dirname(BASE), "采集任务.csv")


def pct(x):
    """占比(0~1)转百分比字符串, 空值给空串。"""
    return "" if x in (None, "") else round(float(x) * 100, 1)


def load_stats(output_dir):
    p = os.path.join(BASE, output_dir, "filter_stats.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    with open(TASKS, encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]

    # 基础字段(按首次出现顺序)
    base_cols = []
    for t in tasks:
        for k in t.keys():
            if k not in base_cols:
                base_cols.append(k)

    # 计算列(与前端表格一致)
    stat_cols = [
        "已采集",
        "L0(总轨迹)", "L1(可用)", "T_DONE", "L1.5(有裁决)", "L2(≥0.5)", "L3(=1)",
        "Est_Tokens(L1)", "Avg_Char_Len(L1)",
        "TASK_DONE数量",
        "工具失败轨迹数", "工具失败占比%",
        "工具误用轨迹数", "工具误用占比%",
        "①不存在工具_轨迹数", "①占比%",
        "②file_path入参_轨迹数", "②占比%",
        "③Windows路径_轨迹数", "③占比%",
    ]
    cols = base_cols + stat_cols

    rows = []
    for t in tasks:
        row = {}
        for k in base_cols:
            v = t.get(k, "")
            if isinstance(v, list):
                v = " | ".join(str(x) for x in v)
            elif isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False)
            elif v is None:
                v = ""
            row[k] = v

        s = load_stats(t.get("output_dir", "")) or {}
        avail = bool(s)
        row["已采集"] = "是" if avail else "否"

        filtered = s.get("filtered_count", 0)
        dropped = s.get("dropped_count", 0)
        row["L0(总轨迹)"] = (filtered + dropped) if avail else ""
        row["L1(可用)"] = filtered if avail else ""
        row["T_DONE"] = s.get("task_done_count", 0) if avail else ""
        row["L1.5(有裁决)"] = s.get("with_eval_count", 0) if avail else ""
        row["L2(≥0.5)"] = s.get("completion_ge_0.5", 0) if avail else ""
        row["L3(=1)"] = s.get("completion_eq_1", 0) if avail else ""

        ts = (s.get("token_stats") or {}).get("L1") or {}
        cs = (s.get("char_len_stats") or {}).get("L1") or {}
        row["Est_Tokens(L1)"] = ts.get("avg_total_tokens", "")
        row["Avg_Char_Len(L1)"] = cs.get("avg_char_len", "")
        row["TASK_DONE数量"] = s.get("task_done_count", "") if avail else ""

        tf = s.get("tool_fail_stats") or {}
        row["工具失败轨迹数"] = tf.get("traj_with_fail", "")
        row["工具失败占比%"] = pct(tf.get("fail_rate"))

        w = s.get("win_stats") or {}
        row["工具误用轨迹数"] = w.get("traj_with_win", "")
        row["工具误用占比%"] = pct(w.get("win_traj_rate"))
        row["①不存在工具_轨迹数"] = w.get("traj_with_r1", "")
        row["①占比%"] = pct(w.get("r1_traj_rate"))
        row["②file_path入参_轨迹数"] = w.get("traj_with_r2", "")
        row["②占比%"] = pct(w.get("r2_traj_rate"))
        row["③Windows路径_轨迹数"] = w.get("traj_with_r3", "")
        row["③占比%"] = pct(w.get("r3_traj_rate"))

        rows.append(row)

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"WROTE {OUT}")
    print(f"ROWS={len(rows)} COLS={len(cols)}")
    print("COLUMNS:", cols)


if __name__ == "__main__":
    main()
