# -*- coding: utf-8 -*-
"""
按三层规则过滤 traj_to_converted.py 输出的轨迹文件(如 testcase/converted.jsonl),
只把三层规则都不命中的轨迹原样写到输出目录, 同时统计每条规则命中的次数(有多少次
工具调用/多少个 signature 触发了它)和命中的轨迹数量(有多少条轨迹里至少出现过一次)。
被过滤掉的轨迹不会丢弃, 而是按命中的规则分类原样写进 <out-dir>/rules/<规则名>/<同名文件>
(rule1_tool_name / rule2_tool_args / rule3_signature 三个子目录), 同一条轨迹命中几条
规则就会出现在几个子目录里, 方便按规则分别人工核查是哪些轨迹被过滤、为什么被过滤。

三层规则(任一命中即整条轨迹被过滤掉, 不写入输出):
  1. 工具越界: assistant 消息里的 tool_calls[].function.name 不在该轨迹自带的
     tools[].function.name 允许集合里。
  2. 参数不合法: 工具名在允许集合里, 但 tool_calls[].function.arguments(JSON 字符串)
     解析失败, 或解析后跟该工具 tools[].function.parameters(JSON Schema)的 required/
     类型/enum 对不上, 或传了 schema 里没定义、且 additionalProperties 显式为 false
     的多余字段 —— 只做这几项基础校验, schema 里损坏/无法识别的部分(比如上游截断
     产生的 "[[trajectory truncated: ...]]" 占位串)一律当作"无约束"跳过, 不算命中。
     另外对 read/write/edit 这三个工具单独加了一条: path 参数里如果出现
     "C:\", "D:\" 这类 Windows 盘符路径(沙箱应该是 Linux 路径), 直接判定不通过。
  3. signature 不可信: assistant 消息的 signature 是 base64, 解出原始字节后要同时包含
     "claude" 和 "think"(大小写不敏感)字样才算通过, 解码失败或缺任一关键词都算命中。

注意: 规则1/2 依赖轨迹自带的 tools 字段。如果某条轨迹的 tools 为空(上游没能把工具
schema 记录下来, 实测 testcase/converted.jsonl 里全部 5302 条都是这种情况), 规则1/2
在这条轨迹上无法判定, 直接跳过(不算命中, 也不算通过验证), 并单独计入
no_tools_schema_trajectories, 避免把"数据缺失"误判成"过滤未命中"。

用法:
  python filter_trajectories.py --traj-in "<单个 .jsonl 或包含若干 .jsonl 的目录>" \
      --out-dir "<输出目录>"
"""
import os
import io
import re
import json
import argparse

from base64_convert import base64_decode_raw


# ── 阶段 0: 从轨迹自带的 tools 字段构建 {工具名: parameters schema} ──────────────
def load_allowed_tools(tools_field):
    allowed = {}
    if not isinstance(tools_field, list):
        return allowed
    for t in tools_field:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        allowed[name] = fn.get("parameters")
    return allowed


# ── 阶段 1: 基础 JSON Schema 校验(required + 类型 + enum), 无法识别的部分当作通过 ──
def _is_truncated_marker(value):
    """上游 trace 采集在序列化过深的结构时会把值整体替换成这种占位串
    (如 "[[trajectory truncated: reason=trajectory-depth-limit, ...]]"),
    此时真实参数值已丢失, 无法校验, 不能当成类型不符去命中规则2。"""
    return isinstance(value, str) and value.startswith("[[trajectory truncated")


def _type_ok(value, jtype):
    if jtype == "string":
        return isinstance(value, str)
    if jtype == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if jtype == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if jtype == "boolean":
        return isinstance(value, bool)
    if jtype == "object":
        return isinstance(value, dict)
    if jtype == "array":
        return isinstance(value, list)
    if jtype == "null":
        return value is None
    return True  # 未知类型关键字, 无法校验, 视为通过


def args_match_schema(args, schema):
    if not isinstance(schema, dict):
        return True  # schema 本身缺失/损坏, 无法校验
    schema_type = schema.get("type")
    if schema_type not in (None, "object"):
        return True  # 顶层不是 object schema, 这里不做结构校验

    if not isinstance(args, dict):
        return False  # 工具调用参数应是 JSON object, 却不是

    required = schema.get("required")
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in args:
                return False

    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    reject_additional = schema.get("additionalProperties") is False

    if properties or reject_additional:
        for key, value in args.items():
            prop = properties.get(key)
            if prop is None:
                if reject_additional:
                    return False  # schema 未定义的字段, 且明确禁止 additionalProperties
                continue
            if _is_truncated_marker(value):
                continue
            if not isinstance(prop, dict):
                continue
            jtype = prop.get("type")
            if isinstance(jtype, str):
                if not _type_ok(value, jtype):
                    return False
            elif isinstance(jtype, list):
                candidates = [t for t in jtype if isinstance(t, str)]
                if candidates and not any(_type_ok(value, t) for t in candidates):
                    return False
            enum = prop.get("enum")
            if isinstance(enum, list) and enum and value not in enum:
                return False
    return True


# read/write/edit 的 path 参数不应该是 Windows 盘符路径("C:\", "D:\" 这类), 沙箱是 Linux 路径
_WINDOWS_PATH_TOOLS = {"read", "write", "edit"}
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\")


def has_windows_style_path(name, args):
    if name not in _WINDOWS_PATH_TOOLS or not isinstance(args, dict):
        return False
    path = args.get("path")
    return isinstance(path, str) and bool(_WINDOWS_PATH_RE.search(path))


# ── 阶段 2: signature 校验 ──────────────────────────────────────────────────
def signature_ok(signature):
    raw = base64_decode_raw(signature)
    lower = raw.lower()
    return b"claude" in lower and b"think" in lower


# ── 阶段 3: 单条轨迹跑三层规则, 返回每层命中次数 + 是否有可用的 tools 约束 ──────
def check_trajectory(obj):
    hits = {"rule1_tool_name": 0, "rule2_tool_args": 0, "rule3_signature": 0}
    allowed = load_allowed_tools(obj.get("tools"))
    has_tools_schema = bool(allowed)

    for m in obj.get("messages", []):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue

        if has_tools_schema:
            for tc in (m.get("tool_calls") or []):
                fn = (tc or {}).get("function") or {}
                name = fn.get("name")
                if name not in allowed:
                    hits["rule1_tool_name"] += 1
                    continue
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    parsed = True
                except (json.JSONDecodeError, TypeError, ValueError):
                    parsed = False
                    args = None
                if (not parsed
                        or not args_match_schema(args, allowed.get(name))
                        or has_windows_style_path(name, args)):
                    hits["rule2_tool_args"] += 1

        signature = m.get("signature")
        if signature:
            try:
                ok = signature_ok(signature)
            except Exception:
                ok = False
            if not ok:
                hits["rule3_signature"] += 1

    return hits, has_tools_schema


# ── 阶段 4: 遍历 --traj-in, 按规则过滤, 通过的原样写入 --out-dir 下同名文件 ──────
def iter_jsonl_files(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for n in names:
            if n.endswith(".jsonl"):
                files.append(os.path.join(root, n))
    return sorted(files)


def new_stats():
    return {
        "total": 0,
        "bad_json": 0,
        "passed": 0,
        "no_tools_schema": 0,
        "rules": {
            "rule1_tool_name": {"hit_occurrences": 0, "hit_trajectories": 0},
            "rule2_tool_args": {"hit_occurrences": 0, "hit_trajectories": 0},
            "rule3_signature": {"hit_occurrences": 0, "hit_trajectories": 0},
        },
    }


def merge_stats(total, part):
    total["total"] += part["total"]
    total["bad_json"] += part["bad_json"]
    total["passed"] += part["passed"]
    total["no_tools_schema"] += part["no_tools_schema"]
    for rule, s in part["rules"].items():
        total["rules"][rule]["hit_occurrences"] += s["hit_occurrences"]
        total["rules"][rule]["hit_trajectories"] += s["hit_trajectories"]


def process_file(path, out_path, rules_dir, indent=""):
    stats = new_stats()
    basename = os.path.basename(path)
    rule_writers = {}  # rule -> 打开的文件句柄, 命中时才惰性创建, 不产出没内容的空文件

    def get_rule_writer(rule):
        w = rule_writers.get(rule)
        if w is None:
            rule_out_dir = os.path.join(rules_dir, rule)
            os.makedirs(rule_out_dir, exist_ok=True)
            w = io.open(os.path.join(rule_out_dir, basename), "w", encoding="utf-8")
            rule_writers[rule] = w
        return w

    try:
        with io.open(path, encoding="utf-8", errors="replace") as fin, \
                io.open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                raw_line = line.rstrip("\n")
                if not raw_line.strip():
                    continue
                stats["total"] += 1
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue

                hits, has_tools_schema = check_trajectory(obj)
                if not has_tools_schema:
                    stats["no_tools_schema"] += 1
                for rule, count in hits.items():
                    if count > 0:
                        stats["rules"][rule]["hit_occurrences"] += count
                        stats["rules"][rule]["hit_trajectories"] += 1
                        # 同一条轨迹命中几条规则就写进几个分类目录, 方便按规则分别排查
                        get_rule_writer(rule).write(raw_line + "\n")

                if sum(hits.values()) == 0:
                    stats["passed"] += 1
                    fout.write(raw_line + "\n")
    finally:
        for w in rule_writers.values():
            w.close()

    r = stats["rules"]
    print(f"{indent}[{os.path.basename(path)}] 共 {stats['total']} 条 | 通过 {stats['passed']} "
          f"| 无tools约束(规则1/2跳过) {stats['no_tools_schema']} | JSON解析失败 {stats['bad_json']} | "
          f"规则1命中 {r['rule1_tool_name']['hit_trajectories']}条/{r['rule1_tool_name']['hit_occurrences']}次 "
          f"| 规则2命中 {r['rule2_tool_args']['hit_trajectories']}条/{r['rule2_tool_args']['hit_occurrences']}次 "
          f"| 规则3命中 {r['rule3_signature']['hit_trajectories']}条/{r['rule3_signature']['hit_occurrences']}次")
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="按工具越界/参数不合法/signature不可信三层规则过滤轨迹文件, "
                    "全部规则都不命中的轨迹原样写入输出目录")
    ap.add_argument("--traj-in", required=True,
                     help="输入: 单个 .jsonl 文件, 或包含若干 .jsonl 文件的目录")
    ap.add_argument("--out-dir", required=True,
                     help="输出目录(自动创建): 通过的轨迹按输入文件同名落在这里, "
                          "被过滤掉的按命中的规则分类落在 <out-dir>/rules/<规则名>/ 下(同名文件), "
                          "同一条轨迹命中几条规则就会出现在几个规则子目录里, 方便按规则分别排查")
    a = ap.parse_args()

    if not os.path.exists(a.traj_in):
        ap.error(f"traj-in 不存在: {a.traj_in}")
    files = iter_jsonl_files(a.traj_in)
    if not files:
        ap.error(f"traj-in 下未找到 .jsonl 文件: {a.traj_in}")
    os.makedirs(a.out_dir, exist_ok=True)
    rules_dir = os.path.join(a.out_dir, "rules")

    print(f"[1] 处理 {len(files)} 个轨迹文件 -> {a.out_dir} (被过滤的按规则分类落进 {rules_dir})")
    total = new_stats()
    file_stats = []
    for f in files:
        out_path = os.path.join(a.out_dir, os.path.basename(f))
        s = process_file(f, out_path, rules_dir, indent="  ")
        merge_stats(total, s)
        file_stats.append({"file": f, **s})

    r = total["rules"]
    print(f"[done] 共 {total['total']} 条轨迹 | 通过 {total['passed']} "
          f"| 无tools约束(规则1/2跳过) {total['no_tools_schema']} | JSON解析失败 {total['bad_json']}")
    print(f"       规则1(工具越界) 命中 {r['rule1_tool_name']['hit_trajectories']} 条轨迹 / "
          f"{r['rule1_tool_name']['hit_occurrences']} 次")
    print(f"       规则2(参数不合法) 命中 {r['rule2_tool_args']['hit_trajectories']} 条轨迹 / "
          f"{r['rule2_tool_args']['hit_occurrences']} 次")
    print(f"       规则3(signature不可信) 命中 {r['rule3_signature']['hit_trajectories']} 条轨迹 / "
          f"{r['rule3_signature']['hit_occurrences']} 次")

    summary = {"traj_in": a.traj_in, "out_dir": a.out_dir, "rules_dir": rules_dir,
               "total": total, "files": file_stats}
    summary_path = os.path.join(a.out_dir, "filter_stats.json")
    with io.open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"       统计已写入 {summary_path}")
    print(f"       被过滤轨迹按规则分类落在 {rules_dir}")


if __name__ == "__main__":
    main()
