# -*- coding: utf-8 -*-
"""
从 OBS 下载 assistant + evaluator 两个原始轨迹目录, 然后跑处理流水线。

流程:
  1. 用 obsutil 把两个 obs 目录下载到 <out_dir>/origin/ (实时打印下载速度)。
  2. 在 origin 下自动定位两个轨迹目录(含 session_report.xlsx 的为 assistant, 另一个为 evaluator),
     调用 run_pipeline.py 完成 筛选→转 pangu→统计。

用法:
  python download_and_run.py <assistant_obs> <evaluator_obs> <out_dir> [--obsutil PATH]

示例:
  python download_and_run.py ^
    "obs://rl-agentdata/zhengnianzu/test/session_analysis/env-claude-99oR/key-5c33/ex-260716171238/" ^
    "obs://rl-agentdata/zhengnianzu/test/session_analysis/env-claude-99oR/key-122a/ex-260716170233/" ^
    output
"""
import os, sys, time, argparse, subprocess

HERE            = os.path.dirname(os.path.abspath(__file__))
RUN_PIPELINE    = os.path.join(HERE, "run_pipeline.py")
DEFAULT_OBSUTIL = r"D:\tools\obsutil_windows_amd64_5.8.3\obsutil.exe"
REPORT_NAME     = "session_report.xlsx"


def obs_leaf(obs_path):
    """obs 路径末段目录名, 如 .../ex-260716171238/ -> ex-260716171238"""
    return os.path.basename(obs_path.rstrip("/").rstrip("\\"))


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def download(obsutil, obs_path, dest_dir):
    """obsutil cp -r -f <obs_path> <dest_dir>, 实时打印 obsutil 进度/速度。

    默认(不加 -flat) obsutil 会在 dest_dir 下重建 obs 末段目录, 即
    dest_dir/<leaf>/...。这里把 dest_dir 设为 origin/, 让其自然落成 origin/<leaf>/。
    """
    os.makedirs(dest_dir, exist_ok=True)
    obs_path = obs_path if obs_path.endswith("/") else obs_path + "/"
    cmd = [obsutil, "cp", "-r", "-f", obs_path, dest_dir]
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()

    # obsutil 用 \r 刷新进度行; 按字符读, 遇 \r/\n 断行, 把含速度(B/s)的进度实时打印
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace")
    buf = ""
    last_print = 0.0
    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            break
        if ch in "\r\n":
            line = buf.strip()
            buf = ""
            if not line:
                continue
            # 进度行(含实时速度)用 \r 原地刷新; 其它信息正常换行
            now = time.time()
            if "/s" in line and now - last_print < 0.2:
                continue   # 限频, 避免刷屏
            end = "\r" if ("/s" in line and "%" in line) else "\n"
            print(f"    {line}".ljust(90), end=end, flush=True)
            last_print = now
        else:
            buf += ch
    print()  # 收尾换行
    proc.wait()
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"obsutil 下载失败(退出码 {proc.returncode}): {obs_path}")

    total = dir_size(dest_dir)
    speed = total / dt if dt > 0 else 0
    print(f"  [ok] {obs_leaf(obs_path)}: {human(total)} / {dt:.1f}s = {human(speed)}/s (平均)", flush=True)


def find_traj_dirs(origin):
    """在 origin 下(递归)找轨迹目录: 含 session_report.xlsx 的为 assistant, 其它含时间戳子目录的为候选。

    返回 (assistant_dir, evaluator_dir)。
    """
    assistant = None
    candidates = []
    for root, dirs, files in os.walk(origin):
        if REPORT_NAME in files:
            assistant = root
        # 轨迹目录特征: 其下子目录形如 2026-07-15_16-04-41_558, 且子目录里有 json
        ts_subdirs = [d for d in dirs if len(d) >= 17 and d[:4].isdigit() and "_" in d]
        if ts_subdirs:
            candidates.append(root)
    evaluator = None
    for c in candidates:
        if c != assistant:
            evaluator = c
            break
    return assistant, evaluator


def main():
    ap = argparse.ArgumentParser(
        description="从 OBS 下载两个轨迹目录后运行处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assistant_obs", help="assistant 轨迹的 obs 路径(含 session_report.xlsx)")
    ap.add_argument("evaluator_obs", help="evaluator(质检) 轨迹的 obs 路径")
    ap.add_argument("out_dir",       help="输出目录(下载落到 <out_dir>/origin/, 结果落到 <out_dir>/)")
    ap.add_argument("--obsutil", default=DEFAULT_OBSUTIL, help=f"obsutil 路径(默认 {DEFAULT_OBSUTIL})")
    a = ap.parse_args()

    if not os.path.exists(a.obsutil):
        ap.error(f"obsutil 不存在: {a.obsutil}")

    origin = os.path.join(a.out_dir, "origin")
    os.makedirs(origin, exist_ok=True)

    print(f"[1] 下载 assistant 轨迹 <- {a.assistant_obs}")
    download(a.obsutil, a.assistant_obs, origin)
    print(f"[2] 下载 evaluator 轨迹 <- {a.evaluator_obs}")
    download(a.obsutil, a.evaluator_obs, origin)

    # 优先按 obs 末段名定位; 找不到再自动探测
    a_name, e_name = obs_leaf(a.assistant_obs), obs_leaf(a.evaluator_obs)
    a_local = os.path.join(origin, a_name)
    e_local = os.path.join(origin, e_name)
    if not (os.path.isdir(a_local) and os.path.isdir(e_local)):
        print("  [i] 按末段名未直接定位到目录, 自动探测中...")
        a_local, e_local = find_traj_dirs(origin)
    if not a_local or not os.path.isdir(a_local):
        sys.exit(f"[error] 未找到 assistant 轨迹目录于 {origin}")
    if not e_local or not os.path.isdir(e_local):
        sys.exit(f"[error] 未找到 evaluator 轨迹目录于 {origin}")

    print(f"[3] 处理: run_pipeline.py")
    print(f"      assistant = {a_local}")
    print(f"      evaluator = {e_local}")
    cmd = [sys.executable, RUN_PIPELINE, a_local, e_local, a.out_dir]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(f"[fail] 处理流水线退出码 {rc}")
    print(f"[done] 全部完成, 结果见 {a.out_dir}")


if __name__ == "__main__":
    main()
