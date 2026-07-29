# -*- coding: utf-8 -*-
"""
从 OBS 或远程服务器(SSH/SFTP) 下载 assistant + evaluator 两个原始轨迹目录, 然后跑处理流水线。

流程:
  1. 把两个来源目录下载到 <out_dir>/origin/ (实时打印下载进度)。
     每个来源既可以是 obs://... 地址(走 obsutil), 也可以是 ssh://user@host/remote/path/
     (走 paramiko SFTP, 适用于数据只落在某台服务器本地磁盘、还没传到 OBS 的情况)。
  2. 在 origin 下自动定位两个轨迹目录(含 session_report.xlsx 的为 assistant, 另一个为 evaluator),
     调用 run_pipeline.py 完成 筛选→转 pangu→统计。

用法:
  python download_and_run.py <assistant_obs> <evaluator_obs> <out_dir> [--obsutil PATH]

示例(OBS):
  python download_and_run.py ^
    "obs://rl-agentdata/zhengnianzu/test/session_analysis/env-claude-99oR/key-5c33/ex-260716171238/" ^
    "obs://rl-agentdata/zhengnianzu/test/session_analysis/env-claude-99oR/key-122a/ex-260716170233/" ^
    output

示例(SSH, 密码通过环境变量 SSH_PASSWORD_ASSISTANT / SSH_PASSWORD_EVALUATOR 传入, 不出现在命令行里):
  SSH_PASSWORD_ASSISTANT=xxx python download_and_run.py \
    "ssh://user@10.0.0.1/mnt/sdb/data/session/env-claude-99oR/26071621/key-1433/" \
    "obs://rl-agentdata/.../ex-260716170233/" \
    output
"""
import os, sys, time, stat, argparse, subprocess
from urllib.parse import urlparse

HERE            = os.path.dirname(os.path.abspath(__file__))
RUN_PIPELINE    = os.path.join(HERE, "run_pipeline.py")
DEFAULT_OBSUTIL = r"D:\tools\obsutil_windows_amd64_5.8.3\obsutil.exe"
REPORT_NAME     = "session_report.xlsx"


def obs_leaf(obs_path):
    """路径末段目录名, 如 .../ex-260716171238/ -> ex-260716171238。
    对 obs:// 和 ssh:// 路径都适用, 因为只是纯字符串操作。"""
    return os.path.basename(obs_path.rstrip("/").rstrip("\\"))


def is_ssh_url(path):
    return path.startswith("ssh://")


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


def _redact_cmd(cmd):
    """打印用: 把 -i/-k(access key / secret key)后面的值替换成 ***, 避免凭证出现在终端/日志里。"""
    out = list(cmd)
    for flag in ("-i", "-k"):
        if flag in out:
            idx = out.index(flag)
            if idx + 1 < len(out):
                out[idx + 1] = "***"
    return out


def download(obsutil, obs_path, dest_dir, obs_cred_args=None):
    """obsutil cp -r -f <obs_path> <dest_dir>, 实时打印 obsutil 进度/速度。

    默认(不加 -flat) obsutil 会在 dest_dir 下重建 obs 末段目录, 即
    dest_dir/<leaf>/...。这里把 dest_dir 设为 origin/, 让其自然落成 origin/<leaf>/。
    obs_cred_args: 可选 ["-i", ak, "-k", sk, "-e", endpoint], 覆盖 obsutil 全局默认凭证。
    """
    os.makedirs(dest_dir, exist_ok=True)
    obs_path = obs_path if obs_path.endswith("/") else obs_path + "/"
    cmd = [obsutil, "cp", "-r", "-f", obs_path, dest_dir] + (obs_cred_args or [])
    print(f"  $ {' '.join(_redact_cmd(cmd))}", flush=True)
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


def parse_ssh_url(ssh_url):
    """ssh://user@host/remote/path/ -> (user, host, "/remote/path/")。
    urlparse 对 ssh:// 能正确切出 netloc(user@host)和 path, 不需要手写正则。"""
    u = urlparse(ssh_url)
    if u.scheme != "ssh" or not u.username or not u.hostname or not u.path:
        raise ValueError(f"非法的 ssh:// 地址(应为 ssh://user@host/remote/path/): {ssh_url}")
    return u.username, u.hostname, u.path


def download_ssh(ssh_url, dest_dir, password):
    """通过 SFTP 把远程服务器上的一个目录递归下载到 dest_dir/<leaf>/, 与 obsutil 落盘方式一致
    (dest_dir 传 origin/, 落成 origin/<leaf>/), 这样下游 run_pipeline.py 的目录探测逻辑不用改。

    密码只作为函数参数短暂存在于这次调用栈里, 建完连接就不再需要, 不写日志、不进 dest_dir。
    """
    import paramiko

    user, host, remote_path = parse_ssh_url(ssh_url)
    remote_path = remote_path.rstrip("/") or "/"
    leaf = obs_leaf(ssh_url)
    local_root = os.path.join(dest_dir, leaf)
    os.makedirs(local_root, exist_ok=True)

    print(f"  $ sftp {user}@{host}:{remote_path} -> {local_root}", flush=True)
    t0 = time.time()

    transport = paramiko.Transport((host, 22))
    try:
        transport.connect(username=user, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            n_files = [0]
            n_bytes = [0]
            last_print = [0.0]

            def walk_download(remote_dir, local_dir):
                os.makedirs(local_dir, exist_ok=True)
                for entry in sftp.listdir_attr(remote_dir):
                    r_path = remote_dir.rstrip("/") + "/" + entry.filename
                    l_path = os.path.join(local_dir, entry.filename)
                    if stat.S_ISDIR(entry.st_mode):
                        walk_download(r_path, l_path)
                    else:
                        sftp.get(r_path, l_path)
                        n_files[0] += 1
                        n_bytes[0] += entry.st_size or 0
                        now = time.time()
                        if now - last_print[0] >= 0.2:
                            elapsed = now - t0
                            speed = n_bytes[0] / elapsed if elapsed > 0 else 0
                            print(f"    已下载 {n_files[0]} 个文件 ({human(n_bytes[0])}, {human(speed)}/s)".ljust(90),
                                  end="\r", flush=True)
                            last_print[0] = now

            walk_download(remote_path, local_root)
            print()  # 收尾换行
        finally:
            sftp.close()
    except paramiko.AuthenticationException:
        raise RuntimeError(f"SSH 认证失败, 请检查用户名/密码: {user}@{host}")
    except Exception as exc:
        raise RuntimeError(f"SFTP 下载失败: {user}@{host}:{remote_path} - {exc}")
    finally:
        transport.close()

    dt = time.time() - t0
    total = dir_size(local_root)
    speed = total / dt if dt > 0 else 0
    print(f"  [ok] {leaf}: {human(total)} / {dt:.1f}s = {human(speed)}/s (平均)", flush=True)


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


def fetch(obsutil, source, dest_dir, ssh_password_env, obs_cred_args=None):
    """按 source 的 scheme 分发到 obsutil(obs://) 或 SFTP(ssh://)。
    SSH 密码从环境变量读, 从不出现在命令行参数或日志里。"""
    if is_ssh_url(source):
        password = os.environ.get(ssh_password_env, "")
        if not password:
            raise RuntimeError(f"缺少 SSH 密码(环境变量 {ssh_password_env} 未设置): {source}")
        download_ssh(source, dest_dir, password)
    else:
        download(obsutil, source, dest_dir, obs_cred_args=obs_cred_args)


def main():
    ap = argparse.ArgumentParser(
        description="从 OBS 或 SSH 服务器下载两个轨迹目录后运行处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assistant_obs", help="assistant 轨迹来源: obs://... 或 ssh://user@host/remote/path/")
    ap.add_argument("evaluator_obs", help="evaluator(质检) 轨迹来源: obs://... 或 ssh://user@host/remote/path/")
    ap.add_argument("out_dir",       help="输出目录(下载落到 <out_dir>/origin/, 结果落到 <out_dir>/)")
    ap.add_argument("--obsutil", default=DEFAULT_OBSUTIL, help=f"obsutil 路径(默认 {DEFAULT_OBSUTIL})")
    ap.add_argument("--obs-ak", default=None, help="OBS Access Key ID(可选; 三个 --obs-* 参数要么都给要么都不给, "
                    "用于覆盖 obsutil 全局默认凭证, 访问另一个账号/桶时用)")
    ap.add_argument("--obs-sk", default=None, help="OBS Secret Access Key(可选, 见 --obs-ak)")
    ap.add_argument("--obs-endpoint", default=None, help="OBS endpoint(可选, 如 obs.cn-east-4.myhuaweicloud.com, 见 --obs-ak)")
    a = ap.parse_args()

    need_obsutil = not (is_ssh_url(a.assistant_obs) and is_ssh_url(a.evaluator_obs))
    if need_obsutil and not os.path.exists(a.obsutil):
        ap.error(f"obsutil 不存在: {a.obsutil}")

    obs_cred_vals = (a.obs_ak, a.obs_sk, a.obs_endpoint)
    if any(obs_cred_vals) and not all(obs_cred_vals):
        ap.error("--obs-ak / --obs-sk / --obs-endpoint 要么都给, 要么都不给")
    obs_cred_args = ["-i", a.obs_ak, "-k", a.obs_sk, "-e", a.obs_endpoint] if all(obs_cred_vals) else []

    origin = os.path.join(a.out_dir, "origin")
    os.makedirs(origin, exist_ok=True)

    print(f"[1] 下载 assistant 轨迹 <- {a.assistant_obs}")
    fetch(a.obsutil, a.assistant_obs, origin, "SSH_PASSWORD_ASSISTANT", obs_cred_args=obs_cred_args)
    print(f"[2] 下载 evaluator 轨迹 <- {a.evaluator_obs}")
    fetch(a.obsutil, a.evaluator_obs, origin, "SSH_PASSWORD_EVALUATOR", obs_cred_args=obs_cred_args)

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
