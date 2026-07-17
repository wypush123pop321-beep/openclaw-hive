# 轨迹处理流水线

从 assistant 轨迹 + 质检(evaluator)轨迹，筛选 → 转 pangu 格式 → 输出统计。
支持两种入口：**从 OBS 下载后处理**（推荐）或 **直接处理本地目录**。

## 入口 A：从 OBS 下载后处理（推荐）

```bash
python download_and_run.py <assistant_obs> <evaluator_obs> <out_dir> [--obsutil PATH]
```

- 用 `obsutil cp -r -f` 把两个 obs 目录下载到 `<out_dir>/origin/`（**实时打印下载速度**）
- 自动定位下载后的两个轨迹目录（含 `session_report.xlsx` 的为 assistant），再跑处理流水线
- `--obsutil` 默认 `D:\tools\obsutil_windows_amd64_5.8.3\obsutil.exe`（obsutil 需已 `config` 好 ak/sk）

示例：

```bash
python download_and_run.py ^
  "obs://rl-agentdata/zhengnianzu/test/session_analysis/env-claude-99oR/key-5c33/ex-260716171238/" ^
  "obs://rl-agentdata/zhengnianzu/test/session_analysis/env-claude-99oR/key-122a/ex-260716170233/" ^
  output
```

## 入口 B：直接处理本地目录

```bash
python run_pipeline.py <assistant_dir> <qc_dir> <out_dir>
```

- `assistant_dir` — assistant 轨迹目录，**必须含 `session_report.xlsx`**；其下每个子目录是一个 session
- `qc_dir` — 质检(evaluator)轨迹目录，其下每个子目录含一份质检 json
- `out_dir` — 输出目录（自动创建）

示例：

```bash
python run_pipeline.py "D:\code\trajs_review\ex-260716171238" "D:\code\trajs_review\ex-260716170233" output
```

## 依赖

- Python 3
- `openpyxl`（读 xlsx）：`pip install openpyxl`
- `obsutil`（仅入口 A 需要）：已配置好凭证（`obsutil config -i=AK -k=SK -e=endpoint`）

## 处理逻辑

1. **筛选**：读 `session_report.xlsx` 的「错误备注」列，保留**为空**或**只有"200空响应"**的 Session（`Session` 列 = assistant 会话子目录名）。
2. **转换**：写 `filtered_sessions.txt`，调 `convert_log.py --session-list` 把筛选后轨迹转 pangu JSONL。
3. **关联评测**：用「首轮 user 内容（去空白）」把每个 session 匹配到质检轨迹的 `【Origin Query】`，取**最早有裁决**的一条作为该 session 的首轮评测结果 `completion`。
4. **统计**：输出 `filter_stats.json`。

## 输出文件（out_dir 内）

| 文件 | 说明 |
|---|---|
| `filtered_sessions.txt` | 筛选保留的 session 名单（每行一个） |
| `pangu_filtered.jsonl` | pangu 格式转换结果（完整对话） |
| `pangu_filtered_truncated.jsonl` | 截断/未完成对话（convert_log 附带产物，可能为空） |
| `pangu_filtered_fold.jsonl` | 含 tool fold 的对话（附带产物，可能为空） |
| `filter_stats.json` | 最终统计 |

## filter_stats.json 字段

| 字段 | 含义 |
|---|---|
| `filtered_count` | 筛选后轨迹数 |
| `with_eval_count` | 有 evaluator 首轮评测结果的数（**含 completion=null 的裁决**） |
| `completion_ge_0.5` | completion 为数值且 ≥0.5 的数 |
| `completion_eq_1` | completion == 1 的数 |
| `pangu_complete` / `pangu_truncated` | pangu 完整 / 截断条数 |
| `dropped_count` | 被筛掉的 session 数 |
| `per_session` | 每个 session 的评测明细（session / has_eval / eval_qc / completion） |

## 文件清单

- `download_and_run.py` — 入口 A：从 OBS 下载后处理
- `run_pipeline.py` — 入口 B：处理本地目录（被入口 A 调用）
- `convert_log.py` — pangu 格式转换器（被流水线调用）
- `README.md` — 本文件

## 输出目录布局（out_dir）

```
out_dir/
├── origin/                      # 入口 A 下载的原始轨迹
│   ├── ex-...(assistant)/
│   └── ex-...(evaluator)/
├── filtered_sessions.txt
├── pangu_filtered.jsonl
└── filter_stats.json
```
