# 22 SwanLab 实验记录（工业模型训练过程）

赛题要求：所有实验按《实验记录规范》在 SwanLab 云端完整记录并公开，至少 3 次
“读历史 → 假设 → 实验”的迭代；记录对象是**工业模型训练过程**（不是驱动 Harness 的大模型）。
本仓库的处理方式：

* 每个 `rsi.cli run` 都把每轮的 hypothesis / change / metrics / parent-candidate sha /
  decision 以指标 + 文本形式记录；
* `--swanlab` 开启 **offline 模式**（镜像默认离线，符合评测期无外网要求）；
* 离线 run 目录可直接由操作者用 `swanlab sync` 上传到云端项目（token 只走环境变量，
  绝不写进仓库/镜像/日志）。

## 本地用法

```bash
# 离线记录（比赛镜像内默认行为）
./.venv/bin/python -m rsi.cli run --tasks A,B,C --rounds 10 --llm off \
    --swanlab --state-dir .scratch/rsi-runs

# 生成的本地 run：
#   .scratch/rsi-runs/<run_id>/swanlab/run-<ts>-<id>/run-<id>.swanlab
```

冒烟证据（2026-11-15 23:48）：`rsi-20261115-234801`（A 1 轮，2 实验，promote recenter
0.720696）生成 `.scratch/swan-smoke/rsi-20261115-234801/swanlab/run-20261115_234801-40t5qr66/`，
包含 config/metrics/run 文件；`SWANLAB_MODE=offline` 时无任何网络请求。

## 上传/公开（操作者执行，需要 SwaLab token）

```bash
export SWANLAB_API_KEY=<your-token>          # 不要写入仓库/镜像/聊天日志
swanlab sync .scratch/rsi-runs/<run_id>/swanlab/run-*   # 上传离线 run
# 或在线模式：SWANLAB_MODE=online SWANLAB_API_KEY=... rsi.cli run --swanlab ...
```

## 记录内容（每轮）

| 字段 | 来源 |
|---|---|
| 任务/轮次/决策 | `rounds.jsonl`：`index/decision/decision_reason/delta` |
| 假设与变化 | `hypothesis`、`change_summary`（Agent 或确定性 fallback） |
| 指标 | `validation_metrics`（selection/report），`drift` 摘要 |
| 模型谱系 | parent/candidate/checkpoint/code sha256 |
| 资源 | `cpu_seconds`、`wall_seconds`、`experiments`、`llm` usage/token |
| 记忆/知识 | `memory_add` 次数、`search_knowledge` 检索次数（kernel 模式） |

## 考核对照

* ≥3 次迭代：严格口径 3×10 run（见 `docs/12_EXPERIMENTS.md` §12.19）与 kernel A 3 轮
  （`docs/18_AGENT_KERNEL.md` §10.3）都远超 3 次；
* “公开”：需操作者用队伍账号 sync（本机开发阶段保持离线，不上传）；
* 计量：SwanLab 记录不含隐藏标签；token 用量在 `summary.json.llm.budget` 与 SQLite
  `llm_calls` 中可审计。
