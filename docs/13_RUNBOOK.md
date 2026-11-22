# 13 · 运行手册（评测方 / 答辩现场）

本手册描述如何加载镜像、无人干预跑完 3 任务 × 10 轮、取回产物，以及常见问题的
处置。所有命令都在评测设备的 Linux/amd64 + Docker 环境执行。

## 13.1 加载与检查

```bash
docker load -i harness.tar
# 期望：Loaded image: zhixun-rsi-harness:0.14.9
docker images | grep zhixun-rsi-harness
docker inspect --format '{{.Architecture}} {{.Os}} {{.Config.Entrypoint}}' zhixun-rsi-harness:0.14.5
# 期望：amd64 linux [/opt/zhixun/bin/entrypoint.sh]
```

镜像不使用 GPU、不需要外网、不写系统目录；唯一可写挂载点是 `/work`（镜像声明
`VOLUME /work`）。

## 13.2 无人干预完整运行（默认行为）

```bash
mkdir -p /host/work
docker run --rm -v /host/work:/work zhixun-rsi-harness:0.14.5
```

等价于：

```bash
docker run --rm -v /host/work:/work \
  -e RSI_TASKS=A,B,C -e RSI_ROUNDS=10 -e RSI_LLM=auto \
  zhixun-rsi-harness:0.14.5 run
```

完成后在宿主机可见：

```
/host/work/rsi-runs/<run-id>/
  state.json            # 看板数据源（每轮原子更新）
  summary.json          # 全局摘要（三任务）
  workspace_report.json # 平台 record 导出（schema 对齐）
  records.jsonl         # 全事件流水（轮次/实验/LLM/Panel）
  rounds.jsonl          # 每轮假设/变更/指标/父子 sha
  A/ B/ C/
    summary.json        # 任务级摘要 + narrative
    rounds.jsonl        # 任务级轮次
    experiments/exp-NNNN/{job.json,candidate.npz,result.json,worker.log}
/host/work/summary.json          # 固定路径副本
/host/work/workspace_report.json # 固定路径副本
/host/work/rsi-runs/assets-check.json  # 启动自检结果
<state-dir>/memory/               # 跨 run 记忆（MEMORY.md + TASK_*.md）
<state-dir>/rsi_state.db          # 跨 run SQLite（rounds/experiments/lessons/strategies）
```

## 13.3 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RSI_TASKS` | `A,B,C` | 任务子集，逗号分隔 |
| `RSI_ROUNDS` | `10` | 每任务轮数 |
| `RSI_LLM` | `auto` | `auto`：有 key 则用 LLM，否则确定性；`on` 强制；`off` 完全离线 |
| `RSI_LLM_ENV` | 空 | 挂载的 KEY=VALUE 文件（如 `/work/llm.env`），镜像内不含凭据 |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | 空 / 空 / `qwen/qwen-3.8-27b` | OpenAI 兼容端点；评测时可不提供 |
| `RSI_STATE_DIR` | `/work/rsi-runs` | 产物根目录 |
| `RSI_SEED` | `42` | 切分/漂移窗口/实验种子 |
| `RSI_SWANLAB` | `0` | `1` 时启用 SwanLab 离线记录（`SWANLAB_API_KEY` 另给） |
| `RSI_EXPERIMENT_TIMEOUT` | `1800` | 单实验秒数上限 |
| `RSI_CONTROLLER` | `auto` | `auto`：有 LLM 用 kernel，否则 pipeline；`kernel` 强制自主内核；`pipeline` 确定性 |
| `RSI_VERIFY_ASSETS` | `1` | 启动前自检数据集/checkpoint/技能/predict 入口（失败即退出） |
| `RSI_ROUND_TURNS` | `20` | kernel 单轮 PI loop 最大 turn 数 |
| `RSI_TOKEN_CAP` | `2000000` | LLM token 上限（开发期可用 `--token-cap 0` 不限） |

预算硬上限在代码中：wall 21600s、CPU 36000s、每任务实验 40（合计 ≤120）、Panel 2 次；容器内无法通过
环境变量突破这些安全值。

## 13.4 单独验证某任务 / 复算基线

```bash
# 单任务，2 轮，离线
docker run --rm -v /host/work:/work -e RSI_TASKS=A -e RSI_ROUNDS=2 -e RSI_LLM=off \
  zhixun-rsi-harness:0.14.5 run

# 自检：加载 checkpoint，复算三层切分与探针的主指标
docker run --rm zhixun-rsi-harness:0.14.5 selftest A
```

## 13.4b v0.4 新增命令

```bash
# Agent 自主内核（需要可用 OpenAI 兼容端点；单轮 2-4 分钟）
docker run --rm -v /host/work:/work -e RSI_LLM=on -e RSI_LLM_ENV=/work/llm.env \
  -e RSI_CONTROLLER=kernel -e RSI_TASKS=A -e RSI_ROUNDS=3 \
  zhixun-rsi-harness:0.14.5 run

# 私有/隐藏风格 JSONL 探针（只推理，不训练）
python -m rsi.cli probe --task A --path private.jsonl --checkpoint <ckpt> --output probe.json

# 合成漂移探针（开发/回归用）
python -m rsi.cli synth --task A --batches 4 --output .scratch/synth_a.jsonl

# 外部独立验证（UCI 逐批）
python scripts/external_validation.py --checkpoint <ckpt> --summary <summary.json> \
  --output docs/evidence/uci_gas_drift.md

# 交付前自检与提交包
python scripts/verify_assets.py --json-out .scratch/assets.json
python scripts/make_submission.py --run-dir .scratch/rsi-canonical/<run-id>   # -> dist/submission_<run>/ + tar.gz

# 完整彩排（构建镜像 -> 解包 -> 镜像内 2 轮；--full 额外 3x10）
bash scripts/rehearse.sh --skip-build
```

## 13.5 预测入口（平台 quote 用）

```bash
docker run --rm -i -v /host/work:/work zhixun-rsi-harness:0.14.5 \
  predict --checkpoint /work/A/checkpoint.npz \
          --postprocess /work/A/postprocess.json --mode batch < /host/inputs.jsonl
```

- 输入：每行 `{"task":"A","inputs":{"features":[128 个 float]}}`；
- 输出：分类 `{"outputs":[{"label":"Ethanol","score":0.93},...]}`、回归
  `{"outputs":{"value":682.5}}`，逐行 flush，顺序与输入一致；
- `--mode batch`（默认）先读全再整批 recenter，与平台批评测一致；`--mode stream`
  用因果滚动窗口，适合在线逐条。

## 13.6 启动看板（可选）

```bash
docker run --rm -p 7861:7861 -v /host/work:/work \
  zhixun-rsi-harness:0.14.5 serve /work/rsi-runs/latest/state.json 7861
```

浏览器打开 `http://<host>:7861`。看板纯静态 + 标准库，无外网依赖；只读
`state.json`，不参与决策。

## 13.7 时限与资源预期

| 场景 | 本机实测 | 评测机预期 |
|---|---|---|
| 单任务 10 轮（离线确定性） | A ≈ 3–5 min；B/C ≈ 10–15 min | 同量级（CPU 单线程） |
| 3 任务 × 10 轮（离线） | ≈ 20–25 min（v0.5：1390s、52 实验、CPU 752s） | < 1h，远低于 6h |
| 3 任务 × 10 轮（LLM on） | ≈ 1.5–2.5h（每次调用 15–110s） | 建议预留 3h；预算不足自动降级 |
| 镜像大小 | 366 MB（load 后） | 上传要求未压缩 tar ≈ 375 MB |
| 磁盘写入 | 每实验 < 2 MB；全量 < 200 MB | 建议 `/work` ≥ 2 GB |

## 13.8 常见问题

| 症状 | 原因 | 处理 |
|---|---|---|
| `llm_available=false`，全链路仍完成 | 未提供端点/key | 正常降级；要 LLM 就挂 `RSI_LLM_ENV` |
| 某轮 `decision=retain`，指标不动 | 候选未超噪声阈值 | 预期行为；A/B 早期会连续 retain |
| 实验 result 有 `error` | 候选参数/数值问题 | 已 catch，不影响后续轮次；详情见 `worker.log` 与 `result.json` |
| `ModuleNotFoundError: numpy` | 旧镜像的 PYTHONPATH bug | 已修复（executor 前置继承 PYTHONPATH），用 0.3.0+ 重建镜像 |
| 看板无数据 | `--state` 未指向运行中的 `state.json` | 指向 `rsi-runs/<id>/state.json` |
| 想要更保守 | 环境 | `RSI_LLM=off` + `RSI_ROUNDS=2` 先冒烟 |

## 13.9 提交材料对应

1. `harness.tar`（本镜像）→ 平台 `POST /image-archives`（`application/x-tar`）；
2. `workspace_report.json` + `records.jsonl` → 云端实验记录；
3. 本手册 + `docs/12_EXPERIMENTS.md` → 系统运行说明 / 技术报告素材；
4. 最终模型 `checkpoint.npz` + `postprocess.json` + 父 sha 链 → 模型配置。

## 13.10 v0.5 新增接口（可复用运行时 / 数据能力）

```bash
# 列出全部已注册任务（内置 + tasks/*/task.json + --task-spec）
python -m rsi.cli tasks

# 用外部 task.json 跑一个新数据集（包契约见 docs/PORTING.md）
python -m rsi.cli run --tasks D --task-spec path/to/task.json --rounds 3 --llm off

# 私有/隐藏风格数据只做推理评估（不进训练）
python -m rsi.cli probe --task A --path private/validation.jsonl \
    --checkpoint datasets/cpu-a-guide-20261110-v1/checkpoint.npz

# 合成漂移数据（固定种子，可复现）
python -m rsi.cli synth --task A --batches 3 --drift 2.5 --noise 0.25 \
    --output .scratch/synthetic/a.jsonl
```

外部预训练（任务 A 的 UCI batch3–7）在镜像中默认**关闭**（`RSI_ALLOW_EXTENDED_DATA=0`），
正式评分只拟合官方 train；需要复现扩展消融时显式设 `RSI_ALLOW_EXTENDED_DATA=1`。
验证/隐藏标签永不参与拟合；训练隔离由 Guard 与 `rsi/policy.py` 双重强制。

## 13.11 批次流模式（stream）

平台按批次下发数据、每批带标签、先给决策才给下一批。镜像 entrypoint 已接好：

```bash
docker run --rm -v /host/work:/work -e RSI_STREAM_SOURCE=official \
  -e RSI_STREAM_BATCHES=10 zhixun-rsi-harness:0.14.5 stream
```

* 每批产物：`$RSI_STREAM_DIR/tasks/<key>/batch_XXXX/{decision.json,predictions.jsonl,run/}`；
  `decision.json` 同时含扁平平台字段（`decision` 字符串、`round_index`、父/候选/最终
  checkpoint sha256、`validation_metrics`）与 `decision_detail`；
* **历史标签自适应默认开启**（`RSI_STREAM_ADAPT=1`）：第 i 批先用“官方 train + 前 i-1 批
  已返回标签”连续训练 30 epoch（lr 5e-4），再做当前批 recenter（w 0.5）；不会使用当前批
  标签。可调 `RSI_STREAM_ADAPT_EPOCHS/LR/WEIGHT`，离线对照可设 `RSI_STREAM_ADAPT=0`；
* 本地/开发：`python -m rsi.cli stream --source official --tasks A,B,C --batches 10`；
* 外部数据测试床：`--source external-a --external-root extra/uci-gas/Dataset`（Track X，
  默认不进入正式谱系；需 `RSI_ALLOW_EXTENDED_DATA=1`；外部批的标签在下一批才进入训练）；
* 平台真实挂载路径未知（docs/19 H1/H2），上线前用 `scripts/emulate_stream_platform.py`
  彩排，并用 `scripts/contract_check.py --online` 复核规则/公告。

## 13.12 评测资源约束（来自官方答疑）

| 约束 | 值 | 我们的应对 |
|---|---|---|
| CPU / 内存 | 4 核 / 8 GiB | OMP/OPENBLAS/MKL 线程=1；实验串行或 `--parallel-tasks ≤3` |
| GPU | 无 | 全 NumPy，无 GPU 依赖 |
| 镜像大小 | ≤2 GB | 376 MB tar（load 后 367 MB） |
| 外网 | 评测期禁网，仅 PyTRiO | LLM 只在 `RSI_LLM_ENV` 提供时启用；知识库/技能/数据全部离线 |
| 大模型 | 仅 PyTRiO Qwen3.8-27B / 3.5-4B | 通过 OpenAI 兼容 env 切换，不打包权重 |
| 时限 | 6 h（wall_seconds=21600） | 时间感知降级：<1500s 剩余时每轮 1 实验、缩 round-turns，保证输出 |

- 批次流默认开启延迟标签校准 `pair_threshold`（A：Ethanol vs Acetaldehyde）。用已返回的历史批标签拟合判定阈值，仅作用于之后批次；关闭：`--no-pair-calibration`；镜像环境变量 `RSI_STREAM_PAIR=0/1`、`RSI_STREAM_PAIR_CLASSES=0,3`。


## 13.13 Workspace v2（平台评测路径）

平台 `execution_protocol=contest.harness.workspace.v2`，镜像固定入口改为
`python3 /app/harness.py describe|train|predict`。镜像同时保留自研 CLI；二者的对应关系、
本地 fake-host 验证命令见 `docs/26_WORKSPACE_V2.md`。

```bash
# 诊断入口（与平台试运行诊断同一调用）
docker run --rm zhixun-rsi-harness:0.14.5 describe
docker run --rm -v /host/work:/work zhixun-rsi-harness:0.14.5 run   # 旧版本地 RSI 运维路径
```

## 13.14 PyTRiO 端点（赛事驱动模型）

平台试运行/正式评测时，驱动 Harness 的大模型由平台通过 PyTRiO 提供（OpenAI 兼容）。
本地开发也可以直接接同一个端点，验证 prompt 兼容性与时延：

```bash
# .scratch/pytrio.env（不要提交密钥；模板见 config/pytrio.env.example）
# LLM_BASE_URL=https://pytrio.com/api/openai/v1
# LLM_API_KEY=trio_xxx
# LLM_MODEL=Qwen/Qwen3.8-27B
# LLM_CONTEXT_WINDOW=64000

RSI_TOKEN_CAP=0 ./.venv/bin/python -m rsi.cli run \
  --tasks A --rounds 1 --llm on --llm-env .scratch/pytrio.env \
  --controller kernel --round-turns 6 --experiments-cap 3 \
  --state-dir .scratch/rsi-pytrio
```

- `/api/openai/v1/models` 实测返回 `Qwen/Qwen3.8-27B`、`Qwen/Qwen3.5-4B`；
- `Qwen/Qwen3.5-4B` 在小 `max_tokens` 下会全部消耗在 reasoning、content 为空，建议
  驱动 Harness 使用 `Qwen/Qwen3.8-27B`（平台允许列表内）；
- 评测镜像内不保存 Key：v2 协议由宿主 `rnd.model()` 注入模型调用，Key 仅由平台/本地 env 使用。
