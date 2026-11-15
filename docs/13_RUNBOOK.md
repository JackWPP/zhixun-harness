# 13 · 运行手册（评测方 / 答辩现场）

本手册描述如何加载镜像、无人干预跑完 3 任务 × 10 轮、取回产物，以及常见问题的
处置。所有命令都在评测设备的 Linux/amd64 + Docker 环境执行。

## 13.1 加载与检查

```bash
docker load -i harness.tar
# 期望：Loaded image: zhixun-rsi-harness:0.3.0
docker images | grep zhixun-rsi-harness
docker inspect --format '{{.Architecture}} {{.Os}} {{.Config.Entrypoint}}' zhixun-rsi-harness:0.3.0
# 期望：amd64 linux [/opt/zhixun/bin/entrypoint.sh]
```

镜像不使用 GPU、不需要外网、不写系统目录；唯一可写挂载点是 `/work`（镜像声明
`VOLUME /work`）。

## 13.2 无人干预完整运行（默认行为）

```bash
mkdir -p /host/work
docker run --rm -v /host/work:/work zhixun-rsi-harness:0.3.0
```

等价于：

```bash
docker run --rm -v /host/work:/work \
  -e RSI_TASKS=A,B,C -e RSI_ROUNDS=10 -e RSI_LLM=auto \
  zhixun-rsi-harness:0.3.0 run
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
| `RSI_TOKEN_CAP` | `2000000` | LLM token 上限 |

预算硬上限在代码中：wall 21600s、CPU 36000s、每任务实验 40（合计 ≤120）、Panel 2 次；容器内无法通过
环境变量突破这些安全值。

## 13.4 单独验证某任务 / 复算基线

```bash
# 单任务，2 轮，离线
docker run --rm -v /host/work:/work -e RSI_TASKS=A -e RSI_ROUNDS=2 -e RSI_LLM=off \
  zhixun-rsi-harness:0.3.0 run

# 自检：加载 checkpoint，复算三层切分与探针的主指标
docker run --rm zhixun-rsi-harness:0.3.0 selftest A
```

## 13.5 预测入口（平台 quote 用）

```bash
docker run --rm -i -v /host/work:/work zhixun-rsi-harness:0.3.0 \
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
  zhixun-rsi-harness:0.3.0 serve /work/rsi-runs/latest/state.json 7861
```

浏览器打开 `http://<host>:7861`。看板纯静态 + 标准库，无外网依赖；只读
`state.json`，不参与决策。

## 13.7 时限与资源预期

| 场景 | 本机实测 | 评测机预期 |
|---|---|---|
| 单任务 10 轮（离线确定性） | A ≈ 3–5 min；B/C ≈ 10–15 min | 同量级（CPU 单线程） |
| 3 任务 × 10 轮（离线） | ≈ 25–35 min | < 1h，远低于 6h |
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
