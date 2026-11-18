# 33 · 多轮漂移策略实验计划（全真模拟）

> 状态：执行中。结果与结论见 `docs/34_EXPERIMENT_REPORT.md`（实验完成后生成）。
> 数据来源：`scripts/sim/campaign.py` 产出的 `.scratch/sim/runs/*/summary.json`、
> `.scratch/sim/campaign-manifest.jsonl`、SwanLab 项目 `zhixun-rsi`。
> 所有「hidden」指标均为公开 validation 尾段留出的**模拟代理**，不是平台隐藏测试流。

## 1. 实验目的

上一轮已经证明：gen2 漂移策略把 A 任务从「每轮 retain」的 0.3330 拉到 0.4802
（3 轮）与 0.4802（10 轮），与平台 trial#3 的 0.4768 同量级。本轮要回答四个问题：

1. **稳定性**：A 上 policy on 相对 off 的收益是否跨种子成立（10 轮，3 次重复）？
2. **超参敏感性**：历史 pair delta 是否在 ±0.25 内稳定，还是刀尖上的运气？
3. **误伤/泛化**：B、C、外部 UCI gas、私有 schema 场景是否会被策略误伤或额外获益？
4. **鲁棒性**：predict 面对 50 行畸形输入时是否仍逐行输出合法 JSON、不崩溃？

非目标：不上传平台、不跑 trial、不声称线上分数；不修改 checkpoint 结构。

## 2. 假设与判定

| 编号 | 假设 | 判定 |
|---|---|---|
| H1 | A policy-on 10 轮 hidden 均值 ≥ policy-off + 0.10 | 3 个种子全部为正，均值差 ≥0.10 |
| H2 | 最优 delta 附近不崩 | 相邻 delta 的 hidden 差 ≤0.05，种子间 std ≤0.04 |
| H3 | schema 漂移场景 policy-on 提升 ≥0.10；external 变化在 ±0.03 内 | 对应场景 csv 对比 |
| H4 | B/C 不因策略误伤 | 与对照臂相比退化 ≤2%，协议零失败 |
| H5 | 畸形 predict 行 100% 合法 | 50/50 行输出合法 JSON、退出码 0 |

## 3. 实验矩阵

| 编号 | 场景 | 任务 | 轮数 | 策略/臂 | 种子 | run 数 |
|---|---|---|---|---|---|---|
| C0 | official（fake provider 冒烟） | A | 2 | policy off/on | 1 | 2 |
| C1 | official | A | 5 | policy off/on | 1,2 | 4 |
| C2 | 离线策略实验室 sweep | A | 3 | delta ∈ {-2,-3,-3.5,-4,-4.5} × margin×{1,2,3} | 确定性 | 1 次扫描 |
| C2b | official | A | 5 | delta -3.75 / -4.0 | 1,2 | 4 |
| C3 | offline arm report + official | B | 5 | shrink/damage/retain | 1 | 1 |
| C4 | offline arm report + official | C | 5 | continue-20/40, retain, shrink | 1 | 1 |
| C5 | external（UCI gas） | A | 4 | policy off/on | 1 | 2 |
| C6 | schema（私有字段改名） | A | 3 | policy off/on | 1 | 2 |
| C7 | anomaly predict | A | — | 50 行畸形输入 | — | 1 |

> 现实约束：PyTRiO 网关单轮真实调用在 1–90 s 间波动，kernel 路由每轮 6 次调用会让
> 完整矩阵膨胀到数小时。因此**因果对比矩阵使用 pipeline 路由**（每轮 1 次真实调用，
> 策略覆盖逻辑完全相同、确定性更强）；kernel 路由的真实 Pi-Agent 端到端证据单独保留
> （既有 0.11.0 的 9 轮/30 轮 kernel run，以及本轮 C1 的 kernel 两轮试点），不混入对比。

除 C2/C3/C4 的离线部分外，全部通过真实 PyTRiO `Qwen/Qwen3.8-27B` 网关调用；对比矩阵每轮 1 次真实
`rnd.model` 调用（pipeline 路由），kernel 路由的每轮 6 次真实调用证据单独保留；SwanLab `mode=online`（无 key 时自动 offline）。

## 4. 代码与工具

- `rsi/platform/submission.py`：`RSI_V2_DRIFT_POLICY=0/1` 消融开关（默认 on）。
- `rsi/platform/plans.py`：`RSI_V2_PAIR_DELTA` 覆盖历史 delta（默认 -4.0）。
- `scripts/sim/full_env.py`：`--policy`、`--pair-delta`、`--hidden-every-round`、`--tag`；
  run 目录与 SwanLab run 名称带 tag，summary 记录 policy/delta。
- `scripts/sim/campaign.py` + `campaigns.json`：矩阵展开、manifest、断点续跑、超时与重试。
- `scripts/sim/strategy_lab.py`：`--sweep`（delta×margin 扫描）、`--arms-report`（每臂
  validation + hidden 一次评估）。
- `scripts/sim/analyze.py`：CSV/Markdown/手写 SVG 汇总。
- `scripts/sim/anomaly.py`：畸形 predict 行鲁棒性探针。

## 5. 验收标准

- 每个 run：kernel 每轮 6 次真实调用、协议 PASS、无 traceback、summary + SwanLab URL 齐全；
- C1：policy-on 均值 ≥ off + 0.10，3/3 种子为正；
- C2b：最优配置种子间 std ≤0.04；
- C5：schema 提升 ≥0.10；external 变化 ≤0.03；
- C7：50/50 合法 JSON、0 崩溃；
- `pytest -m "not integration"` ≥150 passed；emulator A/B/C 10 轮 PASS；
- 有代码改动则重建 0.11.1 镜像 + `verify_image.sh` PASS（**不上传**）。

## 6. 执行顺序

1. **P0**：开关、full_env 新参数、campaign/analyze/anomaly、单测与 C0 冒烟。
2. **P1**：离线 sweep + arms report，确定 C2b/C3/C4 候选。
3. **P2**：C1 消融 6 run，分析 A 的 10 轮学习曲线。
4. **P3**：C2b + C3/C4 full 确认 + C5 external + C6 schema + C7 anomaly。
5. **P4**：`analyze.py` 汇总 → `docs/34_EXPERIMENT_REPORT.md`，更新 docs/32，commit。
6. **P5**：重建 + verify 0.11.1（仅当代码改动），commit；平台动作留给用户确认。

## 7. 风险与处置

- **429/超时**：顺序执行、每 run 重试 1 次、900s 超时；连续失败暂停并报告。
- **hidden 代理局限**：所有结论标注「模拟代理」；不写线上承诺。
- **种子方差**：temperature=0，A 用 3 个种子；报告均值±std。
- **耗时**：manifest 断点续跑；单 run 失败不阻塞后续。
- **Key 安全**：只从 `.scratch/pytrio.env` / `.scratch/swanlab.env`（0600）读取，
  不入库、不入镜像、不打印。

## 执行状态（2026-11-18 晚）

矩阵已跑完 14 个 full_env run（C0 2、C1 4、C2b 4、C3 1、C4 1、C5 2、C6 2 中的 off/on 部分；
C5 on、C6 on 已含），另含 2 个 kernel 试点与 1 个 pipe-test；全部真实 PyTRiO 调用、
SwanLab online。结果与判定见 `docs/34_EXPERIMENT_REPORT.md`。
