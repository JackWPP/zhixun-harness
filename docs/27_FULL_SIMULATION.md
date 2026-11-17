# 全量批次注入模拟（Full-Scale Batch-Stream Rehearsal）

> 目的：用**与比赛同构的批次协议**把全部可用的历史数据按批送进 harness，
> 观察它能否在“只看得见已揭示历史”的前提下持续诊断、实验、决策并保持/提升指标。
> 这不是安全攻击测试，也不触碰平台隐藏测试。

## 1. 协议（honest batch protocol）

每批 `Batch(index, train, dev, prediction, answers, reveal, meta)`：

| 字段 | 含义 | 谁可见 |
|---|---|---|
| `train` | 官方 train + 之前已揭示批的标签行 | harness |
| `dev` | 历史标签中留出的验证行 | harness（只用于选型/门控） |
| `prediction` | 当前批输入（无标签） | harness |
| `answers` | 当前批真实标签 | 仅事后评分脚本 |
| `reveal` | 当前批标签在预测完成后的释放集合 | 下一批的 train/dev |
| `meta.hidden` | 最终隐藏批：永不 reveal、不进历史 | 仅评分 |

不变式（`scripts/sim/prepare_full_batches.py` 审计 + `tests/test_private_sim.py`）：

1. 当前批标签绝不进入当前批 `train/dev/prediction`；
2. `prediction` id 不重复；
3. hidden 批永不 reveal、永不进入历史；
4. 打乱当前批 `answers` 后 `decision.json` / `predictions.jsonl` 逐字节不变（E0）；
5. 全量行数守恒。

## 2. 数据

| 任务 | 来源 | 批数 | 行数 |
|---|---|---|---|
| A | UCI Gas Drift batch1–10（`extra/uci-gas/Dataset`） | 10 | 13,910 |
| B | 官方 B 包全量 validation（按轴承分组） | 10（8 揭示 + 2 hidden） | 1,679 |
| C | 官方 C 包全量 validation（按发动机单元分组） | 10（8 揭示 + 2 hidden） | 23,049 |

UCI #487 在外网下载服务端不支持 Range、文件损坏，已放弃；XJTU-SY 原始振动需 Google
Drive/Dropbox 且要自建 34 维特征，未纳入本轮。两者均记录在案，不阻塞结论。

## 3. L1 —— 协议级全量流

`scripts/sim/run_l1_full.py`：每个 `(task, arm, seed)` 起一个独立 `rsi.cli stream --source full` 进程，
完整跑 10 批（A）或 8+2 批（B/C），逐批预测后再评分，结束后删除批次临时包、保留
`lane_summary.json` / `stream_log` / `stream_report.json`。

主要结果（arrival 批均值；A 越高越好，B/C RMSE 越低越好）：

| 任务 | frozen | recenter | pair | adapt | full |
|---|---|---|---|---|---|
| A | 0.4862 | 0.5589 | 0.5276 | 0.7385 / 0.7625 | **0.7653 / 0.7684** |
| B | 650.02 | 572.65 | – | **435.56 / 432.40** | 497.01 / 502.01 |
| C | **50.37** | 50.94 | – | 51.08 / 51.03 | 未跑（成本高） |

结论：
- A：history-adapt 把均值从 0.486 拉到 0.74+，full arm（Runner 每批实验+门控）再小幅提升，且 worst-case 更好（0.62 vs 0.56）。
- B：adapt 最好（650→433），full 因 selection 噪声反而差；与 private-sim 结论一致。
- C：无漂移时 frozen 最好，adapt 略微伤指标；路由默认 C=frozen 得到验证。

### 3.1 变体（A）

| 变体 | 设置 | frozen/参照 | adapt | full |
|---|---|---|---|---|
| scarcity | label_every=2, label_fraction=0.5 | – | 0.7199 | 0.7068 |
| smallb | 4 批（每批更大） | 0.6883 | 0.7309 | **0.7330** |
| long | b1–b10 两轮，共 20 批 | 0.4862 | **0.8221** | **0.8791** |

标签减半只下降约 0.02–0.04，仍然远高于 frozen；长流上 adapt 达到 0.82（第二轮几乎全对），
说明历史 replay 不会遗忘，反而持续积累。

## 4. L2 —— workspace-v2 真实入口

`scripts/sim/run_l2_batches.py` 复用 `scripts/emulate_workspace_v2.py` 的 fake host：
每批构造 package（`/data/train`=历史 train、`/data/validation`=历史 dev、当前批 inputs 单独存放）→
`python3 /app/harness.py train` → 校验 artifact → 对全部 prediction 行跑 `harness.py predict` →
事后评分。**走的是与平台相同的 v2 协议入口，而不是内部 CLI。**

A/B/C 各 10 轮全部 PASS（逐轮指标见 `docs/evidence/full_sim/full_sim_report.md`）。观察：
- A：round1 0.956（batch1 简单）、round2 0.714、round3 0.594… 与 UCI 漂移趋势一致；pair 阈值在历史 AA 积累后生效（batch5 历史 362 条 AA 时 dev primary 0.7842→0.8481）。
- B：逐批 RMSE 波动大（348→1252→246），因为每个 hidden 批对应的轴承/工况不同；需要与 L1 frozen 同批比较（报告已给出两套表）。
- C：稳定在 37–57 循环 RMSE，无漂移时 v2 路径不引入额外损失。

## 5. L3 —— kernel-LLM 批次流

`scripts/sim/run_l3_kernel.py`：A 4 批、B 2 批、C 2 批，`--controller kernel --token-cap 0`，
让 LLM Agent 在每批内自主诊断/实验/决策/写记忆。预算放开（开发期不设 token 上限）。
结果见 `docs/evidence/full_sim/full_sim_report.md` 与 `.scratch/sim/L3*/`。

A 任务（2 个 arrival 批，kernel 控制器）：两批均 **promote**，逐批预测 macro-F1
0.9558 / 0.7185；batch0 的 kernel 轨迹为 18 次工具调用、166,936 tokens（诊断→3 候选→
执行→决策→记忆），batch1 因模型调用异常自动降级 pipeline（受控降级生效）——
这正是"模型失败有确定性兜底、不中断批次流"的证据。

B/C 各补 1 个 arrival 批（`--hidden-tail 0`）：

| 任务 | 决策 | dev primary（历史 holdout） | 本批预测 primary | wall |
|---|---|---|---|---|
| B（RMSE） | promote | 223.13 | 679.56（该批 167 行、天然难） | 231s |
| C（RMSE） | promote | 54.34 | 51.69 | 407s |

结论：kernel 控制器在三个任务上都能完成"观察→诊断→提案→执行→门控→记录"整环；
A 的外部批次收益大，B/C 在单批上决策按 dev 门控（B 的预测值偏高，说明仅凭单批
dev 选型在回归任务上仍有冒烟风险，与 L1 的结论一致）。

另用赛事指定的 **PyTRiO** OpenAI 兼容端点（`https://pytrio.com/api/openai/v1`，
模型 `Qwen/Qwen3.8-27B`）做了真实驱动验证：

- 第一次 A 1 轮：模型自主提出 3 个候选，其中 2 个 augmentation 组合暴露了 `SIM-009`
  的策略库崩溃（已修复），最终 rollback，97,755 tokens / 2 次调用；
- 修复后第二次 A 1 轮：模型自主完成诊断→3 个候选→比较→promote，
  selection **0.7032→0.7234（+0.0202）**，选中 recenter(batch_std, w=1.0)，
  104,609 tokens / 2 次调用、12 个工具调用，decision/parent/candidate sha 记录完整。

该验证同时确认：只改 `LLM_BASE_URL/LLM_API_KEY/LLM_MODEL` 三个环境变量即可从 tokenplus
切到 PyTRiO，镜像内无需改动；v2 评测时模型调用由宿主 `rnd.model()` 注入，Key 不落盘。

## 6. 测试中发现并修复的问题

见 `.scratch/sim/issues.jsonl` 与报告尾部，摘要：

| id | 问题 | 修复 |
|---|---|---|
| SIM-001 | A 流每类丢 1 行（13,853/13,910） | `_shuffle` 替代 `_strat(...,1.0)`；行数守恒 PASS |
| SIM-002 | L2 fake-host predict 只跑 3 行，无法整批评分 | `_predict_all` 全量预测 + 官方格式校验 |
| SIM-003 | v2 适配层没有 A 的 pair_threshold 候选 | 增加 FIT_PAIR 候选（历史标签拟合，None 回退 none） |
| SIM-004 | `--token-cap 0` 被当成“预算耗尽”，kernel 静默降级 | `TokenBudget` cap<=0 = unlimited |
| SIM-005 | 20 批长流 full lane 与其它作业并发时被 SIGKILL | 单独串行重跑（并行=1） |
| SIM-006 | A 第一批 selection≈1.0，gate 早期偏 promote | 记录为选择偏差，报告区分早期/hidden 指标 |
| SIM-007 | 长流 full 在 batch10 因仓库文档改名报 `FileNotFoundError: docs/27_PPT_REPORT.md` | `code_sha256` 对消失文件跳过；新增回归测试；重跑 lane |
| SIM-008 | repeat=2 长流在 batch11 报重复 sample id（官方 loader 拒绝） | `_add_history` 按 id 去重；新增回归测试；重跑长流 |
| SIM-009 | PyTRiO 真实模型触发的 augment+weight 组合崩溃（`max(标量, 数组)` ambiguous，且扩增后 x/y 不对齐） | `_augment_features` 改 `np.maximum` 并同步扩增标签；新增回归测试 |
| SIM-010 | 本地 v2 emulator 未把上一轮候选挂载为下一轮 `/previous`：A/B（retain，checkpoint 字节不变）通过，C（continue_train，重新训练出不同 checkpoint）第二轮报 `PARENT_DIGEST_MISMATCH` | emulator 增加 `install_previous()`，每轮把选中产物复制为下一轮父挂载；C 10 轮 PASS（0.10.2 代码） |
| SIM-011 | 平台冻结契约把 B 数据集标成 classification/macro_f1，而 B 官方包是 regression/RUL+RMSE；0.10.1 严格跟随 `rnd.task` 导致 B 第一轮在模型调用前抛 `checkpoint task mismatch`，整 run 被判 contestant 失败 | 0.10.4 起训练跟随 checkpoint task（stderr 警告）、predict 增加 task 形状兜底；平台复审的 857 字完整失败详情已归档；待重跑 trial 验证 |

## 7. 复现

```bash
# 数据清单 + 审计
PYTHONPATH=. ./.venv/bin/python scripts/sim/prepare_full_batches.py

# L1 主矩阵（A/B/C，arms×seeds）
PYTHONPATH=. ./.venv/bin/python scripts/sim/run_l1_full.py \
  --tasks A,B,C --jobs 3 --seeds 42,43 --root .scratch/sim/L1

# 变体
PYTHONPATH=. ./.venv/bin/python scripts/sim/run_l1_full.py --tasks A \
  --arms-A adapt,full --label-every 2 --label-fraction 0.5 --seed-tag scarcity

# L2 workspace v2
PYTHONPATH=. ./.venv/bin/python scripts/sim/run_l2_batches.py --tasks A,B,C

# L3 kernel
PYTHONPATH=. ./.venv/bin/python scripts/sim/run_l3_kernel.py

# 汇总
PYTHONPATH=. ./.venv/bin/python scripts/sim/report_full.py
```

## 8. 限制与下一步

- fake host 复刻的是 SDK 1.0.2 的协议与校验，不包含平台宿主的真实资源限制/超时行为；
  真实平台只能在拿到 PyTRiO Key 后 trial 验证。
- L2 的 B/C 每批 hidden 样本少（167 行/轴承组），批次间差异大，结论应以 L1 的 10 批曲线为主。
- L3 的 kernel LLM 延迟高（每批分钟级），本轮只做能力演示，不作为评分路径。
- 未做：真实 4C8G 容器压测、恶意输入/fuzz（按用户要求不做安全攻击）。
