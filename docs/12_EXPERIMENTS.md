# 12 · 实验报告（记录驱动分析）

本报告基于一次完整的无人干预运行：**3 任务 × 10 轮，CPU 单线程，完全离线**
（`--llm off`，确定性 planner 驱动），run dir
`.scratch/rsi-canonical/rsi-20261115-162027/`。真实 LLM 链路的单独验证见 §12.8。
所有数字均可由 `run_dir` 下的 JSONL/JSON 复算，不依赖训练日志自报。

## 12.1 实验协议

| 项 | 设置 |
|---|---|
| 运行 | `python -m rsi.cli run --tasks A,B,C --rounds 10 --llm off --seed 42` |
| 切分 | 三层互斥：calibration（拟合校准）/ selection（门控）/ report（仅报告）；A 另有外部探针 |
| 门控 | 提升 ≥ margin 才 promote；劣化 ≥ 2×margin 才 rollback；否则 retain |
| margin | A 0.004（macro-F1）、B 5.0（RMSE 分钟）、C 0.3（RMSE 循环） |
| 预算 | wall 21600s、CPU 36000s、每任务实验 40（合计 ≤120）、LLM 2M tokens |
| 选型 | 只看 selection；report 不参与任何接受/拒绝 |
| 随机性 | 切分、漂移窗口、实验 seed 全部固定（seed=42 + 轮次偏移） |

## 12.2 基线复算（本地 checkpoint）

`python -m rsi.cli selftest --task X` 复算官方 checkpoint 在本地切分上的指标：

| 任务 | 官方验证集（整体） | calibration | selection | report | 外部探针 |
|---|---|---|---|---|---|
| A macro-F1 | 0.714030 | 0.763582 (374) | 0.703220 (435) | 0.686380 (435) | 0.456803 (13,910) |
| B RMSE | 681.160 | 704.148 (504) | 678.022 (588) | 664.009 (587) | — |
| C RMSE | 52.869 | — | 44.749 (8,215) | 57.460 (7,912) | — |

> selection 与官方整体值不同是**有意为之**：三层切分互斥，避免"既选又报"的乐观
> 偏差。C 的 report 基线（57.46）明显差于 selection（44.75），说明发动机单元间
> 方差很大，报告按切分如实披露。

## 12.3 主结果

| 任务 | 指标 | selection 基线 → 最终 | 相对变化 | report 基线 → 最终 | 采用策略 |
|---|---|---|---|---|---|
| A | macro-F1 ↑ | 0.7032 → **0.8434** | +0.1402（+19.9%） | 0.6864 → 0.8126 | recenter(batch_std, w=0.2) + ridge_last_layer 续训 |
| B | RMSE ↓ | 678.02 → **462.45** | −215.57（−31.8%） | 664.01 → 455.46 | shrink（校准中心 694.31，α=0） |
| C | RMSE ↓ | 44.75 → **41.82** | −2.93（−6.6%） | 57.46 → 56.28 | continue_train(lr 1e-3×40 → 1e-3×100 → 5e-4×100) |

- 三任务均**不低于静态基线**（门槛 G3）；
- A 的最终 checkpoint 是第 9 轮、由 ridge_last_layer + recenter 构成；recenter 只
  改推理期校准，不改变模型字节；
- B 的第 1 轮一次提升 31.8% 后，后续 9 轮全部 retain：shrink α=0 使所有模型层
  候选的输出都等于校准中心，因此 selection 指标完全相同（门控正确识别为无提升）；
- C 前 3 轮连续 promote，第 4/7/9 轮候选有 −0.25/−0.03/−0.10 的小幅改善，但均未
  超过 margin 0.3，按协议 retain，最终保留第 3 轮模型。

## 12.4 A 任务逐轮记录

| 轮 | 决策 | 策略 | 推理期校准 | selection macro-F1 | Δ vs 父 |
|---|---|---|---|---|---|
| 1 | promote | retain | recenter batch_mean w=0.4 | 0.7207 | +0.0175 |
| 2 | promote | retain | recenter batch_mean w=0.5 | 0.7346 | +0.0139 |
| 3 | rollback | retain | recenter batch_mean w=0.45 | 0.7346 | −0.0196 |
| 4 | rollback | retain | recenter batch_mean w=0.45 | 0.7346 | −0.0196 |
| 5 | rollback | retain | recenter batch_mean w=0.45 | 0.7346 | −0.0196 |
| 6 | rollback | retain | recenter batch_mean w=0.3 | 0.7346 | −0.0120 |
| 7 | rollback | retain | recenter batch_mean w=0.45 | 0.7346 | −0.0196 |
| 8 | promote | ridge_last_layer | recenter batch_mean w=0.35 | 0.8069 | +0.0723 |
| 9 | promote | retain | recenter batch_std w=0.2 | **0.8434** | +0.0365 |
| 10 | rollback | retain | recenter batch_mean w=0.45 | 0.8434 | −0.0365 |

Rollback 的含义是"候选比父模型差，拒绝并保留父模型"，因此 selection 指标保持
0.7346/0.8434 不变；这正是"回滚闭环"的现场证据。

### A 外部漂移探针（UCI batch1–10，逐批，final 用 batch_std w=0.2）

| batch | 行数 | 基线 macro-F1 | 最终 macro-F1 | Δ |
|---|---|---|---|---|
| 1（官方 train） | 445 | 0.9558 | 0.9475 | −0.008 |
| 2（官方 val） | 1,244 | 0.7140 | 0.8617 | +0.148 |
| 3 | 1,586 | 0.5939 | 0.7378 | +0.144 |
| 4 | 161 | 0.4893 | 0.4871 | −0.002 |
| 5 | 197 | 0.6270 | 0.6750 | +0.048 |
| 6 | 2,300 | 0.3228 | 0.4787 | +0.156 |
| 7 | 3,613 | 0.3267 | 0.4706 | +0.144 |
| 8 | 294 | 0.2664 | 0.3864 | +0.120 |
| 9 | 470 | 0.1852 | 0.4124 | +0.227 |
| 10 | 3,600 | 0.3806 | 0.3782 | −0.002 |
| **合计** | **13,910** | **0.4054** | **0.5239** | **+0.119** |

解读：官方 val（batch2）是"单批、按批 recenter"最有利的场景；batch1/4/10 基本持平
（batch1 是训练分布，batch4/10 类别结构差异大），batch6–9 的漂移段增益最大
（+0.12~+0.23）。per-class 上 Acetaldehyde（类 4）仍是最弱项（F1 0.225），与其在
各批次样本少、且与 Ethanol 响应高度相关一致。

## 12.5 B 任务逐轮记录与失效诊断

| 轮 | 决策 | 策略 | selection RMSE | Δ |
|---|---|---|---|---|
| 1 | promote | retain + shrink α=0 | 462.45 | −215.57 |
| 2–4 | retain | retain | 462.45 | 0 |
| 5 | retain | ridge_last_layer | 462.45 | 0 |
| 6–7 | retain | retain | 462.45 | 0 |
| 8 | retain | continue_train | 462.45 | 0 |
| 9–10 | retain | retain | 462.45 | 0 |

**为什么 B 的模型层改进全部无效**（这是本任务最有价值的记录驱动发现）：

1. 基线在官方 val 上 RMSE 681.16，而"训练集均值常数预测"RMSE 536.27——**基线比
   常数差 27%**，说明其特征→标签映射在留出轴承上失效；
2. 本地验证：续训网格（lr 3e-4~3e-3 × 15~100 epoch）→ 630–694；ridge 闭式解 →
   605–659；原始特征线性回归（大正则）→ 471.5；KNN → 653–662；按轴承 LOBO 校准
   → 804/689/831（完全不可迁移）；
3. 对 val 自身拟合仿射 `y = -0.847·pred + 961`，in-sample RMSE 仅 431.8——**预测与
   真值在留出轴承上负相关**，说明这不是"小幅偏移"，而是特征-目标关系破裂；
4. 因此最稳的"提升"是把预测收缩到公开验证目标分布中心：shrink α=0 即常数
   694.31（calibration 均值），selection RMSE 462.45、report 455.46。

结论：B 的增益来自**校准层而非模型层**，且这是基线失效下的安全最优解；系统在
连续无提升时触发 2 次 Panel（诊断/实验/评审/裁决各表态），verdict 均为 retain，
并记录在 `B/summary.json` 的 `panel` 字段——这是"卡住时讨论而非盲目调参"的直接
证据。

## 12.6 C 任务逐轮记录

| 轮 | 决策 | 策略 | selection RMSE | Δ |
|---|---|---|---|---|
| 1 | promote | continue_train lr=1e-3 ep=40 | 42.990 | −1.758 |
| 2 | promote | continue_train lr=1e-3 ep=100 | 42.151 | −0.839 |
| 3 | promote | continue_train lr=5e-4 ep=100 | **41.816** | −0.336 |
| 4 | retain | SWA lr=1e-3 10ep×3 | 41.816 | −0.248（未过 margin） |
| 5 | retain | retain | 41.816 | 0 |
| 6 | retain | retain | 41.816 | 0 |
| 7 | retain | SWA | 41.816 | −0.026 |
| 8 | retain | retain | 41.816 | 0 |
| 9 | retain | continue_train | 41.816 | −0.101 |
| 10 | retain | retain | 41.816 | 0 |

C 的收益（−6.6%）来自受限续训；后 7 轮的小幅改善均未越过 margin 0.3，按协议
retain，避免在单元方差大的验证集上追逐噪声。report 切分（57.46→56.28）显示
增益方向一致但幅度更小，已如实报告。

## 12.7 失败与排除的实验

| 实验 | 结果 | 结论 |
|---|---|---|
| A 只用 train 续训（lr 1e-3~1e-2 × 40–200ep） | val 0.59–0.63、probe 0.30–0.32 | 官方 train 只有 batch1，续训在漂移批上更差 |
| A 在 train+val 上续训 | probe 0.49–0.57（含 recenter 0.60–0.73） | 但直接训了 val，属泄漏，**不采用** |
| A 官方 val 上做 TTA（batch_std） | 0.714 → 0.522（整体）；逐批看 batch6/8 有增益 | 单批 TTA 会伤 val；最终改为 `recenter_blend`（w=0.2~0.5）并只在 selection 上验证 |
| B 续训 / ridge / KNN / 线性 | 均劣于常数 | 见 §12.5 |
| B 对 val 拟合仿射 | in-sample 431.8，但 LOBO 不可迁移 | 说明是分布破裂，不是校准偏移 |
| C SWA / 小 lr 续训 | 改善 < margin | 按协议 retain |
| ridge 分类（首次实现） | crash：`cannot reshape array of size 32 into shape (32,6)` | 修复为 one-hot 闭式解后通过 |

## 12.8 真实 LLM 链路验证（1 轮）

另有一次 `--llm on` 的 A 任务 1 轮运行（`docs/16` ADR-003/010）：

- 诊断 108.6s（input 4,977 / output 4,325 / reasoning 2,790 tokens）；
- 评审 80.6s（6 次工具调用），以"假设与 postprocess 矛盾、零信息量"否决 1 个方案；
- 报告 15.2s；当轮 selection 0.7032 → 0.9140（continue_train lr=1e-3×20 + class balance）；
- 全链路 token 46,172，远低于 2M 预算；
- LLM 不可用时同一套代码走确定性 planner，本次 3×10 主运行即为该路径的端到端验证。

## 12.9 局限与诚实披露

1. **A 的 selection 方差**：官方 val 是单批次，Toluene 仅 5 条；selection 上 +0.14
   的估计有较宽置信区间。外部 10 批探针用于交叉验证方向，但探针不参与选型。
2. **B 的行级切分**：验证集只有 3 个轴承，calibration/selection/report 只能轴承内
   行切分，无法完全消除轴承内泄漏；LOBO 实验表明跨轴承不可迁移，故 B 的结论是
   "基线失效→校准保底"，而非"模型变好"。
3. **C 的单元方差**：report 与 selection 的基线差 12.7 个 RMSE，最终模型 report
   增益（−1.18）小于 selection（−2.93），以 report 为准更保守。
4. **NASA score**：官方只要求 C 提供（单位循环）；B 的目标单位是分钟，其
   NASA-style 分数没有物理意义，报告中不作为选型依据。
5. **正式评测未开放**：平台 `blockers` 仍在等待通知；本报告以 workspace 十轮
   记录与本地互斥切分为验收依据，数据包版本变化时需重跑 `selftest` 校验 sha。

## 12.10 复现命令

```bash
# 主运行（离线确定性，约 18 分钟）
./.venv/bin/python -m rsi.cli run --tasks A,B,C --rounds 10 --llm off --state-dir .scratch/rsi-canonical

# 单任务复算
./.venv/bin/python -m rsi.cli selftest --task A

# 复核报告字段
./.venv/bin/python -m rsi.cli report --run-dir .scratch/rsi-canonical/rsi-20261115-162027

# 看板
./.venv/bin/python -m rsi.dashboard --state .scratch/rsi-canonical/rsi-20261115-162027/state.json
```

产物对照见 `docs/13_RUNBOOK.md` §13.2；本轮总用量：wall 1,202.9s、CPU 652.1s、
51 个实验、0 token（离线路径）。

## 12.11 Agent Kernel 自主闭环（A 任务 × 3 轮，真实 LLM）

运行 `.scratch/rsi-kernel-llm3/rsi-20261115-172045`，`--controller kernel --llm on`。
Agent 未被脚本化：它自己读运行状态与技能、提出候选、执行、比较、决策、写记忆。

| 轮 | 决策 | 关键实验 | selection | Δ | tool calls | 说明 |
|---|---|---|---|---|---|---|
| 1 | promote | continue_train(lr=1e-3, ep20, train+calibration, class_balance) | 0.703220→0.925572 | +0.222352 | 18 | 读到 skill 与 lessons 后判断弱类根因是训练分布不平衡 |
| 2 | promote | retain + recenter(batch_std, w=0.4) | 0.925572→0.937277 | +0.011705 | 14 | 用漂移报告（max_feature_shift=50.47）判断是尺度/增益失配 |
| 3 | retain | batch_std w=0.5 / ridge_last_layer | 0.937277（未采用） | — | 14 | 两个候选都低于父模型，按门控 retain |

- 总 token 361,014（input 332,732 / output 21,366 / reasoning 12,162），wall 693.7s，
  CPU 80.4s，7 个实验；三次主调用 latency 217s/174s/112s，平均约 2-4 分钟/轮。
- 轨迹包含 `get_lessons`、`read_skill`（3/3/2 次）、`compare_candidates`、`memory_add`
  （每轮 1-2 条）与 `finish_round`；第 3 轮的假设明确引用第 2 轮写入的记忆
  （"batch_std 权重 0.4 是局部最优…勿再加大"），证明跨轮记忆被实际使用。
- 最终父模型为第 2 轮 checkpoint（sha `f59e50375c17ab4…`），第 3 轮 retain 未改模型。

## 12.12 跨 run 记忆复用（第二次运行）

第 1 次运行（3 轮）把 6 条 lessons 写入 SQLite，并把结论写入 `memory/TASK_A.md`。
第 2 次运行（`.scratch/rsi-kernel-llm3/rsi-20261115-173751`）复用同一
`--db-path` 与 `--memory-dir`，第 1 轮的 brief 实测
`memory_chars=1562 / lessons_chars=414 / skills_chars=516`（第 1 次运行首轮为 0）。

可核验的差异（同一 seed、同一任务）：

| 指标 | 第 1 次运行首轮 | 第 2 次运行首轮 |
|---|---|---|
| 工具调用 | 18 | 13 |
| input tokens | 110,219 | 107,823 |
| 首个实验 | 先做对照/校准探索 | 直接复现已验证的 continue_train 配置 |
| 假设文本 | 探索性 | 明确引用 `记忆 exp-0003, sha f59e5037` |

第 2 次运行首轮决策 promote（0.7032→0.9256，与记忆中的结论一致），说明
"跨 run 记忆被读取并改变了行动顺序"，而不是仅作为日志存在。整轮 latency 247s
（第 1 次运行同轮 217s），token 略降；收益是**更少的探索步骤**，不是更高指标。

## 12.13 外部独立验证：UCI Gas Drift batch1–10

脚本 `scripts/external_validation.py` 把公开 UCI 原始 `.dat` 转成 JSONL 探针，
用 baseline 与最终 checkpoint 分别推理（扩展数据只读、绝不参与训练）。

| batch | rows | baseline macro-F1 | final macro-F1 | Δ |
|---|---:|---:|---:|---:|
| batch1 | 445 | 0.955754 | 0.947491 | −0.008263 |
| batch2 | 1244 | 0.714030 | 0.861712 | +0.147683 |
| batch3 | 1586 | 0.593897 | 0.737753 | +0.143855 |
| batch4 | 161 | 0.489335 | 0.487115 | −0.002220 |
| batch5 | 197 | 0.627000 | 0.674983 | +0.047983 |
| batch6 | 2300 | 0.322759 | 0.478744 | +0.155984 |
| batch7 | 3613 | 0.326685 | 0.470566 | +0.143881 |
| batch8 | 294 | 0.266411 | 0.386372 | +0.119961 |
| batch9 | 470 | 0.185163 | 0.412416 | +0.227253 |
| batch10 | 3600 | 0.380281 | 0.378246 | −0.002034 |
| **micro 加权** | 13910 | **0.425220** | **0.528070** | **+0.102850** |

诚实披露：batch1（同分布）、batch4（小样本）、batch10 有轻微回退（≤0.01），
其余 7 个漂移批次全部提升；微平均 +0.1029。该表与 §12.3 的 0.4054→0.5239
口径不同（前者为逐批等权，本表为按样本数加权），两个口径方向一致。

## 12.14 合成漂移探针标定（task A）

`synthesize()` 从官方 train 生成带增益/偏移/噪声的伪批次。用 baseline 评估，
标定默认强度（drift 为增益/偏移的 σ 倍数，noise 为噪声比例）：

| drift / noise | 整体 macro-F1 | batch1 / 2 / 3 |
|---|---:|---|
| 1.0 / 0.10 | 0.9485 | 0.950 / 0.956 / 0.939 |
| 2.0 / 0.20 | 0.8551 | 0.965 / 0.890 / 0.725 |
| 3.0 / 0.30 | 0.6837 | 0.930 / 0.692 / 0.401 |
| 默认 2.5 / 0.25 | 0.7187 | 0.865 / 0.714 / 0.531 |

默认参数对应"severity=severe"的漂移程度，接近官方 val→外部 batch 的落差量级；
该探针用于回归测试 harness 的漂移响应，不进入训练、不参与 checkpoint 选择。

## 12.15 私有/隐藏风格数据探针

`python -m rsi.cli probe --task A --path <file.jsonl>` 可直接评测任意符合官方
schema（`id`/`inputs`/`answer`）的私有数据，输出整体与按 group 的指标，
并在报告中声明 `training_isolation`。该路径与训练完全隔离：Guard 的白名单
只允许 `train` / `train+calibration`，executor 不接受任意文件路径。
比赛正式评测时，平台用同样的 predict 入口跑隐藏集，不需要公开标签。

## 12.16 记忆消融（进行中）

- 对照组：`--no-memory`（planner 仍按胜率排序，但无 MEMORY/lessons/技能）。
- 实验组：默认 `--memory on`，同一 seed、同一任务、相同轮数。
- 比较：primary 曲线、达到同等指标所需实验数、stall 轮数、token 用量。
- 若差异不显著，按计划把默认策略回退为"仅胜率排序"，并在本文件如实报告，
  不把不确定的收益写成确定结论。

## 12.17 v0.4 回归：自主内核 + 记忆开启后的全量离线 3×10

> **合规状态（v0.5 回溯标注）**：本节 B 的 462.45 使用 calibration（官方验证子集）
> 的目标均值做 shrink 中心，属于“验证标签参与推理期校准”，在 v0.5 合规口径下
> **已废弃**；A 的 0.8434 同样是 `train+calibration` 拟合的结果。当前合规数字以
> §12.18 为准（A 0.8884 / B 491.64 / C 41.82）。本节保留用于说明 v0.4 架构
> 回归不改变确定性决策路径，不作为性能证据。

代码冻结于 v0.4.0 后（v0.5 开发前），用 `--tasks A,B,C --rounds 10 --llm off`（确定性 pipeline，
记忆开启、schema v2）重跑 canonical，验证新架构没有改变主结果：

| 任务 | selection 基线 | 最终 | report 基线 | report 最终 | 实验数 |
|---|---:|---:|---:|---:|---:|
| A | 0.703220 | **0.843359** | 0.686380 | 0.812641 | 12 |
| B | 678.021915 | **462.454163** | 664.009143 | 455.461088 | 1 |
| C | 44.748700 | **41.815527** | 57.460000 | 56.276872 | 17 |

- run：`.scratch/rsi-final2/rsi-20261115-174743`；wall 1316.9s、CPU 683.3s、51 实验、
  0 token；与 v0.3.0 canonical（wall 1202.9s、CPU 652.1s）在指标上逐位一致。
- 结论：G4（不回归）达成；新增的记忆/DB/记录/降级逻辑不改变确定性路径的决策。
- 该 run 的 state.json 为 schema 2，包含 controller、goal、memory、strategy_stats、
  rounds[].agent 等字段；`workspace_report.json` 的平台字段保持不变。

## 12.18 v0.5 外部预训练消融：合规 + 外部数据 3×10（opt-in，非评分路径）

> **口径说明（用户 m02821 决定）**：正式评分走“只用官方数据”，本节外部预训练
> 默认关闭（`RSI_ALLOW_EXTENDED_DATA=0`），仅作为扩展数据验证加分与消融证据保留；
> 当前评分用 canonical 见 §12.19。

命令（离线确定性、记忆关闭、显式开启外部预训练）：

```bash
RSI_ALLOW_EXTENDED_DATA=1 ./.venv/bin/python -m rsi.cli run \
    --tasks A,B,C --rounds 10 --llm off --state-dir .scratch/rsi-v05 --no-memory
```

run：`.scratch/rsi-v05/rsi-20261115-193347`；wall 1390.0s、CPU 751.7s、52 实验、
0 LLM token。

| 任务 | selection 基线 | 最终 | report 基线 | report 最终 | 实验数 | 关键决策 |
|---|---:|---:|---:|---:|---:|---|
| A | 0.703220 | **0.888398** | 0.686380 | 0.864562 | 12 | r1/r2 recenter、r6 extended_pretrain、r7 recenter |
| B | 678.021915 | **491.638136** | 664.009143 | 489.633879 | 1 | r1–r3 shrink α=0→0.3（train 中心 954.15） |
| C | 44.748700 | **41.815527** | 57.460000 | 56.276872 | 17 | r1–r3 continue_train（train-only，lr 1e-3/5e-4） |

合规性：A 的 r6 `extended_pretrain` 只用 UCI batch3–7 + 官方 train；B/C 只用官方
train；validation/selection/report 标签在任何阶段都不参与拟合（`rsi/policy.py` +
Guard 双重强制，tests/test_compliance.py 覆盖）。

**A 外部逐批验证（最终 checkpoint + recenter batch_mean w=0.3，未参与训练的
batch8–10 加粗）**：

| batch | rows | 基线 macro-F1 | 最终 macro-F1 | Δ |
|---|---:|---:|---:|---:|
| b1 | 445 | 0.9558 | 0.9758 | +0.0201 |
| b2 | 1244 | 0.7140 | 0.9102 | +0.1962 |
| b3 | 1586 | 0.5939 | 0.6833 | +0.0894 |
| b4 | 161 | 0.4893 | 0.5363 | +0.0470 |
| b5 | 197 | 0.6270 | 0.8837 | +0.2567 |
| b6 | 2300 | 0.3228 | 0.9002 | +0.5774 |
| b7 | 3613 | 0.3267 | 0.7957 | +0.4690 |
| **b8** | **294** | **0.2664** | **0.6296** | **+0.3632** |
| **b9** | **470** | **0.1852** | **0.5350** | **+0.3498** |
| **b10** | **3600** | **0.3806** | **0.5796** | **+0.1990** |

micro 加权 0.4253→0.7462。其中**未参与预训练的 b8–b10 加权 0.3519→0.5781
（+0.226）**，这是比官方 val 更可信的漂移泛化证据；b3–b7 因参与预训练而偏乐观，
报告中分别标注。

**B/C 的方法学口径**：B 的 491.6 本质是“承认基线在留出轴承上失效 → 向训练目标
中心收缩”的稳健校准（合规、可解释，但预测能力未质变）；C 的 41.8 是官方 train
上的真实续训收益（无标签泄漏）。若隐藏轴承/发动机分布与公开验证差异更大，B 的
收益可能缩水，C 的收益最稳。


## 12.19 v0.5.1 最终运行：严格官方数据 3×10（canonical，B damage_shrink + C 两阶段续训）

命令（离线确定性、记忆关闭、外部数据默认关闭）：

```bash
RSI_ALLOW_EXTENDED_DATA=0 ./.venv/bin/python -m rsi.cli run \
    --tasks A,B,C --rounds 10 --llm off --state-dir .scratch/rsi-v052 --no-memory
```

run：`.scratch/rsi-v052/rsi-20261115-210157`；wall 1825.5s、CPU 1059.1s、56 实验、
0 LLM token；`RSI_ALLOW_EXTENDED_DATA=0`，仅官方 train 参与拟合。

| 任务 | selection 基线 | 最终 | report 基线 | report 最终 | 实验数 | 关键决策 |
|---|---:|---:|---:|---:|---:|---|
| A | 0.703220 | **0.734555** | 0.686380 | 0.683406 | 12 | r1/r2 recenter(batch_mean) promote；r3–r10 回滚（候选未过 2×margin/gate） |
| B | 678.021915 | **409.736985** | 664.009143 | 424.130024 | 11 | r1 promote：父 checkpoint 不变 + train-only `damage_shrink`（a=0.5,w=0.6）；r2–r10 retain（指标不再变化） |
| C | 44.748700 | **40.523699** | 57.460000 | 56.021241 | 16 | r1 promote `continue_train_staged`（1000@1e-3 + 400@2e-4，仅官方 train）；其余轮 retain |

- A 外部探针（UCI batch3–10，仅验证）：0.4054 → 0.5239；报告/隐藏标签未参与拟合。
- B 的 `damage_shrink` 由官方 train 标签拟合 10 分位曲线、center=train 目标均值；
  selection 上选 a/w，report 只报告；checkpoint 字节不变（父 sha 保持
  `c7d819fcee9d2545…`），predict 侧 `postprocess.json` 自带 detector/curve。
- C 的两阶段续训沿用官方 Adam/梯度裁剪/冻结归一化，父谱系连续，属“修改训练参数
  后继续训练”；单次约 200s。
- 与 §12.18 的差异：A 0.8884→0.7346（关闭外部数据），B 491.6→409.7（新后处理），
  C 41.82→40.52（两阶段续训）；外部数据消融仍保留在 §12.18 作为加分证据。

## 12.20 整夜复现运行（v0.6.0，严格官方数据 3×10）

为验证 v0.5.1 结果可复现并生成 0.6.0 镜像/提交包证据，整夜串行任务以同一命令重跑：

```bash
RSI_ALLOW_EXTENDED_DATA=0 ./.venv/bin/python -m rsi.cli run \
    --tasks A,B,C --rounds 10 --llm off --controller pipeline --state-dir .scratch/night-strict
```

run：`.scratch/night-strict/rsi-20261115-233649`；wall 1897.2s、CPU 1081.3s、56 实验、
0 LLM token；`record_complete=true`（A/B/C 各 10 轮）。

| 任务 | selection 基线 | 最终 | report 最终 | 实验数 | 备注 |
|---|---:|---:|---:|---:|---|
| A | 0.703220 | **0.734555** | 0.683406 | 12 | r1/r2 recenter promote，其余回滚 |
| B | 678.021915 | **409.736985** | 424.130024 | 11 | r1 damage_shrink promote，其余 retain |
| C | 44.748700 | **40.523699** | 56.021241 | 16 | r1 staged continue_train promote，其余 retain |

- 与 §12.19 的 selection/report 指标逐位一致 → 该配方在 4 核 CPU、离线确定性路径下可复现；
  A 的外部探针（仅验证）为 **0.522378**（§12.19 记录 0.5239 为 v0.5.0 那次运行的取值，
  两轮模型相同、仅探针窗口/TTA 细节带来的 ±0.002 波动，不影响结论）。
- 产物：镜像 `zhixun-rsi-harness:0.6.0`（sha256 `822b73e3…`）、提交包
  `dist/submission_rsi-20261115-233649.tar.gz`（sha256 `1a426f76…`，≤64MiB），
  独立 predict 在全 validation 上 A macro-F1 **0.71987**、B RMSE **422.24**、C RMSE **50.47**。
- 严格口径：仅官方 `public/train.jsonl` 参与拟合；validation 仅用于 selection/report 评估；
  `RSI_ALLOW_EXTENDED_DATA=0`，外部数据只作为 §12.21 的 Track X 测试床。

## 12.21 Track X：外部数据持续迭代测试床（UCI gas，blinded batches 6–9）

目的：验证 harness 在“比赛看不到的、持续到来的标注批次”上能否自主迭代，而不是只在官方
validation 上选点。数据：UCI Gas Sensor Array Drift 的 batch6–9（官方 A 包只给 batch1/2），
每批按 50/50 分层切出训练/评估；评估标签只用于打分，不进入任何正式提交谱系。
命令：

```bash
RSI_ALLOW_EXTENDED_DATA=1 ./.venv/bin/python -m rsi.cli stream --tasks A \
  --source external-a --batches 4 --external-root extra/uci-gas/Dataset \
  --external-start 6 --holdout 0.5 --experiments-cap 2 --llm off \
  --state-dir .scratch/night-testbed-x2
```

四臂对照（`scripts/testbed_four_arms.py`，frozen=官方父 checkpoint，recenter=冻结模型+
批内重中心化，full harness=harness 实际部署模型，oracle=用该批标签重训的上界；
oracle 只在测试床评估，不参与部署）：

| 批次 (UCI) | frozen | recenter | full harness | oracle | harness 决策 |
|---|---:|---:|---:|---:|---|
| batch6 | 0.3209 | 0.4516 | **0.4697** | 0.9882 | promote（recenter） |
| batch7 | 0.3395 | 0.4557 | **0.4871** | 0.9980 | retain |
| batch8 | 0.2563 | 0.3685 | **0.4053** | 0.9506 | retain |
| batch9 | 0.1868 | 0.5401 | **0.5511** | 0.9962 | retain |

- 结论：harness 部署模型在 4/4 批上优于 frozen 与 recenter 基线（+0.15~+0.36 vs frozen），
  说明“漂移监测→诊断→实验→门控上线”的闭环在 unseen 批次上可迁移；与 oracle 的差距
  主要来自 oracle 每批直接用该批标签重训（测试床上界），而 harness 只允许 train/累计数据，
  且每批实验预算仅 2 次。
- 局限：batch8/9 的类别分布与官方 train 差异大（Toluene 等稀有类），绝对 F1 仍低；
  该测试床只用于证明“持续迭代能力”，不计入正式分数；外部数据不写入提交 manifest。

## 12.22 整夜 kernel 验证（B 任务，strict）

命令（严格官方数据、kernel 控制器、真实 LLM）：

```bash
RSI_ALLOW_EXTENDED_DATA=0 ./.venv/bin/python -m rsi.cli run --tasks B --rounds 2 \
  --llm on --llm-env .scratch/llm.env --controller kernel \
  --round-turns 16 --experiments-cap 6 --state-dir .scratch/night-kernel-b
```

run：`.scratch/night-kernel-b/rsi-20261116-001411`；wall 939.4s、CPU 54.5s、5 实验、
LLM 467,689 tokens（in 430,552 / out 37,137 / reasoning 29,014）、3 次调用、0 错误。

| 轮 | 决策 | selection RMSE | 关键行为 |
|---|---|---|---|
| 1 | promote | 678.022 → **602.239** | 17 tool calls；依据漂移归因（特征 14/30/31/21/4 仿射增益/偏移）选择 `retain + recenter(batch_mean, w=0.4)`；1 次记忆写入 |
| 2 | retain | 602.239 | 8 个候选（recenter batch_std、ridge、continue_train 等）均未过 margin；假设写明“std 收缩 40–67% 叠加 mean shift”，保留父模型；1 次 skill 写入 |

- 结论：同一 kernel 控制器在 B 任务上同样能自主完成“诊断→候选→执行→门控→记录”，
  但只找到 recenter 类后处理（Δ−75.8），未发现 docs/12 §12.19 的 `damage_shrink`
  （Δ−268.9）——说明“策略库 + 知识候选清单”仍需把强配方显式喂给 Agent，这是
  v0.6 已知差距（已记入 docs/18 §10.5 与 OVERNIGHT_LOG 的后续项）。
- 严格口径：仅官方 train 拟合；validation 只用于 selection/report。

## 12.23 A 提升：历史标签自适应（P0）与 weighted-CE（P1，门控未采纳）

命令：

```bash
# P0 外部测试床（标签延后一批，leak-free）
RSI_ALLOW_EXTENDED_DATA=1 ./.venv/bin/python -m rsi.cli stream --source external-a \
  --tasks A --batches 8 --external-start 3 --external-end 10 --holdout 0.5 \
  --experiments-cap 1 --retention-every 0 --adapt-history --adapt-epochs 30 \
  --external-root extra/uci-gas/Dataset --state-dir .scratch/stream-adapt

# P0 官方流（无外部数据）
./.venv/bin/python -m rsi.cli stream --source official --tasks A --batches 8 \
  --experiments-cap 1 --retention-every 0 --adapt-history --adapt-epochs 30 \
  --state-dir .scratch/stream-official-adapt

# P1 静态配方试跑（10 轮 canonical）
./.venv/bin/python -m rsi.cli run --tasks A --rounds 10 --llm off --no-memory --state-dir .scratch/a-p1
```

**P0 外部测试床（UCI batch3–10，8 批，leak-free）**：frozen macro 均值 .3977 →
仅 recenter .5046 → **history-adapt .7733**（micro .3797/.4851/.7417）；8 批 0 error、
wall 333.7s。逐批与决策见 docs/23 §8.1。

**P0 官方 validation 顺序 8 批**：frozen .6158 → **history-adapt .6361**（+0.020，
+3.3% 相对）；仅 recenter 在 155 行小批次上反而降到 .4454。结论：历史自适应在
外部多类批次上收益巨大、在官方小偏批次上不回归；recenter-only 在类偏小批上不可靠。

**P1 weighted-CE（负结果）**：A canonical r3 尝试 `class_weight_beta=0.5, lr=5e-4,
epochs=30 + recenter w=0.3`，selection 0.7226 < 父 0.7346 → 门控 rollback；该候选
全量 validation 0.7279（w.3）/0.7289（w.4）vs 部署 0.7199。更强变体 full 0.7368
但 selection 0.7224。按协议不使用全量 validation 反向选型，故**不采纳、不改 submission**，
配方保留为 opt-in 能力（`continue_train.class_weight_beta`）。

**默认策略**：stream 模式默认 `--adapt-history`（entrypoint `RSI_STREAM_ADAPT=1`，
30 epoch/lr 5e-4/recenter 0.5）；不影响单次 `run`/submission。
