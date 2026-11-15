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
