---
title: 实验设计与证伪：单变量、噪声阈值与负对照
slug: experiment-design-falsification
tags: experiment, design, falsification, ablation, seed, 实验设计, 证伪, 消融, 噪声
applies: 每轮提出候选实验、判断提升是否真实、决定 promote/retain/rollback 之前
not_applies: 评估切分不可复现；预算不足以复跑种子；指标实现中途变化
data_needed: 固定切分与种子、父模型指标、门控 margin、同类实验的历史胜率
sources: Popper (1959) The Logic of Scientific Discovery；Kohavi, Tang & Xu (2020) Trustworthy Online Controlled Experiments, Cambridge；Henderson et al. (2018) Deep reinforcement learning that matters, AAAI（种子方差）
---

# 一句话结论

一个合格实验 = **单一变量 + 可证伪预测 + 预先声明的成功标准 + 负对照**。提升必须先超过"种子噪声"，再看是否越过门控 margin；失败实验也要记录，否则无法区分"没效果"和"没试对"。

## 适用条件

- 有可复现的切分/种子/指标实现。
- 每轮实验数量与预算受限（本仓库每轮 ≤3、每任务 ≤40）。
- 需要向评审解释"为什么接受/拒绝了某个候选"。

## 方法步骤

1. **写假设**：一句话预测方向与幅度，例如"gain 漂移导致覆盖失配；batch_mean recenter w=0.4 将把 selection macro-F1 提升 ≥0.004"。假设必须能被数据否定。
2. **单变量**：一次只改一个策略或一个后处理参数；多变量同时改无法归因。
3. **预注册成功标准**：主指标、margin、允许的 report 变化范围；不能用同一数据既选又报。
4. **噪声基线**：同配置多 seed 复跑估计 σ；本仓库参考 A≈0.0029、B≈4.72、C≈0.35，门控至少 2σ（或任务 margin）。
5. **负对照**：retain/常数预测/随机参数作为对照臂；任何"提升"必须优于对照。
6. **预算与停止**：连续 N 轮无提升触发 explore/panel；不要用无限次选择迭代把 selection 拟合掉。
7. **记录一切**：假设、参数、父/候选 sha、指标、决策、失败原因；失败实验是负知识。

## 与 harness 的映射（可执行）

| 字段/工具 | 要求 |
|---|---|
| `propose_experiment.hypothesis` | 必须含方向+幅度+解释，不能写"试试看" |
| `expected_effect` / `risk` | 预先声明预期与风险，评审据此 veto |
| `gate(margin)` | 提升 ≥margin 才 promote；劣化 ≥2×margin rollback |
| `selection` / `report` | 选择只看 selection；report 只报告 |
| `strategy_stats` | 用历史胜率排序，避免重复失败配方 |
| `--seed` 多跑 | 噪声阈值来源；A 换种子曾出现 0/10 复现 |

## 如何验证

- **复跑**：同配置 3 seed，均值±sd；提升 < 1σ 不采信。
- **消融矩阵**：策略×任务×种子的小矩阵，报告主指标与相对基线/常数预测的差。
- **决策回放**：用 rounds.jsonl 重放门控，确认每个 promote 都有 selection 证据。
- **反事实**：把候选换成 retain 后指标不变，说明收益来自随机波动而非改动。

## 数据需求

固定切分 manifest 与 sha、随机种子、父模型指标、门控 margin、历史实验记录。种子噪声需要重复运行预算；若预算不足，至少在 report 中声明单次结果的不确定性。

## 实验证据（公开 + 本仓库）

- 本仓库：A 换种子 0/10 复现，说明单 seed 的 +1.8% 不可信；B 的变差实验与常数预测对比显示"相对官方基线"会夸大提升。
- 本仓库历史：C 第二轮 +0.0106 曾因 margin=0 被接受，按 2σ 噪声阈值（0.7）应拒绝——这是门控阈值的数据依据（见 docs/08_roadmap.md）。
- 本仓库：B 首轮 damage_shrink 一次提升 31.8%，后续 9 轮 retain，说明单次大幅提升可能来自校准形式而非可持续改进，需要多 seed 复核。
- 公开：在线对照实验的显著性/多重比较与停止规则（Kohavi 2020）；深度 RL 种子方差常被低估（Henderson 2018）。

## 风险与不适用条件

- 反复用 selection 尝试不同参数会把它变成训练集（选择过拟合）。
- 门控 margin 过小会把噪声当提升；过大则错过真实小收益，需要数据驱动标定。
- 中途改代码/数据/指标会让实验不可比；必须记录 code sha 与数据 sha。
- 失败实验不记录会导致"重复踩坑"，也会让评审无法判断探索程度。

## 来源

Popper, The Logic of Scientific Discovery, 1959；Kohavi, Tang & Xu, Cambridge 2020；Henderson et al., AAAI 2018；本仓库 docs/08_roadmap.md、docs/12_EXPERIMENTS.md、docs/16_DECISIONS.md。
