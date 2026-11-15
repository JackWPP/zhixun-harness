---
title: 特征归因：标准化均值位移、KS、方差比与相关性变化
slug: feature-attribution
tags: attribution, feature, importance, shap, permutation, 归因, 特征, 重要性
applies: 已确认发生漂移，需要回答"是哪些特征、以什么方式在变"，以缩小实验搜索空间
not_applies: 特征高度共线且没有分组信息；或只有聚合指标、拿不到逐特征原始值
data_needed: 参考/目标窗口的逐特征值；有模型时可做置换重要性，无标签时只做无监督归因
sources: Breiman (2001) Random Forests, Machine Learning 45(1)（置换重要性）；Lundberg & Lee (2017) A unified approach to interpreting model predictions, NeurIPS；Strobl et al. (2007) Bias in random forest variable importance measures, BMC Bioinformatics
---

# 一句话结论

优先用**不需要标签、可解释、便宜**的逐特征归因：标准化均值位移（方向）、KS（形状）、方差比（增益/噪声）、与参考相关性变化（结构）。把 top-k 特征写进实验假设，而不是对所有特征做同一套校准。

## 适用条件

- 有逐特征原始值（本仓库 A=128、B=34、C=24 维）。
- 需要把"整体漂移"定位到少量特征，从而选择 recenter 模式、增广方式或重训重点。
- 模型可访问时（本仓库 `ExtendedModel.hidden/predict` 可读），可加模型侧解释。

## 方法步骤

1. **标准化均值位移**：`Δ_j = |μ_t - μ_r| / σ_r`，按降序取 top-k；记方向（增/减）。
2. **逐特征 KS**：捕捉"均值没变但分布形状变了"的情况（双峰、饱和截断）。
3. **方差比与高阶矩**：`std_t/std_r` 偏离 1 提示增益不稳或噪声变化；必要时看偏度/峰度。
4. **相关性结构变化**：参考与目标相关矩阵之差 `corr_shift`，按特征聚合；结构变化大而边缘分布小 → 多变量/概念类问题。
5. **模型侧归因**（可选）：置换重要性（打乱单个特征后指标下降量）或梯度/线性系数；SHAP 类方法成本高、需背景集，离线预算下谨慎。
6. **合成归因表**：每特征一行 `{mean_shift_sd, ks, std_ratio, js, corr_shift, score, direction}`，写入报告与知识引用，供 Agent 直接引用 `[K#]`。
7. **从归因到动作**：top-k 集中在少数特征且方向一致 → 增益/基线校正；全特征均匀 → 批次级 recenter；相关结构变 → 重训/增广。

## 与 harness 的映射（可执行）

| 报告字段 | 含义 | 下一步 |
|---|---|---|
| `evidence.top_shifted_features` | 标准化均值位移最大的 8 个特征下标 | 与 `attribution` 交叉确认 |
| `attribution[].feature/score/direction` | top-k 归因（增强后字段） | 生成"特征 i 均值上移 1.2σ"的可证伪句 |
| `attribution[].ks`、`std_ratio` | 形状/方差证据 | 选择 `recenter(batch_std)` 或 `augment=True` |
| `correlation_shift` | 相关结构变化强度 | > 0.2 时优先考虑续训而非只校准 |

## 如何验证

- **消融**：只对 top-k 特征做增益/偏移注入，确认探针指标变化方向与归因一致。
- **稳定性**：不同窗口重复计算 rank，top-k 覆盖率 ≥ 60% 才算稳定归因。
- **动作验证**：在 selection 上验证校准是否过门控；report 只做一次事后复算。
- **反例检查**：把归因 top-1 换成随机特征做同样动作，确认收益消失（排除"任何校准都涨"的假象）。

## 数据需求

逐特征值、特征名映射（若有）、分窗方案。模型侧归因需要可复现的预测函数与固定背景集；无标签也能完成第 1–4 步。

## 实验证据（公开 + 本仓库）

- 公开：置换重要性简单但会被相关特征稀释/放大（Strobl 2007；Breiman 2001）；SHAP 给出一致性更好的局部解释但成本高（Lundberg & Lee 2017）。
- 本仓库：A 的 128 维 e-nose 特征在 UCI 各批次间整体偏移明显，逐批 recenter（本质是特征级覆盖校准）使外部探针 0.4054→0.5239；per-class 上 Acetaldehyde 仍最弱，说明归因要结合类别构成而非只看总体。
- 本仓库：`rsi/drift.py` 的增强版 `attribution` 直接给出 top-k 特征与证据，Agent 可引用后提出针对性实验。

## 风险与不适用条件

- 传感器阵列高度相关，单特征归因可能是冗余的；必须看分组（同一传感器多通道）。
- 均值位移大但标签不变 ≠ 模型一定受损；需要配合性能证据。
- 用验证集标签算归因会构成泄漏；无监督归因优先。
- 归因不稳定时不要据此做精细校准；退回批次级方法更安全。

## 来源

Breiman, Machine Learning 2001；Strobl et al., BMC Bioinformatics 2007；Lundberg & Lee, NeurIPS 2017；Rabanser et al., NeurIPS 2019；本仓库 docs/12_EXPERIMENTS.md。
