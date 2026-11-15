---
title: 校准迁移：recenter 覆盖校准与 shrink 输出校准
slug: calibration-transfer
tags: calibration, recenter, shrink, postprocess, temperature, 校准, 迁移, 推理期
applies: 协变量漂移、置信度失配或回归输出整体偏移，且没有足够新标签重训
not_applies: 概念漂移（P(Y|X) 变）；目标批次太小（方差估计不稳）；标签先验剧烈变化
data_needed: 目标批次特征（无标签即可做 recenter）；shrink 中心来自训练目标均值；分类温度校准需要校准集标签
sources: Saerens et al. (2002) Adjusting the outputs of a classifier to new a priori probabilities, Neural Computation；Guo et al. (2017) On calibration of modern neural networks, ICML；Alexandari et al. (2020) Maximum likelihood with bias-corrected calibration under label shift, ICML；Sugiyama et al. (2012) Density ratio estimation in machine learning, Cambridge
---

# 一句话结论

**recenter 修"输入覆盖"（无标签、零 checkpoint 字节）**：把推理批次重新标准化回训练统计。**shrink 修"输出尺度"（回归）**：把预测向训练目标均值收缩。两者都只改推理期，必须用 selection 门控验证；先验变化大时 recenter 可能害大于利。

## 适用条件

- 已确认特征整体/部分偏移（`feature_shift` 或逐特征 KS 明显），且标签分布没有剧烈变化。
- 模型 checkpoint 不得改变结构/归一化协议（本仓库硬约束：候选只能是续训）。
- 有一批目标特征（推理时天然可得）用于估计批统计量。

## 方法步骤

1. **batch_mean recenter**：`x' = x - (μ_batch - μ_train)`，只校均值；适合整体偏移、形状不变。注意：若 batch 内类别构成不同，批均值本身有偏，可用更稳健的窗口/分位数。
2. **batch_std recenter**：`x' = (x - μ_batch) / σ_batch * σ_train + μ_train`，同时校均值与尺度；适合增益漂移。小批次（< 64 行）时尺度估计噪声大，需要收缩或降权。
3. **混合权重**：`pred_final = (1-w)·pred_raw + w·pred_calibrated`，w∈[0,1]；本仓库 A 任务实测有效区间 0.2–0.5。
4. **shrink（回归）**：`y' = α·pred + (1-α)·center`，center 取训练目标均值（如 B 的 694.31 分钟）；α 越小越保守。α=0 等价于常数预测，只应作为对照，不是长期策略。
5. **分类输出校准**（可选）：温度缩放/Platt/保序回归，需要校准集标签；标签先验变化时用 BBSE/EM 类先验修正（Alexandari 2020），但不改变特征。
6. **选择与记录**：每次候选都记录校准类型、模式、权重、父模型 sha、selection 指标；report 只做一次复算。

## 与 harness 的映射（可执行）

| postprocess 规格 | 适用证据 | 参数范围 |
|---|---|---|
| `{"type":"recenter","mode":"batch_mean","weight":0.4}` | 整体均值位移、形状不变 | weight 0–1（executor 校验） |
| `{"type":"recenter","mode":"batch_std","weight":0.2}` | 均值+尺度同时变（增益漂移） | 同上；小批次需降权 |
| `{"type":"shrink","alpha":0.5,"center":694.31}` | 回归输出整体偏移、标签先验变 | α 0–1；center 用训练目标均值 |
| `{"type":"damage_shrink",...}` | RUL 特定退化曲线 | 见 `rul-calibration` |

## 如何验证

- **门控**：只有 selection 主指标越过 margin 才 promote；小提升（< 1σ 噪声）不采信。
- **批次内稳定性**：把目标批次随机切两半分别算校准权重，结果敏感说明批次太小。
- **无标签自检**：recenter 后检查特征 KS 是否下降、置信度熵是否回到训练区间；这些是必要条件不是充分条件。
- **外部探针**（只验证）：A 用 UCI batch3–10 复算宏 F1，确认校准在更严重漂移段的方向一致性。

## 数据需求

目标批次特征（推理期可得）；训练特征均值/方差必须来自 checkpoint 冻结统计（不可用验证集重算）。shrink 中心来自训练目标均值。分类温度校准需要独立的校准集标签（本仓库用 calibration 切分，注意合规边界）。

## 实验证据（公开 + 本仓库）

- 本仓库：A 任务 recenter(batch_mean, w=0.4/0.5) 连续两轮 promote，selection macro-F1 0.7032→0.7346；后续 batch_std w=0.2 再升到 0.8434（该严格 run 最终 0.7346）。
- 本仓库：A 外部 UCI 漂移探针（只验证）宏 F1 0.4054→0.5239（+0.119），batch6–9 提升最大（+0.12~+0.23）。
- 本仓库：B 任务 shrink（center=694.31，α=0）在官方口径下 RMSE 678.02→409.74，但 α=0 使预测退化为常数，后续轮次指标完全不变被门控识别为无提升——这是"校准收益真实但不可持续"的现场证据。
- 公开：分类输出需事后校准（Guo 2017）；标签先验变化下的校准有系统偏差（Saerens 2002；Alexandari 2020）。

## 风险与不适用条件

- **标签先验漂移时 recenter 有风险**：批均值混入类别构成变化，会把"类别比例变化"误当传感器漂移。
- 小批次方差估计不稳定，batch_std 可能放大噪声。
- shrink 把预测拉向均值会压缩动态范围，对高 RUL 段伤害大。
- 反复用 selection 调权重会隐性过拟合；权重搜索要有预算与记录。
- 概念漂移没有无标签校准解。

## 来源

Saerens et al., Neural Computation 2002；Guo et al., ICML 2017；Alexandari et al., ICML 2020；Sugiyama et al., Cambridge 2012；本仓库 docs/12_EXPERIMENTS.md。
