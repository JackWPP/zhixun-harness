---
title: 传感器增益/基线漂移：识别、校正与增广
slug: sensor-gain-baseline-drift
tags: sensor, gain, offset, baseline, drift, 传感器, 增益, 基线, 校正
applies: 特征整体均值/尺度偏移、标签分布相对稳定，怀疑传感器老化、温漂或批次校准差异
not_applies: 输入变化是真实工况变化或标签先验变化；缺少参考批次无法估计校正参数
data_needed: 参考批次与目标批次的逐特征值；若做物理校正，需要重复测量/标定样本
sources: Vergara et al. (2012) Chemical gas sensor drift compensation using classifier ensembles, Sensors & Actuators B（UCI gas drift）；Fonollosa et al. (2015) Calibration transfer and drift counteraction in chemical sensor arrays, Sensors & Actuators B；ISO 11843 系列（检测能力与校准，作为方法学参考）
---

# 一句话结论

增益漂移 = **乘性**（尺度变），基线漂移 = **加性**（零点变）。两者最常见的组合是仿射变换 `x' ≈ a·x + b`。先用逐特征均值/方差与 KS 判断，再用参考批拟合并只做推理期校正；训练时可加入同族增广提升鲁棒性。

## 适用条件

- 目标批次与参考批次的特征边缘分布有稳定差异，且差异可由少数参数（gain/offset）解释。
- 标签先验没有明显变化（否则均值位移混入类别构成变化）。
- 有参考批（训练集）与目标批的逐特征值。

## 方法步骤

1. **诊断**：逐特征比较 `μ_t vs μ_r`、`σ_t vs σ_r`；仿射漂移的特征通常满足 `(μ_t-μ_r)/σ_r ≈ const` 或成比例于 `a-1`，而形状（偏度）变化小。
2. **稳健拟合**：在特征对上用 Theil-Sen/Huber 回归估计 `x_t ≈ a·x_r + b`（逐特征分位数匹配更稳）；避免用均值单点估计。
3. **校正**：推理期 `x' = (x_t - b)/a`（同时用于 batch_mean/scale 的 recenter 是它的对角近似）；注意只对确认漂移的特征做，避免伤害稳定特征。
4. **训练增广**：对训练批随机施加 gain∈[0.9,1.1]、offset∈[−0.5σ,0.5σ] 的批次级变换（同一批共享参数，模拟传感器批次误差），提升模型对漂移的鲁棒性；不应改变归一化协议之外的结构。
5. **监控**：每批次复算均值/方差/KS；把漂移参数纳入记录，便于跨批次追溯与回滚。
6. **区分真实信号**：若领域物理上输入应随工况变化，则不能盲目校正；先看标签是否同步变化。

## 与 harness 的映射（可执行）

| 现象 | 报告证据 | 动作 |
|---|---|---|
| 均值整体同向偏移 | `feature_shift>0.45`、`causes` 含 `sensor_or_environment_gain_drift` | `recenter(batch_mean)`，w 0.3–0.5 |
| 尺度同时变 | `ks_max` 高、逐特征方差比偏离 1 | `recenter(batch_std)`，w ≤ 0.4 |
| 多批次反复漂移 | 外部探针逐批指标波动 | 训练 `augment=True` + 推理校准组合 |
| 标签分布也变 | `label_shift.js>0.1` | 先处理先验（class_balance/输出校准），再做传感器校正 |

## 如何验证

- **注入实验**：对稳定批次施加已知 gain=1.1/offset=0.5σ，确认检测器与校正能把 KS/指标拉回；这是最便宜的因果验证。
- **逐批外部探针**：A 用 UCI batch1–10（只验证不训练）观察校正后的宏 F1 与逐批漂移参数关系。
- **负对照**：对未漂移特征做同样校正，确认无收益（避免"校正一切"的过拟合）。
- **一致性**：同一批次多次打分，校正参数应稳定；否则批次太小。

## 数据需求

参考批逐特征统计量（训练集冻结统计）、目标批特征。做物理标定需要重复测量或标准气体/标准件；本仓库只能做数据驱动的仿射校正，不能替代硬件标定。

## 实验证据（公开 + 本仓库）

- 公开：UCI Gas Sensor Array Drift 数据集记录 10 个批次、36 个月采集的 e-nose 漂移，是传感器漂移研究的常用基准；分类器集成与校准迁移是其标准对策（Vergara 2012；Fonollosa 2015）。
- 本仓库：A 任务的官方 train/validation 对应 UCI batch1/batch2，batch3–10 为更严重漂移段；逐批 recenter（对角增益/基线校正）使外部探针宏 F1 0.4054→0.5239，batch6–9 提升最大。
- 本仓库：`causes` 逻辑中"feature_shift 大 + 标签 JS 小"会同时标记协变量与传感器/环境漂移，实际处置仍以 recenter + 增广为主。

## 风险与不适用条件

- 校正参数用目标批标签拟合是泄漏；只能用无标签特征统计。
- 真实工况变化与传感器漂移在统计上不可分，必须结合领域知识/标签判断。
- 单批校正可能过拟合该批，下一批再次漂移；需要在线滚动窗口与回滚机制。
- 极端饱和/截断不是仿射变换，校正无效，需要修硬件或换特征。
- 增广会改变训练分布，可能伤害稳定批性能，必须过门控。

## 来源

Vergara et al., Sensors & Actuators B 2012；Fonollosa et al., Sensors & Actuators B 2015；UCI Machine Learning Repository #224 Gas Sensor Array Drift（CC BY 4.0）；ISO 11843（校准与检测能力，方法学参考）；本仓库 docs/15_DATA.md。
