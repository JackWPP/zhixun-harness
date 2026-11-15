---
title: 残差与性能监控：有标签、标签延迟与无标签三种情形
slug: residual-performance-monitoring
tags: performance, residual, monitoring, 残差, 性能, 监控, 标签延迟, 指标
applies: 需要判断"模型是不是真的变差了"、以及在没有即时标签时用什么代理指标
not_applies: 评估集本身不可信或与线上分布不一致；或指标定义随批次变化导致不可比
data_needed: 预测值与对应标签（即时或延迟）；无标签期需要置信度、输入统计等代理
sources: Montgomery (2019) Introduction to Statistical Quality Control, Wiley；Krawczyk et al. (2017) A survey on drift adaptation in data stream classification, Information Fusion；Sculley et al. (2015) Hidden technical debt in machine learning systems, NeurIPS
---

# 一句话结论

性能监控要回答两个问题：**"变差了多少"** 和 **"证据够不够格触发动作"**。有标签时看均值与分布（RMSE/MAE/bias/分位数），标签延迟时用残差代理与置信度代理，并明确标注代理的不可靠性。

## 适用条件

- 存在可复算的评估协议（固定切分、固定指标实现）。
- 需要决定是否触发校准/重训/回滚，而不是只出报表。

## 方法步骤

1. **回归**：滚动窗口算 RMSE、MAE、bias（平均残差）、残差标准差与 P90 绝对误差；bias 系统性正/负提示输出校准问题，方差变大提示输入噪声或概念漂移。
2. **分类**：滚动窗口算 macro-F1、每类 recall、预测类别先验、置信度均值与熵；置信度骤降（分布外）先于准确率下降出现。
3. **标签延迟**：滞后 k 轮才有标签时，先监控代理（输入统计、置信度、预测分布），标签到位后再回填真实指标；代理只做预警，不做 promote 证据。
4. **不对称代价**：工业 RUL 常用 NASA score 等非对称指标；同时报告 RMSE（平台主指标）与代价指标，避免优化一侧牺牲另一侧。
5. **触发规则**：性能变化超过"2×种子噪声"或 2×margin 才触发动作；连续 N 轮无改善触发 explore/panel，而不是无限重试。
6. **隔离证据源**：监控报告集只用于事后陈述，禁止用它选择动作；selection 用于门控；这是本仓库三层切分的硬约束。

## 与 harness 的映射（可执行）

| 字段/工具 | 用途 | 备注 |
|---|---|---|
| `performance_before/performance_target/performance_drop` | 同窗口父模型指标变化 | drop 为相对变化，正负号含义随任务方向 |
| `metrics.primary`、`rmse`、`mae`、`bias`、`nasa_score` | 回归多维诊断 | 主指标是 RMSE，bias/nasa 只做解释 |
| `macro_f1`、`per_class_f1`、`severe_miss_rate` | 分类诊断 | 类别缺失会扭曲 macro-F1 |
| selection / report 切分 | 门控 / 报告 | report 不参与接受拒绝 |
| `--seed` 与种子复跑 | 估计噪声阈值 | A 约 0.0029、B 约 4.72、C 约 0.35 |

## 如何验证

- **重放**：同一 run 目录用 records.jsonl 复算每轮指标，与 state.json 对账（本仓库 Recorder/StateDB 支持）。
- **噪声基线**：同一配置跑 3 个 seed，取主指标标准差作为"无改善"判定带；提升小于 1σ 视为噪声。
- **标签延迟仿真**：把标签人为延后 k 轮，验证代理预警与真实指标回填的一致性。
- **反事实检查**：性能下降但输入分布几乎不变时，优先怀疑标签/概念漂移或评估协议变化。

## 数据需求

每轮的预测与标签（至少抽样）、切分 manifest、随机种子、指标实现版本。无标签期需要模型置信度与输入统计；报告需注明代理指标与真实指标的时间差。

## 实验证据（公开 + 本仓库）

- 本仓库严格官方口径 run：A selection macro-F1 0.7032→0.7346；B RMSE 678.02→409.74；C RMSE 44.75→40.52；A 外部探针 0.4054→0.5239（只验证）。
- 本仓库发现：B 的 `shrink α=0` 让所有候选输出等于训练目标均值，selection 指标完全相同——门控正确识别为"无提升"，说明监控必须同时看指标与预测分布，不能只看一个数字。
- 公开：Sculley 2015 指出"模型上线后数据依赖会悄悄腐化"，性能监控必须与输入监控配对（Krawczyk 2017）。

## 风险与不适用条件

- **把报告集当调参证据**会造成隐性过拟合，属于泄漏（见 `compliance-data-boundary`）。
- 小窗口指标方差大（A 的 Toluene 仅 5 行），单批变化不宜触发大动作。
- 指标实现不同（NASA score 归一化方式）会让跨仓库比较失效。
- 反馈回路：用模型自己的预测做标注/筛选会放大偏差。
- 代理指标与真实性能的相关性未经本地验证时，只能做低优先级预警。

## 来源

Montgomery, Introduction to Statistical Quality Control, 8th ed., Wiley 2019；Krawczyk et al., Information Fusion 2017；Sculley et al., NeurIPS 2015；本仓库 docs/12_EXPERIMENTS.md 与 docs/11_COMPLIANCE.md。
