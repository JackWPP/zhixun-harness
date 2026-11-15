---
title: 漂移类型学：协变量、标签/先验与概念漂移
slug: drift-taxonomy
tags: drift, taxonomy, diagnosis, covariate, label, prior, concept, 漂移, 类型学, 诊断
applies: 监控指标下降或分布告警后，需要先判断"漂移是哪一类"再决定实验方向
not_applies: 尚无参考窗口（只有单批数据）；或数据标签本身不可信、连训练集都无法定义
data_needed: 参考窗口与目标窗口的特征列（无标签即可）；若判断标签/概念漂移还需两侧标签
sources: Gama et al. (2014) A survey on concept drift adaptation, ACM CSUR 54(3)；Lu et al. (2018) Learning under concept drift: A review, IEEE TKDE 30(12)；Moreno-Torres et al. (2012) A unifying view on dataset shift, Pattern Recognition 45(1)
---

# 一句话结论

先分类再动手：**P(X) 变 = 协变量漂移**（可用无监督校准缓解）；**P(Y) 变 = 标签/先验漂移**（用类别权重或输出校准，recenter 有风险）；**P(Y|X) 变 = 概念漂移**（无监督校准基本无效，必须用标签重训或换头）。

## 适用条件

- 有稳定的参考分布（训练集或上一阶段窗口），且目标窗口行数足够（分类每类 ≥ 20、回归 ≥ 200 更稳）。
- 监控指标（macro-F1 / RMSE）出现超出门控 margin 的变化，或特征统计量报警。
- 目标是**选择下一步实验**，而不是解释物理机理。

## 方法步骤

1. **对齐两个窗口**：参考窗口取 train 或 calibration（不许用 selection/report 的标签做判断依据），目标窗口取当前轮流的 selection 子窗口。
2. **看协变量**：逐特征标准化均值差 `Δ_j = |μ_t - μ_r| / σ_r`，统计 `shifted_fraction = mean(Δ_j > 1)`；配合逐特征 KS。`feature_shift>0.45 且 shifted_fraction>0.1` → 协变量漂移候选。
3. **看标签/先验**：分类比较两类先验的 JS 散度与缺失/新增类别；回归看目标均值漂移（以 σ 归一）与标准差比。`label_js>0.1` → 标签/先验漂移候选。
4. **看性能残差**：同一切分上父模型的指标变化。`|performance_drop|>0.15` 而协变量位移不明显 → 概念漂移候选。
5. **交叉裁决**：多因同时命中记为复合漂移；只有协变量没有标签变化时，优先怀疑传感器增益/环境（见 `sensor-gain-baseline-drift`）。
6. **写一句可证伪假设**：例如"batch 级均值偏移导致 coverage 失配，batch_mean recenter 应把 selection macro-F1 提高 ≥ margin；若 report 反而下降则假设被否"。

## 与 harness 的映射（可执行）

| 报告字段 | 判读 | 候选动作 |
|---|---|---|
| `feature_shift`、`shifted_fraction` | >0.45 且 >0.1 | `recenter(batch_mean)`，w 从 0.3 试起 |
| `ks_max`、`ks_p95` | 个别特征极强偏移 | `recenter(batch_std)` 或按特征分组校准 |
| `label_shift.js`（分类） | >0.1 | `continue_train(class_balance=True)`；慎用 recenter |
| `label_shift.mean_shift_sd` / `std_ratio`（回归） | 均值/方差同时变 | `shrink` 或重训，不要盲目 recenter |
| `performance_drop` | 超 2×margin 且协变量小 | 概念漂移，转 `model-transfer-continued-training` |
| `causes` 含 `sensor_or_environment_gain_drift` | 无需标签 | 增益/基线校正，见专门文档 |

## 如何验证

- 造一个已知漂移的对照：对参考窗口做固定 gain/offset 或标签重采样，确认检测器能在预期方向报警（本地可用 `rsi.drift` + 合成数据复算）。
- 决策验证只看 selection 指标是否越过 margin；report 只在决策后复算一次，防止用报告集反向调参。
- 复合漂移时做单因素消融：先只改推理期校准，再只改训练策略，比较各自贡献。

## 数据需求

参考≥数百行、目标窗≥数百行；至少 2 个窗口才能谈"漂移"。分类需要完整类别表（缺失类别本身是强信号）；回归需要目标均值/方差。标签非必需，但有标签才能区分概念漂移。

## 实验证据（公开 + 本仓库）

- 公开：概念漂移综述把漂移分为 sudden/gradual/incremental/recurring，并指出无监督检测器无法区分 P(X) 与 P(Y|X)（Gama 2014；Moreno-Torres 2012）。
- 本仓库：A 任务官方验证集是单批 batch2，类别极偏（Toluene 仅 5 行），因此单批 macro-F1 抖动不能当漂移证据；UCI 外部 10 批探针上基线宏 F1 0.4054 → recenter 后 0.5239（只验证、不训练）。
- 本仓库：B/C 的 report 切分指标明显差于 selection（C：57.46 vs 44.75），说明单元间方差大，单窗口结论必须标注不确定性。

## 风险与不适用条件

- **检测器 ≠ 因果归因**：统计位移可能来自采样、批次构成变化或标签缺失，不是真实物理漂移。
- 小样本下 JS/KS 剧烈波动；类别缺失会让 macro-F1 结构性变化，应单独报告。
- 概念漂移没有无监督解；用协变量方法处理概念漂移是常见错误。
- 参考窗口自身被污染（例如复用验证集）会让所有比较失真。

## 来源

Gama et al., ACM Computing Surveys 2014；Lu et al., IEEE TKDE 2018；Moreno-Torres et al., Pattern Recognition 2012；Quiñonero-Candela et al., Dataset Shift in Machine Learning, MIT Press 2009。
