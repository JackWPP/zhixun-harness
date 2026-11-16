---
name: drift-diagnosis
description: 从漂移报告判断成因（协变量/先验/概念漂移）并选择第一条实验方向
version: 1
tags: drift, diagnosis, A, B, C
---

# 何时用
每轮开始读 drift 报告后，若 severity 为 moderate/severe，或历史连续两轮 retain。

# 步骤
1. 先确认监测时点。`prequential_covariate_only` 只能看到输入特征；此时 label_shift、标签误差和概念漂移均为未知，不得推断当前性能下降。
2. 同时读 median/MAD `feature_shift`、`shifted_fraction`、`ks_p95` 和 `correlation_shift`。单个 `max_feature_shift` 或 KS 极值只作为局部线索，不单独触发严重级别或模型更新。
3. 定位前几个特征，区分整体位置变化、分布形状变化与相关结构变化。若证据互相矛盾或窗口很小，明确写“归因不确定”，先 retain 或提出低成本诊断实验。
4. KS/相关变化不能直接证明传感器增益漂移，也不能推出 `batch_std` 更合适。A 的 batch_std 已有退化证据；优先比较 parent、保守 batch_mean recenter 与 shrink，batch_std 仅能作为隔离候选并通过 selection 门控。
5. 只有标签按平台时序解封后，才更新 prequential loss、label_shift 或性能下降；区分标签先验变化和特征条件关系变化，不把协变量变化称为概念漂移。
6. 结论写成一句带证据、适用边界和拒绝条件的可证伪假设，交给 propose_experiment。无标签证据不足时不启动续训。

# 坑
- 不要用单批 val 的指标当漂移证据：A 的验证批次类别极偏。
- 协变量漂移 ≠ 概念漂移；后者需要新标签，不能只靠无监督校准。
- 外部探针只做证据，不进入决策上下文。

# 验收
每轮至少引用两个当前可见数值字段，注明窗口行数和监测时点；声明一个可证伪假设、一个具体候选策略及拒绝条件。无标签阶段不得引用 label_shift 或性能 drop。
