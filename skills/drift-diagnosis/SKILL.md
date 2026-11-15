---
name: drift-diagnosis
description: 从漂移报告判断成因（协变量/先验/概念漂移）并选择第一条实验方向
version: 1
tags: drift, diagnosis, A, B, C
---

# 何时用
每轮开始读 drift 报告后，若 severity 为 moderate/severe，或历史连续两轮 retain。

# 步骤
1. 看 feature_shift 与 max_feature_shift：整体偏移大而形状不变 → 协变量漂移，优先推理期 recenter。
2. 看 ks_max/ks_p95：个别特征强偏移 → 传感器增益/环境漂移，recenter 的 batch_std 模式更合适。
3. 看 label_shift（分类）与 mean_shift_sd（回归）：标签分布明显变化时，recenter 有风险，改用 shrink 或保留父模型。
4. 看 performance_before/after drop：若 drop 超过 2 倍 margin，说明现有 checkpoint 已失效，优先校准而不是继续训练。
5. 结论写成一句可证伪假设，交给 propose_experiment。

# 坑
- 不要用单批 val 的指标当漂移证据：A 的验证批次类别极偏。
- 协变量漂移 ≠ 概念漂移；后者需要新标签，不能只靠无监督校准。
- 外部探针只做证据，不进入决策上下文。

# 验收
每轮诊断至少引用 2 个数值字段（feature_shift、ks_max、label_shift 或 performance drop），并指向一个具体策略族。
