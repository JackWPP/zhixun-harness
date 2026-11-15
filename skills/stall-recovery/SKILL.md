---
name: stall-recovery
description: 连续多轮无提升时的破局顺序与回退策略
version: 1
tags: stall, recovery, panel
---

# 何时用
连续 2 轮 retain/rollback，或 selection 主指标停滞且漂移仍为 severe。

# 步骤
1. 先读 get_history 与 get_lessons：确认失败是否重复同一策略族。
2. 换策略族而不是只调学习率：训练类 → 校准类，或闭式 ridge → SWA。
3. 检查是否已接近 selection 上限：若 report 与 selection 同步改善，可保留；否则回退父模型。
4. 需要时 delegate 给 Critic 做失败归因，或 convene_panel 召专家组（每任务最多 2 次）。
5. 把本轮结论写成 lesson（避免/优先），下一轮 brief 会自动带上。

# 坑
- 不要用更多 epoch 硬扛；A 上继续训练会让外部漂移批变差。
- 不要连续重复同一失败实验；Guard 会拒绝并把违规写进记录。
- 预算紧张时优先保留当前最优模型并写清降级决策。

# 验收
破局轮必须说明：上次失败原因、本轮改变了哪个因素、预期改善多少。
