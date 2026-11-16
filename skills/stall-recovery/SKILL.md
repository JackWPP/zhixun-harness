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
3. 读候选的 promotion_gate 与逐类支持数：若增益只来自极少样本、支持充分类别退化，保留父模型并换机制；不得请求审计集指标。
4. 需要时 delegate 给 Critic 做失败归因，或 convene_panel 召专家组（每任务最多 2 次）。
5. 把本轮结论写成 lesson（避免/优先），下一轮 brief 会自动带上。

# 坑
- 不要没有新假设就增加 epoch；先区分输出头不足、输入尺度与类别不均衡，再用独立候选证伪。
- 不要连续重复同一失败实验；查历史中的策略、参数、后处理与父 SHA，避免重复消耗预算。
- 预算紧张时优先保留当前最优模型并写清降级决策。

# 验收
破局轮必须说明：上次失败原因、本轮改变了哪个因素、预期改善多少。
