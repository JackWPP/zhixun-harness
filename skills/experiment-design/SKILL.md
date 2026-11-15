---
name: experiment-design
description: 为单轮设计 1-3 个可证伪、单变量、可复现的实验
version: 1
tags: experiment, hypothesis, budget
---

# 何时用
诊断完成后、调用 propose_experiment 之前。

# 步骤
1. 每个实验只改变一个因素：策略、学习率、epoch、后处理类型或混合权重。
2. 假设写成"如果 X，则 selection 主指标相对父模型变化 ≥ margin"，并说明机制。
3. 同时给出一个保守对照（retain 或 shrink），避免整轮实验全失败。
4. 参数必须在策略目录范围内；每轮最多 3 个实验，优先低成本策略（retain/recenter 先于 continue_train）。
5. 预期风险写清楚（过拟合、漂移加剧、预算消耗）。

# 坑
- 不要在同一轮同时改学习率和后处理权重，否则无法归因。
- 不要用 report split 或 probe 指标做选择，只看 selection。
- 训练数据只能用 train 或 train+calibration。

# 验收
每个实验都有：策略、参数、后处理、假设、预期效果、风险；执行后可用 selection 指标证伪。
