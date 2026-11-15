---
name: drift-evidence-triage
description: 用增强漂移报告（drift_type/attribution/recommended_actions/knowledge_hints/next_checks）做证据分级，选出第一个最便宜的单变量实验
version: 1
tags: drift, triage, evidence, attribution, calibration, A, B, C
---

# 何时用
- 每轮 `brief.drift` 非空，尤其 `severity` 为 moderate/severe，或 `causes` 含多项。
- 报告里 `attribution` / `recommended_actions` / `next_checks` 有内容，但不确定先做哪个。
- 上一轮 retain 或实验被 gate 回滚，需要换一个信息量更大的方向。

# 步骤
1. 读 `drift_type` 与 `causes`：`sensor_gain_drift`/`covariate_shift` 先考虑推理期校准；`label_prior_shift` 先看类别覆盖与重加权；`concept_drift` 才考虑续训。
2. 看 `attribution` 前 3 行：`direction`、`std_ratio`、`ks` 判断是整体仿射（增益/基线）还是个别特征异常；只报 top 特征编号与幅度。
3. 读 `next_checks`：先做其中不耗实验预算的检查（样本量、缺类、单位/量程、bias 分区间）。
4. 从 `recommended_actions` 选**风险最低且单变量**的一条：优先 postprocess 校准（不改 checkpoint），再考虑 `ridge_last_layer` 诊断，最后才 `continue_train*`。
5. 用 `knowledge_hints` 检索知识库，把选择理由写成 `[K#]` 引用 + 本报告数值，形成可证伪假设。
6. 若 `severity=severe` 且证据存疑（小样本、缺类、指标刚过阈值），先跑 retain+none 负对照再动模型。

# 坑
- `attribution` 是相关证据不是因果：特征 3 偏移不代表改特征 3；先做整批校准对照。
- `recenter` 会改变推理输入统计量，`batch_std` 比 `batch_mean` 更激进；一次只改 mode/weight 一个因素，注意过校。
- 回归任务优先 `shrink`/`damage_shrink`（只用官方 train 标签），不要用目标窗口标签调参。
- 概念漂移的续训必须用两阶段低学习率；一轮全量重训容易把父模型优点覆盖掉。
- 报告稳定（stable/mild）时不要为了"有事做"而 propose；retain 是合法且常见的正确动作。

# 验收
- 本轮结论至少引用 `drift` 的 3 个字段（type/attribution/score/next_checks 之一类），并说明为何排除其他动作。
- 提案是单变量、参数在 catalog 范围内、有明确"否则放弃"的判据。
- 实验后在 finish_round 里记录：报告预测与实际 selection 变化是否一致，不一致时写一条 lesson。
