---
name: leakage-audit
description: 提交前检查训练/选择/报告数据是否隔离，避免用验证集训练或调参
version: 1
tags: leakage, compliance, audit
---

# 何时用
每轮 decide 之前，以及任务收尾写报告之前。

# 步骤
1. 核对本轮实验的 data 参数：只允许 train 或 train+calibration。
2. 核对选择依据：只使用 selection split 的公开指标；report split 只用于汇报，不能据此改参数。
3. 核对历史：promote 决策的候选必须来自父 checkpoint 的真实续训或推理期校准。
4. 核对记录：每轮写下 hypothesis、change_summary、父/候选 checkpoint sha、指标、decision。
5. 发现越界立即 rollback，并把教训写入 memory。

# 坑
- 归一化与目标缩放只能在训练集拟合，续训沿用父模型统计。
- 外部/私有数据只能做推理探针，标签不得进入训练或决策。
- 候选与父指标完全相同时仍要如实记录（retain 是允许的）。

# 验收
任何一轮都能回答：用了哪些数据、依据哪份指标、父模型是谁、最终保留哪个 sha。
