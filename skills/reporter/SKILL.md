---
name: reporter
description: 任务收尾时生成可审计的技术摘要与实验记录
version: 1
tags: report, records, narrative
---

# 何时用
最后一轮 finish_round 之后，或收到收尾指令时。

# 步骤
1. 汇总十轮的 hypothesis → experiment → selection 指标 → decision → checkpoint sha。
2. 统计 promote/retain/rollback 次数、实验数、CPU 秒与 token 用量。
3. 给出基线→最终的对照，并区分 selection、report 与外部 probe 证据。
4. 列出未解决问题与下一轮建议（不要夸大收益）。
5. 检查记录完整性：每轮是否都有父/候选 sha 与决策理由。

# 坑
- 不要用训练日志自报指标替代平台评测。
- 不要把 probe 指标写进 agent 决策依据。
- 数字必须能追溯到 rounds.jsonl / workspace_report。

# 验收
摘要中每个数字都能在记录里找到对应字段；缺失项明确标注为未验证。
