---
title: 实验记录与 SwanLab 日志：可复现、可审计、可回放
slug: swanlab-logging
tags: logging, swanlab, reproducibility, records, 日志, 记录, 复现, 审计
applies: 每次运行/每轮实验都需要记录参数、指标、决策与产物；提交前核对可复现性
not_applies: 纯离线只读分析（仍建议记录查询与结论）；无法记录产物的环境
data_needed: run 配置、每轮指标与决策、checkpoint 路径与 sha、代码/数据版本、耗时
sources: 本仓库 docs/12_EXPERIMENTS.md 与 docs/17_REPRODUCIBILITY.md；SwanLab 官方文档（离线记录与同步）；MLflow/W&B 实验管理实践（作为公开方法参考）
---

# 一句话结论

记录的目标不是"多"，而是**可回放**：给定记录，能重建同一配置、同一数据、同一决策。每轮写清「假设 → 参数 → 指标（selection/report/probe 分开）→ 决策 → 原因」，并把 code sha、数据 sha、checkpoint sha 绑定在一起。

## 适用条件

- 所有正式运行与人工实验；提交材料需要证明过程可复现。
- 调试/探索运行可降级，但关键结论必须有完整记录。
- 评测环境禁止联网时，采用**本地 JSONL/JSON + 离线日志目录**，联网时再可选同步。

## 方法步骤

1. **run 级元数据**：run id、任务、seed、代码 commit、数据 manifest sha、开始/结束时间、预算、入口命令与环境变量（不含凭据）。
2. **round 级记录**：轮次、父模型、假设、策略+完整参数、后处理、门控 margin、selection/report 指标、决策（promote/retain/rollback/explore）、原因、耗时/token 用量。
3. **产物绑定**：checkpoint 路径 + sha256、日志文件、面板报告；失败轮也记录错误类型（OOM/NaN/超时）而不是只记成功。
4. **敏感信息**：绝不记录平台凭据、隐藏标签、短信/验证码、cookie；路径中也不应出现敏感串。
5. **离线优先**：本地写 records.jsonl / rounds.jsonl / summary.json；SwanLab 仅在允许联网时 `--swanlab` 同步，且只同步安全字段。
6. **回放与对账**：从记录重建 run 配置；对账 state.json、rounds.jsonl、summary.json 与 report 指标；抽样复跑确认一致。
7. **保留策略**：小文件长期保留，大 checkpoint 按需归档；提交物只含必要文件。

## 与 harness 的映射（可执行）

| 产物 | 内容 | 位置 |
|---|---|---|
| `records.jsonl` | 每轮完整记录（参数/指标/决策/耗时） | run 目录 |
| `rounds.jsonl` | 轮次摘要与门控结果 | run 目录 |
| `summary.json` | 最终指标与最佳轮 | run 目录 |
| `workspace_report.json` | 面板/报告数据 | run 目录 |
| `--swanlab` | 可选在线同步 | 仅在允许联网时 |
| `make_submission` | 打包 checkpoint + 记录 | 提交目录 |

## 如何验证

- **对账脚本**：读 records.jsonl 重算每轮 primary 指标，与 summary 对比；不一致即记录 bug。
- **复跑一致性**：同 seed 重跑首轮，指标应在噪声内一致；不一致检查随机性来源（seed/线程/数据顺序）。
- **审计抽样**：随机抽一轮，仅凭记录恢复配置并复跑。
- **敏感串扫描**：grep 记录与日志中的 token/password/cookie 等模式；命中即脱敏。
- **缺失检查**：确认失败实验也在记录中，否则统计胜率会偏。

## 数据需求

run 配置、每轮指标与决策、checkpoint 血缘、代码/数据版本。离线环境下不需要外部服务；联网同步是可选增强，不是正确性前提。

## 实验证据（公开 + 本仓库）

- 本仓库 docs/12：严格 run 的 selection/report 指标、种子复跑与外部探针结论均来自 records/rounds 记录；`--swanlab` 为可选开关，离线 JSONL 记录是权威来源。
- 本仓库 docs/17：提交与复现要求（代码+数据+种子+记录）明确；state.json 与 summary.json 对账是验收步骤。
- 公开：SwanLab 提供实验跟踪与离线记录能力；MLflow/W&B 的 run/metric/artifact 划分是实验管理的通行做法。
- 诚实提醒：本仓库未对在线同步链路做端到端评测；知识库文档只记录字段与流程，不宣称云端可用。

## 风险与不适用条件

- 记录隐藏标签或平台数据会直接违反合规；云端同步前必须脱敏。
- 只记录成功实验会让 strategy_stats 失真；失败与错误必须入账。
- 指标定义中途变更会让历史记录不可比；变更需版本化。
- 大文件全量入库会拖慢运行；记录路径与 sha 而非二进制本体。
- 时钟/随机性来源未固定时，"同 seed 一致"无法保证。

## 来源

本仓库 docs/12_EXPERIMENTS.md、docs/17_REPRODUCIBILITY.md；SwanLab 官方文档；MLflow Tracking 文档；W&B 实验管理实践（公开方法参考）。
