---
title: 合规与数据边界：训练/验证/报告切分与外部数据使用
slug: compliance-data-boundary
tags: compliance, leakage, data, boundary, split, 合规, 泄漏, 数据边界, 切分
applies: 任何实验设计、校准拟合、外部数据使用、记录与提交之前；评审怀疑泄漏时
not_applies: 已明确处于离线沙箱之外的本地复现实验（仍需遵守数据许可）
data_needed: 切分 manifest、数据许可信息、实验记录、checkpoint 血缘
sources: 本仓库 docs/11_COMPLIANCE.md 与 docs/15_DATA.md；UCI Gas Sensor Array Drift 数据集许可（CC BY 4.0）；NASA C-MAPSS / PRONOSTIA 公开数据使用条款；比赛平台赛题规则与提交规范
---

# 一句话结论

把数据按 **train（可拟合）/ calibration（有限拟合）/ selection（门控）/ report（只报告）/ hidden（不可见）** 分层使用；外部数据只能做 **probe（只验证）**，不能参与拟合，除非规则显式允许且 opt-in 打开。任何越界使用都视为泄漏，收益作废。

## 适用条件

- 设计实验、选择校准参数、决定是否引入外部数据时。
- 评审或自查需要证明"没有使用不该用的标签"。
- 记录/提交前核对 checkpoint 与数据血缘。

## 方法步骤

1. **先写数据地图**：每份数据标注来源、许可、可用角色（train/calibration/selection/report/probe）、sha256、行数、切分方式。
2. **拟合边界**：模型参数、特征统计、校准中心/权重只能在 train（及规则允许的 calibration）上估计；selection 只做选择决策；report 只在决策后算一次。
3. **外部数据三种用法**：只读探针（验证鲁棒性，不拟合）；显式训练扩展（仅当规则允许且 opt-in）；完全不使用。默认只用第一种。
4. **禁止项**：读取平台隐藏标签/凭据；联网抓取；修改评测系统/平台状态；改变网络结构或加载协议；用 report 反向调参；用目标批标签拟合校准。
5. **可追溯**：每次实验记录 code sha、数据 sha、父/候选 checkpoint sha、使用的切分；失败实验同样记录。
6. **提交前检查**：候选 checkpoint 结构与元信息未变、只含续训参数、无外部数据混入（除声明外）、无凭据/隐私字段。

## 与 harness 的映射（可执行）

| 约束 | 落地位置 | 证据 |
|---|---|---|
| 只能用官方 train 拟合 | 策略参数 `data="train"` | records.jsonl 的 strategy.params |
| 外部数据只 probe | A 的 `extended_pretrain` 需 `RSI_ALLOW_EXTENDED_DATA=1` | 默认关闭，记录 opt-in |
| 选择只看 selection | 门控逻辑 | rounds.jsonl 的 decision + metrics.selection |
| report 只报告 | 最终评估 | report 不出现在决策输入 |
| checkpoint 必须可官方加载 | 加载器校验 | structure sha / meta 对比 |
| 禁止联网/读写平台 | Docker 网络隔离、agent 工具白名单 | 运行时无 shell/网络工具 |

## 如何验证

- **静态检查**：grep 代码中是否有网络调用、平台 API 写入、绝对路径外写；检查 agent 工具清单是否含 shell/网络。
- **记录审计**：随机抽 3 轮，用 records.jsonl 重放决策，确认拟合数据角色合法。
- **血缘校验**：对候选 checkpoint 重新计算结构摘要，确认与父模型仅参数不同、元信息/normalize 协议一致。
- **外部数据审计**：确认探针数据未进入任何 fit 调用；`RSI_ALLOW_EXTENDED_DATA` 未开启。
- **泄漏反查**：若指标异常高，检查是否用了 report 标签、时间索引、单元泄漏或伪标签。

## 数据需求

切分 manifest（含 sha）、数据许可与出处、实验记录、checkpoint 血缘信息。知识库本身也需记录来源（公开论文/标准名称）且不含抓取内容。

## 实验证据（公开 + 本仓库）

- 本仓库 docs/11：G1 禁止改网络结构、G2 只允许续训候选、G3 外部数据只做 probe；agent 无 shell/网络；提交物为 checkpoint + 记录。
- 本仓库 docs/15：A 官方 train/val=batch1/batch2；batch3–10 仅作漂移探针（13,910 行，只验证不训练）；A 数据集为 UCI #224，CC BY 4.0。
- 本仓库教训：C 第二轮 +0.0106 曾在 margin=0 时被接受——记录可回溯才让它成为可解释的阈值错误而非隐性调参。
- 公开：数据许可要求署名/一致性使用（UCI CC BY 4.0）；比赛规则对隐藏标签与平台访问有明确禁止项。

## 风险与不适用条件

- 规则文本模糊时，"没被禁止"不等于"允许"；保守做法是不用。
- calibration 切分的标签可用于轻量校准，但若用于模型选择则越界。
- 外部数据即使只做 probe，也可能因分布重叠间接泄漏信息；需在报告中声明。
- 记录缺失会让合规无法自证；宁多记不少记。

## 来源

本仓库 docs/11_COMPLIANCE.md、docs/15_DATA.md；UCI ML Repository #224（CC BY 4.0）；NASA C-MAPSS 数据说明；PRONOSTIA/IEEE PHM 2012 数据说明；比赛平台赛题与提交规则。
