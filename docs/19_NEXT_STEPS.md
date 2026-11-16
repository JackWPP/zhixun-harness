# 19 · 恢复后的 8–9 小时验证计划

本计划针对当前 checkout 的真实状态制定：官方 A/B/C 训练起点已在本地且哈希匹配；
2026-11-15 的 canonical 运行产物和镜像缺失；A/B/C 官方预测输入契约及在线 Kernel
证据正在补齐。试运行次数有限，所以下面的阶段都先在本地/自有服务器完成，不在计划
中自动启动平台试运行或正式评测。

| 时间块 | 工作与退出条件 | 预计用时 |
|---|---|---:|
| P0 修复与留档 | `pytest` 全绿；A/B/C 官方输入的 batch/stream predictor 测试全绿；Agent tool、history/search、reviewer 均不暴露 report/probe；技能只写入 run state；关键路径写清 SHA 和当前版本 | 40 分钟 |
| 远端镜像 | 通过 SSH 将不含凭据、缓存和 Git 元数据的源码/数据快照传到 Ubuntu amd64；容器内 `selftest A/B/C` 通过；`docker run` 无交互完成 A×1；A/B/C 官方样例逐行预测符合 task schema | 65 分钟 |
| LLM 兼容矩阵 | Tokenplus Qwen3.8 对 A/B/C 各做 1 轮 Kernel；每轮至少 10 turns，要求真实 `propose → run → compare → decide → finish`，核对 usage/token 账、selection-only 输入、记忆/技能是否实际改变下一轮；超 120k tokens 或 6 分钟即停止该轮 | 135 分钟 |
| 离线完整基准 | 在远端跑严格官方数据 A/B/C × 10、固定 seed；导出每轮 metrics、sha、资源和回滚记录；与本地 v0.5.1 文档结果比较。A report 若下降、模型输出若越过 baseline，先诊断再继续 | 55 分钟 |
| 稳定性复验 | 选 A 的 recenter、B 的 damage_shrink、C 的 staged continuation，用第二个 seed 做针对性复验；不重复完整搜索。对候选增益做 paired/bootstrap 置信区间，检查 batch/stream 差异 | 75 分钟 |
| 镜像与评测彩排 | 构建单个 `linux/amd64` docker-save tar；检查 ≤64 MiB 的模型产物包、入口权限、预测 stdout、非零错误码、`/work` 写入、超时和断网降级；保存镜像 digest 和 tar SHA | 65 分钟 |
| 云端记录与文档 | 本地保留 JSONL/SQLite/manifest；确认 SwanLab 是否可登录并真实同步所有候选、决策、指标和产物。未确认云端链接前，只称“本地完整审计记录”，不称“云端已归档” | 35 分钟 |
| 冻结候选 | 完整代码/镜像/模型配置三者哈希固定；一人复核源声明与报告；形成一次可提交的 review 包，然后再人工决定是否占用线上试运行 | 20 分钟 |

## 每轮实验决策规则

1. 提案必须从公开 drift 证据出发，写出单一、可证伪假设及对应主指标。
2. 训练与拟合只用官方 train；selection 用于候选门控；report/probe 只留在 supervisor
   审计结果中，不进入模型提示、检索、记忆或技能。
3. 每轮最多 3 个候选。任何候选未达到门控 margin 都保留父检查点；已 promote 后若
   新输入窗口退化，运行时还需有 previous champion 和确定性恢复路径。
4. LLM 仅提出/排序实验；Guard 负责参数、数据与预算约束；独立评估器复算结果；所有
   自动决策与 fallback 都明确记录来源，不能把 supervisor autoclose 记作 Agent 决策。
5. 先完成离线和自有服务器验证。正式平台最多 10 次试运行、正式评测最多 2 次；每次
   平台运行前冻结镜像并跑完整 preflight，确认当时的试运行额度和版本状态。

## 需要真实确认的事项

- PyTRiO 提交页当前可选 `Qwen/Qwen3.8-27B` 或获准的 Sampler；Tokenplus 只用于开发
  期验证。PyTRiO endpoint 和鉴权尚未得到验证。
- 平台 `train / evaluate / output` SDK 的隐藏流数据时序、标签预算与 checkpoint 生效
  接口仍需以平台 SDK/答疑记录确认。CLI predictor 合规不等于流式闭环已接入。
- 规则要求完整云端实验记录链接/查询依据；代码当前的 SwanLab 记录是 offline mode，
  必须真实同步并核实完整字段，或补充其他认可的云端查询依据。
- 当前 LLM 冒烟只完成一次 A×1，且发生 6-turn 自动收尾；必须补足明确 decide/finish
  的多轮原始轨迹后，才能声称“Agent 自主完成闭环”。
