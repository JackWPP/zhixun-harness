# Overnight log — v0.6 整夜自主 RSI 优化

Goal（**工作方式**，不是交付物里的硬目标）：用会话 goal + workflow + 智能体团队组织 Lead 的一整夜工作；
harness 内部只有“软目标 + 硬预算”，目标未达成也必须在预算内产出有效预测/报告并降级收尾。

- 开始：2026-11-15 23:10（Asia/Shanghai）
- 目标：2026-11-16 07:45 前完成并汇报
- 正式口径：**只用官方数据**（`RSI_ALLOW_EXTENDED_DATA=0`）；外部数据仅 Track X 测试床
- 平台：不上传镜像 / 不创建提交 / 不试运行（无 PyTRiO Key，正式评测未开放）
- 基线冻结：git tag `v0.5.1`（HEAD 6a2f754）；v0.5.1 strict canonical：A 0.734555 / B 409.736985 / C 40.523699
- 环境：Python 3.12.16 (.venv) + numpy 1.26.4；本机 12 线程/15GiB；比赛目标 4 核 8G

## 时间表（滚动更新）

| 时间 | 阶段 | 状态 |
|---|---|---|
| 23:10–23:20 | P0 冻结/日志/任务板/spawn | ✅ |
| 23:20–00:30 | P1 批次协议 + 并行 + emulator | 🚧 |
| 00:30–01:40 | P2 知识库 + 目标/状态 + 看板 v3 | ⏳ |
| 01:40–03:20 | P3 性能冲刺 + Track X | ⏳ |
| 03:20–04:50 | P4 全量 3×10 + kernel LLM | ⏳ |
| 04:50–06:00 | P5 镜像 0.6.0 + submission + SwanLab | ⏳ |
| 06:00–07:00 | P6 文档/evidence/commit | ⏳ |
| 07:00–07:45 | 汇总汇报给用户 | ⏳ |

## 已完成

- [x] git tag `v0.5.1`；`.scratch/night/`、`.scratch/workflow/`、`docs/evidence/` 建立
- [x] 任务板：task-3 driftlab / task-4 contractlab / task-5 webui；spawn driftlab、contractlab；webui 派活
- [x] 会话 goal（id `goal-bc9090d3-c73e-4d56-8ebe-e73b07019e5c`）已设置；mission goal 不进交付物（用户 m03315 纠偏）
- [x] `rsi/stream/` 批次协议实现：
  - `sources.py`：official（官方 validation 顺序切片，累计标签训练）、external-a（UCI batch3-10，Track X）
  - `protocol.py`：每批临时 package、decision.json、predictions.jsonl（官方输出形状）、train-only postprocess 评估
  - `runner.py`：每批复用 `Runner(rounds=1)`；promote/retain/rollback 门控 + retention 测遗忘；`--parallel-tasks` 线程并行
  - `rsi/cli.py stream` 子命令；`tests/test_stream.py` 6 用例（含 2 批端到端 105s）
- [x] `Runner.__init__(configs=[...])` 允许在批次 package 上复用完整训练循环

## 运行中的实验

- Track X 外部 A 批次 smoke：`.scratch/stream-x-smoke`（2 批，UCI b3/b4，experiments_cap=2）

## 待办 / 下一步

- P1：契约 emulator 与 contractlab 结果合并；`scripts/run_overnight.sh` 一键夜间编排
- P2：driftlab 知识库/诊断/检索工具接入 kernel；GoalLoop advisory 展示；state v3 + 看板 v3（webui）
- P3：A/B/C 性能冲刺（已冻结配方只做合规内增益）；Track X 8 批四臂
- P4：strict 3×10 + kernel LLM 3 轮 + 并行时间测量
- P5：镜像 0.6.0（4C8G/2GB 约束）、submission、SwanLab 证据
- P6：README/docs/evidence/commit

## 风险与降级

- 若 LLM 慢/超时：kernel 自动降级 pipeline，记录 `degraded`
- 若内存紧张：`--parallel-tasks 1`，降低 experiments_cap
- 若 emulator 契约推断错误：保持“只读校验 + 不提交”，等 PyTRiO Key 后再 trial 验证
- 若整夜任务未全部完成：保留已完成证据 + 明确未完成项，不虚构指标

## 23:25 更新（P1/P2 收尾）

- [x] P1 批次协议：`rsi/stream/` 完成；`stream` 子命令可跑；Track X 外部 A 2 批 smoke 通过
      （b3 promote recenter 0.6212 / b4 retain 0.7083，decision/predictions/stream_report 齐全）
- [x] `docs/21_STREAM_PROTOCOL.md`（输入 batch package / 输出 decision+predictions / Track S/X / 平台对接）
- [x] `scripts/run_overnight.sh`（严格 3×10 → kernel 3 轮 → Track X 4 批 → 契约自检 → 镜像重建）
- [x] P2 知识库：driftlab 交付 `rsi/knowledge.py` + `knowledge/` 12 篇 + 索引（fts5）；三个接线缺口已补：
      prompt 渲染 `brief["knowledge"]`；`_kernel_brief` drift 白名单加 drift_type/knowledge_hints/attribution；
      kernel env 的 knowledge_searches/hits 合并进 state，`_task_state.knowledge` 采用 dashboard 约定
      `{available,searches,hits,docs[],skills_used}`。
- [x] 状态 v3：`schema_version=3`、`goal_soft`（A 0.7346/B 409.74/C 40.524，advisory）、`knowledge`、`batches`；
      kernel 记录 `agent.proposal_sources`；pipeline 记 seed 来源。
- [x] 看板 v3（webui）：schema 3、`/api/goal`、`/api/knowledge`、提案来源/软目标/批次/知识面板；17 tests passed。
- [x] 测试：`pytest tests -q -m "not integration"` → 81 passed。
- [x] 夜间长跑启动：
      - strict 3×10 pipeline → job `bash-f950b4b0991040e0`，`.scratch/night-strict`，日志 `.scratch/night/strict.log`
      - kernel LLM A 3 轮（strict, round-turns 16, exp-cap 8, wall 5400）→ job `bash-37230e680fde41c7`，`.scratch/night-kernel`，日志 `.scratch/night/kernel.log`

## 仍未完成

- contractlab：baseline 重下比对、`docs/19_SUBMISSION_CONTRACT.md`、`scripts/contract_check.py`、`scripts/emulate_stream_platform.py`
- driftlab：`rsi/drift.py` 增强（drift_type/attribution/recommended_actions/knowledge_hints + to_dict 顶层 score）与 2 个技能、docs/20
- P3 性能冲刺与 Track X 4 批完整四臂；P4 kernel 验证结果收集；P5 镜像 0.6.0/submission/SwanLab；P6 文档与 commit

## 23:52 更新（kernel 夜间 run 完成）

- [x] kernel LLM A 3 轮（strict）完成：`.scratch/night-kernel/rsi-20261115-232545`
      0.7032→0.720696（r1 promote，r2/r3 retain）；3 轮 0 失败；`search_knowledge` 1 次/3 hits；
      stall→panel 1 次（4 角色）；463,125 tokens、wall ≈27 min；证据写入 docs/18 §10.4。
- [x] 契约层双写：`decision.json` 现同时含扁平平台字段（`decision` 字符串、`round_index`、
      parent/candidate/checkpoint sha256、`validation_metrics`）与 `decision_detail` 嵌套记录；
      `docs/21` 同步；`tests/test_stream.py` 5 unit passed。
- [x] 镜像链路更新：`stage_image.sh` 打包 `knowledge/`；entrypoint 新增 `stream` 子命令与
      `RSI_STREAM_SOURCE/BATCHES/HOLDOUT/DIR`；`verify_assets.py` 增加知识索引自检（PASS）。
- [x] 新增 `tests/test_goal_nonblocking.py`（4 passed，软目标不阻塞完成）。
- [x] `scripts/testbed_four_arms.py`（Track X 四臂证据导出：frozen/recenter/full-harness/oracle）。
- [x] SwanLab 离线冒烟：`.scratch/swan-smoke/rsi-20261115-234801` 生成本地 run；
      `docs/22_SWANLAB.md` 写明 sync/公开方式（token 不落盘）。
- [x] driftlab 交付已合入：`rsi/drift.py` 增强（drift_type/attribution/recommended_actions/
      knowledge_hints/next_checks + to_dict score）、`knowledge/` 12 篇、skills 新增 2 个、
      `docs/20_DRIFT_KNOWLEDGE.md`、`tests/test_drift_extra.py`；全量非集成测试 **104 passed**。
- [ ] strict 3×10 仍在跑（C 阶段继续训练较慢），Track X 4 批排队中（同一后台脚本串行）。



## 最终状态（2026-11-16 00:30）

- strict 3×10 复现完成：A 0.734555 / B 409.736985 / C 40.523699；wall 1897s、CPU 1081s、56 实验。
- Track X 4 批（batch6–9）+ 四臂证据完成：full harness 4/4 优于 frozen/recenter（docs/12 §12.21）。
- kernel 验证：A 3 轮（0.7032→0.7207，panel+知识检索）、B 2 轮（0.6780→0.6022，未发现强配方，已如实记录）。
- 镜像 0.6.0 重建、verify_image PASS、docker load 通过；submission 包由整夜 strict run 重新生成。
- 全量测试 108 passed（不含 integration）。
- 未上传、未提交、未试运行（无 PyTRiO Key；用户要求先不动平台）。
- 已知差距：kernel 的候选/知识清单未显式包含 damage_shrink 等强配方；正式评测契约仍需有 Key 后 trial 验证。
