# 18 · Agent Kernel（自主闭环）

v0.3.0 的 runner 是"固定六步流水线"：drift → diagnose → propose → criticize → execute → gate，Agent 从未看到实验结果，也不能决定下一步。v0.4 把控制权交给 LLM 内核，确定性只保留在**安全边界**（Guard）与**收尾兜底**（supervisor）。

## 1. 两种控制器

| | pipeline（确定性） | kernel（自主） |
|---|---|---|
| 决策者 | 规则路由 + 预置策略库 | PI Agent（LLM） |
| 看到实验反馈 | 否（下一轮才间接使用） | 是（`run_experiment` 返回公开指标） |
| 能否改写计划 | 否 | 是（随时 propose/run/decide/delegate） |
| 适用 | 无 LLM、调试、消融基线 | 正式评测与演示 |

`--controller auto`：LLM 端点可用 → kernel，否则 pipeline；kernel 连续 2 轮失败自动降级 pipeline 并在记录中标记 `fallback`。两条路径产出同一套 `rounds.jsonl` / `state.json` / `workspace_report.json`。

## 2. 一次 kernel 轮次

```
supervisor (runner)
  ├─ drift_report(selection 窗口)            # 确定性
  ├─ brief = 任务/父指标/预算/记忆快照/lessons/胜率/历史/漂移/目录/技能索引
  ├─ kernel.run_round(host, brief)           # PI Agent, max_turns=20
  │    observe → hypothesize → experiment → evaluate → decide → remember
  │    工具：get_run_state / get_history / get_lessons / search_records
  │          propose_experiment / run_experiment / compare_candidates
  │          decide / finish_round / delegate / convene_panel
  │          memory_* / skill_manage / get_context / list_skills / read_skill
  ├─ guard 复核（每次动作一个审计条目）
  ├─ 应用 decision：promote 更新 parent checkpoint + postprocess，retain/rollback 保留
  ├─ GoalLoop.judge（fail-open）             # 是否继续下一轮
  └─ Recorder → JSONL + SQLite；MemoryReviewer / SkillCurator（任务收尾）
```

Kernel 忘记 `decide` 时 supervisor 用 `best_candidate + selection.gate` 自动收口（`kernel_autoclose`）；忘记 `finish_round` 时自动摘要；没有任何实验且没有 decision → 本轮失败并回退 pipeline。

## 3. Guard：模型"能做"与"允许做"的边界

| 规则 | 实现 |
|---|---|
| 只能训练 `train` / `train+calibration` | `Guard.ALLOWED_TRAIN_DATA`，`train+all-public` 明确拒绝 |
| 每轮 ≤3 个实验、每任务 ≤40 | `validate_experiment` 计数 |
| promote 必须过统计门控 | `check_decision` 用 `selection.gate()` 重算，LLM 只能更保守（把 promote 降为 retain/rollback） |
| 策略/参数/后处理白名单 | `validate_catalog` → `validate_proposal` |
| 不许把私有/隐藏/probe 路径交给训练 | 训练数据枚举，不存在"任意路径"入口；probe 指标永不进 Agent 上下文 |
| 记忆/技能注入与危险内容 | `scan_text` / `check_skill_text`；技能禁代码围栏、外链、危险命令 |
| 所有动作可审计 | `env.trace` + `records.jsonl` + `state.json` 的 `agent` 字段 |

## 4. 多 Agent

- 角色：Experimenter（设计候选）、Diagnostician（归因）、Critic（审查/否决）、Arbiter（边界裁决）、Reporter（收尾摘要）。
- `delegate(role, brief)`：隔离子上下文、独立工具集、≤6 turns，结果以摘要返回主内核。
- `convene_panel(issue)`：4 角色 `asyncio.gather` 并行给立场，最多 2 次/任务；卡住或指标反复不升时触发。
- 每个角色的 system prompt 只包含该角色需要的边界与输出契约（见 `rsi/agent/prompts.py`）。

## 5. 目标循环（Goal Loop）

`GoalState` 持久化在 SQLite；每轮结束由确定性 judge 判定：

- 完成 10 轮 → done；
- 预算或 wall clock 耗尽 → done；
- 降级模式下完成 `target-2` 轮 → done；
- 否则 continue（fail-open，judge 异常不阻塞）。

continuation 是普通 user message，不改 system prompt（保护 prefix cache）。子目标固定四条：完成轮次、每轮有有效实验或明确 retain 理由、只用允许数据、预算内保留最优模型。

## 6. 预算与时限

| 项 | 默认 | 说明 |
|---|---|---|
| 每轮 kernel turns | 20 | `--round-turns` / `RSI_ROUND_TURNS` |
| 单轮 wall | 1800s | `RSI_ROUND_TIMEOUT` |
| 单实验 | 900s | `RSI_EXPERIMENT_TIMEOUT` |
| 实验工具 timeout | max(实验+240, 600)s | PI 的 `tool_timeout` 必须覆盖同步子进程 |
| token | 开发 0=不限，比赛 8,000,000 | `--token-cap` / `RSI_TOKEN_CAP`；`TokenBudget` 记账并在 92% 触发 tripwire |
| 整 run wall | 21600s（6h） | 平台上限；wall 95% 停止开新实验 |

## 7. 上下文与提示词

- kernel system prompt 冻结（约 1.2K tokens），brief 目标 <4K tokens，单轮上限约 8K。
- 只注入公开信息：官方指标、selection 窗口、漂移统计、历史决策、记忆与技能索引。
- 实验返回被裁剪为 `selection/report` 的少量字段；不返回 probe、外部数据或隐藏集。
- LLM 端点：OpenAI 兼容（`LLM_BASE_URL/LLM_API_KEY/LLM_MODEL`），本地 `.scratch/llm.env` 不入库；比赛时切换 PyTRiO 端点只改 env。

## 8. 失败与降级矩阵

| 故障 | 行为 |
|---|---|
| LLM 超时/断网/400 | 本轮 `ok=false` → pipeline fallback，记录 error |
| 连续 2 轮 kernel 失败 | 本任务剩余轮次切 pipeline |
| 实验子进程崩溃 | `result.ok=false` + `log_tail`，Agent 可换策略重试 |
| Agent 重复无效计划 | Guard 拒绝 + 胜率提示；连续 stall 触发 panel |
| 记忆文件被外部改动 | 备份 `.bak` 并拒绝覆盖，事件告警 |
| token 预算 92% | `before_llm_call` 抛错，当前轮收口，进入预算停止 |
| DB 损坏 | 备份后用 `replay_from_jsonl` 重建 |

## 9. 与"完全自由 Agent"的区别

- 目标固定：最大化 `selection` 主指标，同时守住赛题约束（只用公开数据、续训自父检查点、不改隐藏标签）。
- 行动空间固定：策略/参数/后处理白名单 + 工具集合；没有 shell、没有网络、没有任意外部数据入口。
- 结果可复现：同一 seed + 同一 LLM 温度下 trace 记录完整；LLM 非确定性显式记录在报告中，不冒充确定性。
- 可解释：每轮都有 hypothesis / experiment / decision / evidence，可直接映射到赛题"实验记录规范"。

## 10. 验证里程碑

| 里程碑 | 证据 |
|---|---|
| 工具层完整 | `tests/test_kernel_offline.py`（Guard 拒绝、完整一轮、自动收口） |
| 真实 LLM 闭环 | A 任务 1 轮：17 tool calls、3 实验、promote 0.7032→0.9256、agent 自写记忆（`docs/12_EXPERIMENTS.md`） |
| 多轮自主 | A 任务 × 3 轮 kernel run：18/14/14 tool calls，promote→promote→retain，selection 0.7032→0.9373，361K tokens，wall 694s（§10.1） |
| 跨 run 记忆 | 第 2 次运行 brief 注入 1562 chars memory + 414 chars lessons，首轮 tool calls 18→13，直接复现已验证配置（§10.2） |
| 合规内核闭环 | A 任务 × 3 轮 strict kernel run（仅官方 train）：promote→retain→retain，selection 0.7032→0.7207，8 次记忆写入、1 次自建技能、1 次 panel，440,715 tokens，wall 1443s（§10.3） |
| 不回归 | 离线 3×10 与 v0.3.0 指标一致或更好 |

### 10.1 A 任务 3 轮真实 LLM 轨迹

| 轮 | 决策 | 实验 | 结果 | calls | latency |
|---|---|---|---|---|---|
| 1 | promote | continue_train(lr=1e-3, ep20, train+calibration, class_balance) | 0.7032→0.9256 | 18 | 217s |
| 2 | promote | retain + recenter(batch_std, w=0.4) | 0.9256→0.9373 | 14 | 174s |
| 3 | retain | batch_std w=0.5、ridge_last_layer | 均 < 父模型 | 14 | 112s |

第 3 轮的假设直接引用第 2 轮记忆（"batch_std 权重 0.4 是局部最优…勿再加大"）。

### 10.2 跨 run 记忆生效的证据

第 2 次运行（复用同一 db 与 memory 目录）首轮：

- brief：`memory_chars=1562`、`lessons_chars=414`、`skills_chars=516`；
- 假设文本明确写"复现已验证配置（记忆 exp-0003, sha f59e5037）"；
- 工具调用从 18 降到 13，input tokens 110,219→107,823；
- 首轮仍 promote 到同一水平（0.9256），说明省下的是探索成本而非指标。

这正是 G3 想验证的"记忆改变行动"，而不仅是"记忆文件存在"。

### 10.3 合规口径下的内核证据（仅官方 train）

2026-11-15 19:58，A 任务、3 轮、`--controller kernel --llm on`、`RSI_ALLOW_EXTENDED_DATA=0`、
`--memory` 开启（run `rsi-20261115-195836`）：

| 轮 | 决策 | 主要候选 | selection | tool calls |
|---|---|---|---|---|
| 1 | promote | recenter(batch_mean, w=0.4) | 0.7032→0.7207 | 14 |
| 2 | retain | recenter 系数与 ridge 均未过 margin | 0.7207 | 17 |
| 3 | retain | 同上；连续 2 轮无提升触发 panel（4 角色） | 0.7207 | 17 |

- 3 轮全部由 kernel 自主完成，0 次降级、0 次 guard violation；
- 8 次 `memory_add`（task scope）、1 次 `skill_manage`（agent 自建/修补技能）；
- 1 次 `convene_panel`：diagnostician/experimenter/critic/reporter 四角色会诊，
  结论为"类 3 在重度协变量漂移下坍缩，继续调 recenter 收益有限"；
- token 440,715（3 轮），wall 1443s，CPU 109.9s；对比 v0.3 固定 pipeline，A 的
  提升幅度变小（0.7346 strict pipeline vs 0.7207 kernel），这是合规约束下的真实
  上限，也是"Agent 会自我收敛到保守策略"的证据；
- 已知问题（已修）：panel 计数此前未写入 `agent_stats["panels"]`，
  `_record_panel(count=True)` 只在 supervisor 触发的 panel 上计数，kernel 工具
  触发的 panel 通过 `KernelEnv.summary()` 计数，避免重复。
