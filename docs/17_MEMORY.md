# 17 · 记忆与技能（Memory & Skills）

本文描述 v0.4 新增的"可沉淀、可复用、可审计"的记忆子系统。它解决旧版本的核心缺口：每轮实验之间、每次运行之间，Agent 不携带任何经验，10 轮等于 10 次独立尝试。

## 1. 三层结构

| 层 | 载体 | 生命周期 | 容量 | 谁写 |
|---|---|---|---|---|
| 工作记忆（frozen snapshot） | 每轮 brief 的内存字符串 | 单轮 | ≈1.5K tokens | Recall 组装，只读 |
| 长期记忆 | `memory/MEMORY.md`、`memory/TASK_<A\|B\|C>.md` | 跨轮、跨 run | 4000 / 2000 字符 | Agent（`memory_add/replace/remove`）+ 每任务 Review |
| 经验索引 | `rsi_state.db`（SQLite WAL + FTS5） | 跨 run | 无界 | Supervisor 每轮事务写入 |

技能（程序性记忆）单独存放：`skills/<name>/SKILL.md`，索引进 brief，正文按需 `read_skill` 加载（渐进披露，参考 Hermes / agentskills.io）。

设计原则（与 Hermes 对齐）：

- **快照冻结**：记忆在任务开始时渲染一次并注入 kernel 的 system/brief，之后本轮内的写入只回显给工具，不改变当前 prompt —— 保护 prefix cache，行为可复现。
- **有界策展**：容量满时工具返回当前条目列表并要求 `replace`/`remove`，而不是无限追加；条目用 `§` 分隔，去重后写入。
- **外部漂移保护**：写入前 round-trip 校验文件；若检测到手工改动（非本进程写入），先把现有内容备份为 `.bak.<timestamp>` 再拒绝覆盖，避免静默吞掉队友的修改。
- **注入扫描**：记忆/技能文本写入前扫描越权指令（"忽略上一条"、`http(s)://`、代码围栏、`rm -rf` 等），命中则拒绝并记审计事件。记忆只是数据，不是系统提示。
- **审计真源**：`records.jsonl` / `rounds.jsonl` 仍是审计真源；SQLite 可从 JSONL 重建（`StateDB.replay_from_jsonl`），坏库会备份并重建。

## 2. 文件与目录

```
<state-dir>/<run-id>/memory/
  MEMORY.md            # 环境/工具/通用经验（4000 字符）
  TASK_A.md            # 任务 A 专属（2000 字符）
  TASK_B.md
  TASK_C.md
<state-dir>/rsi_state.db # 结构化经验库（默认跨 run 复用）
skills/                  # 仓库内，seed 技能 + Agent 新建技能
  drift-diagnosis/SKILL.md
  experiment-design/SKILL.md
  calibration-playbook/SKILL.md
  leakage-audit/SKILL.md
  stall-recovery/SKILL.md
  reporter/SKILL.md
```

环境变量：`RSI_MEMORY_DIR`（默认 `<run-dir>/memory`）、`RSI_DB_PATH`（默认 `<state-dir>/rsi_state.db`）。`--no-memory` 关闭记忆与 DB（消融实验用，planner 仍可用胜率排序）。

## 3. SQLite schema（节选）

| 表 | 关键列 | 用途 |
|---|---|---|
| runs | run_id, controller, llm_model, tasks, seed, status | 运行索引 |
| rounds | run_id, task, round_index, decision, primary_value, ... | 每轮记录 |
| experiments | run_id, task, round_index, experiment_id, strategy, postprocess_type, primary_value, promoted, cpu_seconds | 每次实验 |
| llm_calls | run_id, task, role, ok, input/output/reasoning_tokens, latency | token 计量 |
| lessons | task, scope, text, evidence, wins, uses | 经验条目 |
| strategy_stats | task, strategy, postprocess_type, trials, promotions, mean_delta, best_delta | 策略胜率 |
| skill_usage | skill, run_id, task, uses, last_used_at | 技能遥测 |
| events | run_id, task, level, message | 与 JSONL 同步 |
| session_fts / session_text | kind, task, text | FTS5 全文检索（不可用时 LIKE 回退） |
| goals | session, state_json | GoalState 持久化 |

## 4. 一轮中的记忆流

1. `run_task` 开始时 `Recall.memory_snapshot(task)` 渲染冻结快照；`Recall.lessons`、`Recall.stats_text` 从 DB 取历史。
2. `brief` 携带：任务/指标/父模型指标/预算/记忆快照/lessons/策略胜率/最近 5 轮/漂移/策略目录/技能索引。
3. Agent 通过工具读写记忆：`memory_add`（scope=task|global）、`memory_replace`、`memory_remove`、`skill_manage`。
4. 本轮结束：`Recorder` 把 round/experiments/llm_calls/strategy 写入 DB；`MemoryReviewer` 每任务收尾生成 1-3 条 deterministic lesson（可选用 LLM 精炼），写 DB + 高价值条目进 `TASK_<X>.md`。
5. `SkillCurator` 统计技能使用；连续 ≥2 个 run 未使用的非 seed 技能归档（archive 不删除）。

## 5. 策略胜率如何影响决策

`planner.rank_proposals(plans, stats)`：同一 (strategy, postprocess) 的历史 `win_rate = promotions / trials` 降序排序，稳定排序；本轮 seed plan 无历史时置顶以保留探索；无 stats 时保持原序。kernel 模式下胜率作为 brief 的一部分交给 LLM 参考；pipeline 回退时直接按胜率排序。

## 6. 消融与"如实报告"

- 运行 A：`--memory on`（默认），记录每轮 agent 是否引用记忆/技能/检索。
- 运行 B：`--no-memory`，其他条件相同。
- 比较：primary 曲线、达到同等指标所需实验数、连续 stall 轮数。若记忆没有带来稳定收益，如实写入 `docs/12_EXPERIMENTS.md`，并把默认策略回退为"仅胜率排序"。
- 记忆内容只来自公开训练/验证数据与运行统计；不写隐藏标签、不写外部数据。私有/外部数据只进入 `probe` 报告。

## 7. 已知限制

- 快照冻结意味着同一任务内新写入的记忆要下一轮/下一次运行才生效（这是有意的，换取缓存与可复现性）。
- Review 的 deterministic 版只做规则提炼；LLM 精炼是可选增强，失败不影响主流程。
- FTS5 在极旧 SQLite 上不可用，回退 LIKE 检索，召回质量下降但在报告里标注。
- 记忆是纯文本，不是参数；比赛评分中的"单位增益成本"按 token/CPU 计入，记忆本身不改变 checkpoint。
