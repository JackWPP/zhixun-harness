# 14 · Agent 设计（运行时 / 角色 / 工具 / Hook）

本文件说明 harness 的"策略层"如何工作：谁在决策、能看到什么、不能做什么、失败
时怎么办。工业模型层的改进始终由 `rsi/strategies.py` 的白名单完成，Agent 只做
**选择与审查**，不写模型代码、不执行任意命令。

## 14.1 为什么是 PI Agent

需求：成熟循环、工具注入、对上下文与 token 的显式控制、能离线嵌入镜像。
实测四种候选（详见 `docs/16_DECISIONS.md` ADR-003）：

| 候选 | 结果 |
|---|---|
| pi-agent-core-lc 0.9.1 | ✅ 工具调用/thinking/usage 全正常，44MB，Hook 与 schema 完备 |
| openai-agents 0.24 | ✅ 可用，59MB，留作备选 |
| pydantic-ai-slim 2.68 | ❌ Qwen thinking 模式下 HTTP 400（tool_choice 冲突） |
| DSH 0.2.2-rc.1 | 可行但 928MB + Node，不适合评测镜像 |

选择 pi-agent-core-lc，并用 `pi-agent-harness-lc` 的 sessions/compaction 作为历史
管理的补充；不自己实现 agent loop。

## 14.2 运行时封装（`rsi/agent/runtime.py`）

```python
class PIRuntime:
    def available(self) -> bool: ...          # 懒加载，pi 未安装/key 缺失则为 False
    def call(self, role, system_prompt, user_prompt, toolset_kind, context) -> RoleResult
    def _before_llm_call(self, current_context, last_budget): ...   # tripwire
    def _event_loop(self): ...                # 持久事件循环（修复 httpx 绑定旧 loop）
    def status(self) -> dict: ...             # backend/model/calls/tokens
```

关键点：

1. **单次调用 = 短会话**：每个角色每次调用新建 `Agent`，但共享进程级事件循环与
   LLM 连接池；调用结束即回收消息历史，避免上下文膨胀。
2. **Hook 闸门**：`before_llm_call` 在 `fraction > 0.92` 或 token 预算耗尽时直接
   raise，PI 捕获后终止该轮调用；Runner 记录 `tripwire` 事件并切确定性策略。
3. **usage 记账**：遍历 assistant message 的 `usage.input/output/reasoningTokens`，
   累加到 `TokenBudget`，写 `records.jsonl` 的 `llm_call` 事件。
4. **降级**：`available()=False`、超时、异常、空响应都返回 `RoleResult(ok=False)`，
   由 Coordinator 走确定性回退——**任何单次失败都不会终止整个 run**。

## 14.3 六角色与调度表

| 角色 | 职责 | 触发频率 | 上下文 | 工具 |
|---|---|---|---|---|
| Diagnostician | 读漂移报告，给出 severity/causes/focus/evidence | 每轮 1 次 | task+drift+history | `get_context`, `submit_diagnosis` |
| Experimenter | 从策略库提 1–3 个候选（假设/预期/风险） | 每轮 1 次 | +catalog+metrics | `get_context`, `submit_plan` |
| Critic | 逐条审批/否决，检查越界、矛盾、泄漏、零信息量 | 每轮 1 次 | +plans | `get_context`, `submit_review` |
| Arbiter | 仅当提升幅度在 2×margin 边界且有 ≥2 候选时裁决 | 边界情况 | +review+gates | `get_context`, `submit_verdict` |
| Reporter | 任务结束写 narrative（引用轮次/指标/sha） | 每任务 1 次 | +全部轮次摘要 | `get_context` |
| Panel | 卡住时四角色各表态，找新方向 | 最多 2 次/任务 | +issue+history | `get_context`, `submit_standpoint` |

正常一轮 = **3 次 LLM 调用**（诊断/实验/评审）+ 0–2 次执行；Arbiter 很少触发；
Reporter 每任务一次。以实测平均 ~6K tokens/次估算，3 任务 × 10 轮 ≈ 25–30 万
tokens，远低于 800 万平台预算，也远低于单任务 200 万内部上限。

## 14.4 工具契约（`rsi/agent/tools.py`）

Agent 能调用的**全部**工具：

| 工具 | 参数（pydantic） | 作用 |
|---|---|---|
| `get_context` | `section: task\|drift\|history\|metrics\|catalog\|budget` | 读取只读上下文片段，≤6000 字符 |
| `submit_diagnosis` | `summary, severity, causes, focus, evidence, confidence` | 提交诊断（唯一出口） |
| `submit_plan` | `strategy, params_json, postprocess_json, hypothesis, expected_effect, risk` | 提交候选实验 |
| `submit_review` | `approvals: list[int], vetoes: list[str], note` | 审批/否决 |
| `submit_verdict` | `decision, chosen, reason, confidence` | 边界裁决 |
| `submit_standpoint` | `standpoint, evidence, confidence` | Panel 表态 |

没有 shell、文件写入、网络、代码执行工具；`params_json`/`postprocess_json` 只是
字符串，由 Coordinator 解析后交给 `validate_proposal` 二次校验（策略必须命中
白名单、参数必须落在区间、postprocess 必须合法）。因此即使模型被诱导，也无法
产生越界实验或任意代码执行。

## 14.5 提示词设计（`rsi/agent/prompts.py`）

- 中文 system prompt，硬性声明：不得编造数字、不得访问隐藏标签、只通过工具提交、
  每轮 1–3 个 plan、必须给出可检验的假设；
- user prompt 注入 JSON 上下文块（task/drift/history/parent metrics/catalog/budget），
  ≤8K tokens；工具再按 section 增量取数，避免一次性塞满 64K 窗口；
- `OUTPUT_CONTRACT` 规定每个 `submit_*` 是唯一被采信的输出；自由文本只作解释，
  不参与决策；
- 对 64K/128K 窗口的适配：上下文块 + 工具增量 + 单轮历史 ≤ 12K，远低于上限；
  即便模型只有 64K，也不会触发硬截断。

## 14.6 协调器的确定性约束（`rsi/agent/coordinator.py`）

```python
diagnose(context)            # 失败 → _fallback_diagnosis（由 drift 证据构造）
propose(context, diagnosis)  # 失败/全部非法 → planner.fallback_proposals
criticize(context, proposals) # 无任何 approve → 批准前 2 个（保底）
arbitrate(...)               # 只允许把 promote 降级为 retain/rollback
panel(context, issue)        # 失败 → _deterministic_standpoint
report(context, summary)     # 失败 → 确定性摘要
```

设计原则：**LLM 可以"更保守"，不能"更激进"**。例如 Arbiter 只能否决 promote，
不能把 retain 强行改成 promote；Critic 的否决必须带具体理由；Experimenter 的
方案必须过白名单校验。这样策略层的自我改进被限制在可审计的动作空间内。

## 14.7 上下文与历史

- `build_context()` 生成每轮上下文：任务元信息、漂移摘要、父模型指标、最近 3 轮
  摘要、当前最优轮、策略目录、预算剩余；
- 看板的 `state.json` 与 `rounds.jsonl` 是唯一权威历史，Agent 不直接读文件；
- 单轮内消息在调用结束后丢弃，不跨角色共享（除显式注入的摘要），避免"越聊越
  长"和角色间污染；
- 压缩策略：工具返回小 JSON；历史超限时 `build_context` 只保留最近 3 轮 + 最优轮
  摘要，配合 PI 的 `before_llm_call` 预算闸门兜底。

## 14.8 失败矩阵

| 失败 | 处理 | 是否影响交付 |
|---|---|---|
| 端点 401/超时 | `RoleResult(ok=False)`，确定性回退 | 否 |
| 模型返回非 JSON | `_parse_json_object` 尝试抽取；失败则回退 | 否 |
| 模型提交越界参数 | `validate_proposal` 拒绝；回退 planner | 否 |
| token 超限 | tripwire；后续调用全部走确定性 | 否 |
| 实验 worker 崩溃 | `result.json` 记录 traceback；该候选计失败；其他候选继续 | 否 |
| 全部候选失败 | gate 返回 retain，保留父模型 | 否（性能不倒退） |
| 磁盘/权限异常 | Runner 捕获、tasks 标记 failed，继续下一任务 | 部分任务可能失败 |

## 14.9 与"自研 loop"的差异

自研 loop 的风险不在循环本身，而在：工具调用协议、流式/非流式差异、reasoning
块处理、usage 计量、超时与重试、事件循环生命周期、上下文压缩。这些边界正是
PI 已经解决并实测通过的；我们只需在其上写 6 个角色、6 个工具和一套确定性降级，
代码量与风险都可控（`rsi/agent/` 约 1,400 行）。
