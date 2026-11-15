# 10 · 系统架构

## 10.1 设计目标

1. **合规**：工业模型只做官方 baseline 的续训/推理期校准，不改结构、归一化与
   checkpoint 格式；每轮有父→候选的 sha256 链；训练/选型/报告数据严格分离。
2. **自主**：没有"if metric < x: 换配方"的硬编码决策树。诊断、实验设计、评审、
   裁决由 LLM Agent 在受控工具集内完成；所有候选必须来自白名单策略库。
3. **可控**：预算、参数区间、数据可见性、沙箱、tripwire 五层约束；任一层失效
   都不影响交付（确定性地降级）。
4. **可审计**：每轮假设 / 修改 / 指标 / sha256 / 用量落盘；`workspace_report.json`
   直接对齐平台 schema。

## 10.2 分层视图

```
                    ┌─────────────────────────────────────────────┐
                    │              rsi/runner.py                  │
                    │  3 tasks × N rounds · budget · state · 报告 │
                    └───────┬─────────────────────────┬───────────┘
                            │                         │
             ┌──────────────▼───────────┐   ┌─────────▼────────────┐
             │  rsi/agent/coordinator   │   │ rsi/records + state  │
             │  Diagnostician           │   │ records.jsonl        │
             │  Experimenter  ──┐       │   │ rounds.jsonl         │
             │  Critic          │ veto  │   │ state.json (dashboard)│
             │  Arbiter (edge)  │       │   └──────────────────────┘
             │  Panel (stuck)   │       │
             └───────┬──────────┘       │
                     │ tool calls       │
        ┌────────────▼───────────────┐  │
        │ rsi/agent/runtime.py (PI)  │  │
        │  Agent + hooks + skills    │  │
        │  before_llm_call tripwire  │  │
        └────────────┬───────────────┘  │
                     │ ExperimentProposal │
        ┌────────────▼───────────────┐  │
        │ rsi/executor.py            │  │
        │  参数/区间二次校验          │  │
        │  subprocess + timeout      │  │
        └────────────┬───────────────┘  │
                     │ job.json          │
        ┌────────────▼───────────────┐  │
        │ rsi/worker.py (隔离进程)    │  │
        │  strategies → candidate.npz│  │
        │  selection/report/probe    │  │
        └────────────┬───────────────┘  │
                     │ metrics + sha256   │
        ┌────────────▼───────────────┐  │
        │ rsi/selection.py gate       │  │
        │ promote / retain / rollback │  │
        └────────────────────────────┘  │
```

## 10.3 目录与模块职责

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `rsi/tasks.py` | 三任务静态定义（包名、类别、主指标、基线值、噪声阈值 margin） | 只读官方 `baseline-info.json` |
| `rsi/splits.py` | 官方验证集 → calibration/selection/report + A 的 UCI 外部探针 | 组不重叠（C 单元级、B 轴承内行级并披露）；切分一次固化并记 sha |
| `rsi/drift.py` | 特征偏移、KS、相关性、标签分布、窗口性能跌幅 → severity/causes/evidence | 只读统计，不下结论 |
| `rsi/strategies.py` | 白名单策略：模型层（retain/continue_train/ridge/SWA）+ 推理期（recenter/shrink） | 结构/归一化不可变；参数 clamp |
| `rsi/executor.py` | 实验编排：写 job.json、spawn worker、超时、CPU/实验计数、失败留 log_tail | 每个实验独立目录，worker 不写外部 |
| `rsi/worker.py` | 单实验执行：训练/校准/保存/评估/写 result.json | 不读隐藏数据；异常自包含为 result.json |
| `rsi/selection.py` | 门控：`delta ≥ margin` promote；`delta ≤ -2×margin` rollback；否则 retain | 分类看升、回归看降；margin 来自基线噪声估计 |
| `rsi/inference.py` | recenter（batch_mean/batch_std，weight 混合）、shrink、因果流式 | 只改预测，不改 checkpoint 字节 |
| `rsi/records.py` | JSONL + state.json 原子写 | 追加式、可重启恢复 |
| `rsi/runner.py` | 主状态机、预算、Panel 触发、summary/workspace_report | 单任务失败不终止全局 |
| `rsi/predict.py` | 平台独立预测入口 | stdin/stdout 逐行 JSON，顺序一致 |
| `rsi/agent/*` | PI 运行时、6 角色、提示词、工具契约、Hook | 无 LLM 时确定性降级 |
| `rsi/dashboard/*` | 零依赖 WebUI | 只读 state.json，不参与决策 |

## 10.4 一轮的完整时序

1. **数据切分**（任务开始一次）：`build_splits` 把官方 validation 切成三份互斥
   子集，写 `splits/manifest.json`（行数 + sha256 + 协议）。
2. **父模型评估**：在 calibration/selection/report/probe 上计算父指标，作为本轮
   gate 的基线。
3. **漂移窗口**：按 seed+round 从 selection 中取累计窗口，与 train 比较得到
   `DriftReport`，序列化后注入 Agent 上下文。
4. **诊断**：Diagnostician 调 `get_context(section=...)` 取数，调
   `submit_diagnosis` 提交结构化归因（severity/causes/focus/evidence/confidence）。
5. **实验设计**：Experimenter 调 `get_context(catalog)` 看白名单与参数范围，提交
   1–3 个 `submit_plan`（strategy/params/postprocess/hypothesis/expected_effect/risk）。
6. **评审**：Critic 逐条 `approve/veto`，否决理由必须具体（越界、与假设矛盾、
   零信息量、泄漏风险）。
7. **执行**：Executor 二次校验（参数 clamp、类型、区间、postprocess 合法性），
   最多执行 2 个候选；worker 子进程训练/校准/评估，写回 metrics + 新 checkpoint
   sha256。
8. **门控**：`best_candidate` 按主指标方向选最优候选；在 2×margin 边界触发
   Arbiter（LLM 只能更保守）；最终 promote/retain/rollback。
9. **记录**：`rounds.jsonl` + `state.json` 写入本轮全部字段；看板实时显示。
10. **收尾**：任务结束写 `A/summary.json`（含 narrative），全局写
    `summary.json` 与 `workspace_report.json`；`state.json` 置 done。

## 10.5 状态与恢复

- `state.json` 是**只读快照**，由 `recorder.write_state` 在每轮后原子替换；
  看板只读它，不写。
- `records.jsonl` 是追加日志，进程崩溃后可用 `read_records()` 重建时间线。
- `run_dir/<task>/experiments/exp-NNNN/` 保留每个实验的 `job.json`、`worker.log`、
  `candidate.npz`、`result.json`，便于答辩时逐实验复盘。
- Runner 未实现"断点续跑"，但每个实验目录本身可独立复现；正式评测只需一次
  完整 6h 内跑完，断点续跑不是必需项。

## 10.6 预算与超时

| 资源 | 默认上限 | 计量点 | 超限行为 |
|---|---|---|---|
| wall clock | 21600s（6h） | runner 每轮检查 | 停止新任务/新轮次，写 summary |
| CPU 秒 | 36000s | executor 累计 worker `process_time` | 同上 |
| 实验数 | 40 | executor 计数 | 停止新实验，只能 retain |
| LLM token | 2,000,000 | runtime usage 汇总 | tripwire 拒绝后续调用，确定性降级 |
| 单实验超时 | 1800s | `subprocess.run(timeout)` | 杀进程，记失败 |
| 单次 LLM 调用 | 240s | `asyncio.wait_for` | 记超时，角色回退 |

## 10.7 降级阶梯（保证评测环境永远有输出）

```
LLM 可用？ ──否──> 确定性 planner（SEEDED 策略序列）──> 正常闭环
   │是
   ├─ 单角色失败 ──> 该角色确定性回退（诊断由 drift 证据构造，实验用 planner 候选）
   ├─ token 预算耗尽 / context > 92% ──> tripwire，后续全确定性
   ├─ 全部候选失败 ──> retain 父模型（安全状态）
   └─ 任务异常 ──> 标记 failed，继续下一任务，仍写合规 summary
```

## 10.8 安全与合规边界

- Agent 工具集只有 `get_context` + `submit_*`，**没有** shell/文件/网络工具；
- 所有训练数据来自官方 public + 内部切分；probe 数据（UCI）只用于报告，代码路径
  与训练完全分离；
- 实验只能在 `run_dir/experiments/` 下写文件，worker 用相对路径 + 沙箱目录；
- 不访问隐藏测试流、不读取平台凭据、不发起未授权外网请求；
- 镜像内无升权、无持久化系统修改；Python 依赖固定版本；
- 受控迭代：LLM 每轮最多 3 个 plan，Critic 有 veto，预算硬上限，Panel 最多 2 次。
