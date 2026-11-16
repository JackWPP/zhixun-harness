# Private-Sim：诚实批次协议与同域漂移适应测试床

> 目的：验证 harness 在**平台按批次下发、标签延迟返回**的协议下，能否只靠"已揭示的历史标签 + 当前批无标签输入"持续适应，而不是靠偷看当前批标签刷分。
>
> 结论先行：**没有任何单一机制在三个任务上都赢**——历史连续训练（adapt）在 A 的外部流和 B 上收益巨大，但在 A 官方小偏批流和 C 上会伤指标；门控（用已揭示的滚动窗口验证后再上线）可以把损失压住，但仍然需要按任务配置。所有数字都在诚实协议下重测过；旧版（把当前批 validation 标签写进 package）的数字已作废。

## 1. 协议泄漏与修复

旧版 `rsi/stream/protocol.py` 把 `batch.validation`（当前批带标签行）写进 package 的 `public/validation.jsonl`，而 `Runner` 用这份数据切 calibration/selection/report 并据此拟合后处理、选择候选——等于**用当前批标签选型**。因此旧数据（A full/adapt hidden 0.76–0.77、pair 0.768 等）都是乐观口径。

修复后的诚实协议（`rsi/stream/sources.py`、`protocol.py`、`runner.py`）：

```text
Batch(index, train, dev, prediction, answers, reveal, meta)
  train       = 官方 train + 之前批次已揭示的标签行（唯一可训练数据）
  dev         = 已揭示历史的 holdout（用于 Runner 切分与门控）
  prediction  = 当前批输入（无标签，写 predictions.jsonl）
  answers     = 当前批标签（只在预测之后评分，绝不回流）
  reveal      = 平台返回标签后进入历史的行；hidden 批 reveal 为空
```

- `materialise_package` 只写 train/dev/prediction，不写当前批标签；
- `decision.json` **在评分之前写出**，只依赖已揭示历史与当前批输入；
- `evaluation.json` 存放 `prediction_metrics/prediction_count/wall_seconds`，答案变化只影响它；
- E0 不变性测试：打乱当前批 answers 后 `decision.json` 与 `predictions.jsonl` 必须逐字节相同（已 PASS）。

## 2. 实验设计（E0–E10）

| 编号 | 内容 | 目的 |
|---|---|---|
| E0 | 打乱 answers 不变性 | 证明协议无泄漏 |
| E1 | arms × seeds 主对照（A: UCI b3–b7 arrival，b8–b10 hidden） | 机制排序 |
| E2 | hidden 诚实留出 | 泛化而非拟合当前批 |
| E3 | 标签稀缺（label_every=2 / label_fraction=0.5 / 叠加） | 平台可能延迟给标签 |
| E4 | 缺类批次（去掉 Acetaldehyde） | 极端先验漂移 |
| E5 | gain/offset/noise 扰动 | 传感器漂移形态 |
| E6 | 长流两轮（b3–b7 重复） | 是否遗忘 |
| E7 | 未知任务零代码（合成 task.json） | 可复用性 |
| E8 | kernel LLM 2 批自主 | Agent 在批次协议下工作 |
| E9 | 官方 validation 8 批重放 | 官方小批流上的行为 |
| E10 | 官方 strict 3×10 回归 | 确认提交路径未受影响 |

Arms：`frozen`（不训练/无后处理）、`recenter`（固定 recenter w=0.4）、`pair`（仅历史标签阈值校准）、`adapt`（历史连续训练 + recenter）、`full`（Runner 实验 + adapt + pair）。

## 3. 结果

### 3.1 A：外部 UCI 流（arrival b3–b7 → hidden b8–b10，macro-F1）

| arm | hidden 均值 |
|---|---|
| frozen | 0.2774 |
| recenter | 0.4453 |
| pair | 0.3539 |
| adapt | 0.7578（seed43 0.7647） |
| full（无门控） | 0.7810（seed43 0.7849） |
| adapt（滚动门控） | 0.7663（seed43 0.7647，gate 4/5） |
| **full（滚动门控）** | **0.7866**（gate 4/5） |

外部多类流上，**历史标签连续训练是数量级收益**（+0.48），门控保留 4/5 次适应。

### 3.2 B/C：官方流（RMSE，越低越好）

下列为**最后一个隐藏批/最后两个隐藏批**上的指标（B/C 隐藏批 = 官方 validation 最后 2 段，各约 209 行）：

| task | frozen | recenter | adapt（无门控） | adapt（滚动门控） | full（滚动门控） |
|---|---|---|---|---|---|
| B | **357.47**（328.6/386.4） | 473.12 | 446.96 | 454.22 | 586.43 |
| C | **56.29**（46.3/66.3） | 57.54 | 58.68 | 57.87 | 未跑 |

注意口径：B **arrival** 批上 adapt 明显优于 frozen（arrival 均值 476 vs 702），但最后两个 hidden 段 frozen 反而更好（357 vs 447）——说明 hidden 段的 target 分布与 arrival 段不同，或 2 个 209 行小批噪声过大。**不能把 arrival 的收益当作 hidden 泛化收益**；B 的默认 arm 仍保留 `adapt` 但结论标注为"混合/待更多隐藏批验证"。C 无漂移，任何适应都伤，默认 frozen。

### 3.3 官方 validation 8 批（E9，A，macro-F1 均值）

| arm | 均值 |
|---|---|
| frozen | 0.6158 |
| pair | **0.6571** |
| adapt（无门控） | 0.5974（第 5、6 批类别极偏时伤） |
| adapt（滚动门控） | 0.5940 |
| full（滚动门控） | 0.6192 |

→ 官方小批流上"每次到达都续训"是负收益，滚动门控也救不回来（.594），full 只是回到 frozen 水平；**官方 A 流应默认 pair**（.6571）。

### 3.4 稳健性（A，adapt，UCI 外部流）

| case | hidden 均值 |
|---|---|
| 正常 | 0.7578 |
| label_every=2 | 0.5812 |
| label_fraction=0.5 | 0.7597 |
| 两者叠加 | 0.5850 |
| 缺 Acetaldehyde | 0.7495 |
| gain / offset / noise / 组合 | 0.7632 / 0.7567 / 0.7572 / 0.7632 |
| 两轮长流 | 0.7631 |

结论：标签稀缺是主要衰减因子，缺类和传感器扰动几乎不伤；长流不遗忘。

## 4. 机制迭代：门控（dev → 滚动已揭示窗口）

第一轮用初始 dev 池做 before/after 门控：A 外部流仍收益（.766/.774），但 B 上把有效适应拒掉（hidden 538.8 vs 不门控 476.1，kept 1/6）。原因：初始 dev 来自官方 train，与后续漂移批次分布不同。

第二轮改为**滚动窗口门控**：用最近一个已揭示批次的标签行（≤2000 行）作为 before/after 的验证集，只在线性感受野内比较，通过 `improved(task, after, before, margin)` 才保留 adapted checkpoint，否则回滚父模型；`decision.json.adapt_gate` 记录 `gate_mode/gate_rows/before/after/kept`。

实测结果：

| case（hidden 口径） | 无门控 | dev 门控 | **滚动门控** |
|---|---|---|---|
| A 外部 adapt | .7578 | .7663 | .7663 |
| A 外部 full | .7810 | .7383 | **.7866** |
| B adapt | 446.96 | 538.83（kept 1/6） | 454.22（kept 4/6） |
| C adapt | 58.68 | — | 57.87（kept 2/6，仍伤） |
| A 官方 adapt（arrival 均值） | .5974 | — | .5940 |

→ 滚动窗口门控对 A-full 有小幅帮助（.781→.787），但对 B/C 与官方 A 无法救回（B 略差、C 和官方 A 仍伤）。所以最终不是"无条件开 adapt"，而是**按任务/来源路由**（E1b）：

| task | official 流 | private-sim / external 流 |
|---|---|---|
| A | `pair` | `full` |
| B | `adapt`（arrival 收益大；hidden 证据混合） | `adapt` |
| C | `frozen` | `frozen` |

该策略已实现为 `rsi.stream.runner.resolve_arm` + `arm="auto"`（CLI 默认），并有单元测试固化；显式传 `--arm` 仍可覆盖。

实跑验证（E1b）：A/official→pair（mean .6571）、B/private-sim→adapt（arrival 479.4 / hidden 454.2）、C/private-sim→frozen（arrival 49.93 / hidden 56.29），0 errors。

## 4b. 可复用性与 Agent 证据

- **E7 未知任务零代码**：用 `scripts/make_demo_task.py` 生成 16 维 3 类合成任务 D（`task.json` + checkpoint），`stream --task-spec ... --tasks D --source official --batches 4 --arm auto` 直接跑通，4 批 0 errors，首/末 arrival primary 0.863/0.727（任务本身是合成漂移，指标下降属预期，仅证明零代码可跑）。
- **E8 kernel LLM 批次协议**：A private-sim，2 个 arrival 批 + 1 个 hidden 批，controller=kernel，3 批 0 errors，wall 1431.6s，arrival 决策 promote/promote，hidden primary 0.980；说明 Agent 循环可在"每批预测→评分→揭示"的协议下工作，但单批 LLM 成本高（约 8 分钟/批），适合作为上限演示而非默认路径。
- **E0 不变性**：`run_e0_invariance.py` → PASS（3 批，打乱当前批 answers 后 decision.json / predictions.jsonl 逐字节相同）。

## 5. 复现

```bash
PYTHONPATH=. ./.venv/bin/python scripts/research/run_private_sim_matrix.py
PYTHONPATH=. ./.venv/bin/python scripts/research/run_private_sim_sensitivity.py
PYTHONPATH=. ./.venv/bin/python scripts/research/run_private_sim_gated.py
PYTHONPATH=. ./.venv/bin/python scripts/research/run_e0_invariance.py
PYTHONPATH=. ./.venv/bin/python scripts/research/summarize_private_sim.py
```

## 6. 边界与局限

- 外部 UCI 只作**开发测试床**，不进入提交训练谱系；正式提交仍严格只用官方 train。
- 隐藏批永不揭示；`predictions.jsonl` 是唯一对平台的输出契约，`decision.json` 可审计。
- 门控阈值用 `config.margin`（A 0.004）与任务方向，可能存在"太小导致频繁回滚/太大导致放行噪声"的权衡，需在真实平台上再校准。
- 官方 private B 包与 A 同域；本测试床的"未知任务"只验证协议可跑通，不声称分布一致。
- E8 的 kernel 结果受 LLM 延迟与 token 影响，单批成本高，只做 2 批演示，不代表 10 批稳定性。
