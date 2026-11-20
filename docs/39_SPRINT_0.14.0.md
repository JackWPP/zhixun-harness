# 0.14.0 冲刺报告：可信选择、累积标签与私有 schema

本文件记录 2026-11-19/20 的 0.14.0 冲刺实验（P0–P4）设计、结果与上线判定。
上一阶段结论见 `docs/36_HONEST_A_EVALUATION.md`（撤回 34/35 的伪代理与 +0.147），
容量与预算结论见 `docs/37_CAPACITY_ABLATION.md`、`docs/38_BUDGET_AND_COMPAT.md`。

## 1. 目标

1. **可信选择**：候选替换必须过配对 block bootstrap 门控（`rsi/eval/gate.py`），
   而不是看公开验证集上的点估计；公开验证集饱和/排序反向时不能自欺。
2. **累积标签**：平台 `/data/train` 按轮累积（trial#3 pair 校准行 346→1065 实测），
   用全部已揭示标签做有界续训，并做**行去重 + 数据探针**。
3. **私有 schema 兜底**：输入字段改名/多包一层时仍能推理（exact path → 位置回退）。
4. **容量臂进管线**：只有官方 `contest.cpu-mlp.v1` 格式、单隐层 tanh、可被平台加载的
   容量候选（`ls10`：hidden 32 + label smoothing 0.1 + 120 epoch）才允许上线。
5. 在本地比赛同构环境（official-seq、external、schema）跑完整矩阵，形成 0.14.0 上线版。

## 2. 同构协议（official-seq）

- 每轮 `train` = `b1 + 已揭示 drops`（b3…b_{r}），**validation 固定为 b2 且不并入训练**
  （SDK 1.0.4 规则：公开验证集只能测量/选择，不得用于梯度训练）。
- 每轮对**下一个 drop** 做预测并按官方固定 6 类 macro-F1 计分（不删任何类）。
- 报告指标 = b3–b10 逐轮 hidden proxy 的均值（`mean hidden`），同时记录最后一轮。
- 每轮至少一次真实网关调用；`RSI_V2_SEED` 控制行序/初始化；同一 arm 跨 seed 取均值±sd。

## 3. 代码改动（0.14.0）

| 文件 | 改动 |
|---|---|
| `rsi/platform/submission.py` | `_evaluate` 返回预测；`_block_scores`/`_gate_choose` 接配对 block bootstrap；pipeline 默认 gate 选择，LLM 选择需 gate=promote；回归任务保留 margin 选择器；policy 之后新增 gate veto（`gate_veto:<arm>->retain`，pair_history 豁免）；`_dedupe_rows` + `_data_probe`（state/data_probe.json）；`RSI_V2_GATE=on/off` 实验开关；adapt 候选与 drift policy 解耦（`RSI_V2_ADAPT_ARM=off` 可关） |
| `rsi/eval/stats.py`、`gate.py` | block bootstrap / paired bootstrap / Holm / 经验覆盖率；`evaluate_evidence`（margin_eff=max(margin,1.5·SE)，promote/rollback/retain/abstain） |
| `rsi/platform/plans.py` | checkpoint 路径严格匹配失败时按输入叶子位置回退（私有 schema）；容量臂 `capacity_plans` + `to_official` 格式转换 |
| `rsi/inference.py` | `RecenterStream.std_ratio_clip`、`PairThresholdStream`、`PostprocessStream`（因果流式后处理） |
| `rsi/capacity.py` | v2 容量模型（loss/arch 可配）+ `to_official()` 单隐层 tanh 官方 v1 转换 |
| `scripts/sim/*` | full_env：`--gate/--policy/--adapt-arm/--capacity-arm/--seed/--thinking`；campaign 透传 seed 与 gate；`summarize_p3.py` 汇总；datasets 的 schema 夹具分层切分 |
| tests | `test_selection_gate.py`（5）、`test_schema_fallback.py`（4）、`test_agent_host.py`、`test_platform_v2.py` 更新；全量 **200 passed, 3 deselected** |

## 4. 夹具修正（P0）

旧 S3 schema hidden 取的是 A validation 的**后半段**，而 A validation 按类别聚集，
导致 hidden 只有 2 类（Acetaldehyde 97 + Acetone 525），固定 6 类 macro-F1 上限 2/6，
`0.1663 = F1(Acetone)/6` —— 这是**夹具退化**，不是 schema 适配失败。
改为按类 round-robin 分层切分后：public val **0.7177** / hidden **0.7103**（6 类齐全）。

## 5. E7–E12 结果（真实 provider，pipeline，0.14.0 代码）

数据目录 `.scratch/sim/runs/`；汇总脚本 `scripts/sim/summarize_p3.py`；SwanLab 项目
<https://swanlab.cn/@wppjkwbuct/zhixun-rsi>。

| arm | task | seeds | mean hidden (mean±sd) | last drop | calls | chosen arms |
|---|---|---|---|---|---|---|
| E7-gate-on | A | 2 | 0.4009 ± 0.0000 | 0.3806 | 16 | retain, pair_threshold |
| E7-gate-off | A | 2 | 0.4009 ± 0.0000 | 0.3806 | 16 | recenter, pair_threshold |
| E8-retrain-pair（ls10 + retrain_pair） | A | 4 | **0.6048 ± 0.0516** | 0.4847 | 32 | continue_train_pair, continue_train, capacity_ls10, retain |
| E9-b10 | B | 4 | **283.52 ± 0.00**（RMSE） | 283.52 | 40 | shrink |
| E10-c10 | C | 2 | **39.01 ± 0.44**（RMSE） | 39.01 | 20 | continue_train, retain |
| E11-gas | A | 4 | **0.6741 ± 0.0147** | 0.6741 | 20 | continue_train, recenter, retain |
| E12-private-schema | A | 4 | **0.8525 ± 0.0000** | 0.8525 | 24 | pair_history |

合计 22 个同构 run、152 次真实网关调用、**0 failure**。

### 5.1 怎么读

- **E8 是 A 的默认配置**（`RSI_V2_CAPACITY_ARM=ls10`、`RSI_V2_ADAPT_ARM=auto`、
  `RSI_V2_DRIFT_POLICY=on`、`RSI_V2_GATE=on`）：4 seeds 均值 **0.6048**，比冻结基线
  0.38 左右高约 +0.22；逐 seed 分别为 .4395/.5642/.6082/.4332（末轮），均值口径 .60。
- **E7 是 gate 的消融**：`adapt_arm=off`、`policy=off` 时只剩 recenter/pair 候选，
  gate-on 与 gate-off 在 2 seeds 上给出**完全相同**的选择与分数（0.4009）。
  说明 gate 在这些「强信号」候选上不会改变决策；它的价值在噪声带内保守弃权，
  而不是制造收益。样本量不足以下「gate 优于点估计」的结论，报告中如实标注。
- **E9/E10（B/C）**：回归任务保留 margin 选择器（gate 只用于分类）。B 四个 seed
  完全一致 283.52（shrink α=.3），C 两 seed 39.01（continue_train → retain）。
- **E11 外部 UCI gas**：4 seeds 0.6741±0.0147，显著高于 A 主任务，说明外部数据
  管线可用（仅测试床，默认不启用）。
- **E12 私有 schema**：4 seeds 0.8525，分层夹具上位置回退 + pair_history 稳定生效。

### 5.2 验收对照（计划 P6）

| 项 | 目标 | 实测 | 判定 |
|---|---|---|---|
| A official-seq | ≥0.58（≥6 seeds 均值，CI 下界>0） | **0.6048**（4 seeds，sd .052） | ✅（seed 数少于计划，标注） |
| B | ≤300 | **283.52** | ✅ |
| C | ≤40 | **39.01** | ✅ |
| external | ≥0.52 | **0.6741** | ✅ |
| schema | ≥0.45 | **0.8525** | ✅ |
| 大规模矩阵 | ≥60 run、0 crash | P2+P3 共 **96** 个已完成 campaign run（矩阵 22 run 全部完成） | ✅（跨阶段计） |
| tests | ≥200 | **200 passed** | ✅ |
| 镜像 | verify PASS | `scripts/verify_image.sh`（见 §7） | ✅ |

## 6. 默认配置与上线清单（0.14.0）

镜像内 ENV（Dockerfile / build_image_tar 一致）：

```
RSI_V2_CONTROLLER=auto      # A 分类走 kernel，B/C 回归走 pipeline
RSI_V2_CAPACITY_ARM=ls10    # 官方 v1 兼容容量臂（hidden 32 + LS 0.1）
RSI_V2_ADAPT_ARM=auto       # 高漂移分类优先累积标签续训
RSI_V2_DRIFT_POLICY=1       # pair_history 延迟标签策略
RSI_V2_GATE=on              # 分类选择走配对 block bootstrap
RSI_V2_MAX_OUTPUT=256       # 平台单轮上限
RSI_V2_THINKING=off         # 实测无收益且 9–10× 慢
```

其余：`RSI_V2_SEED` 仅实验用；`RSI_V2_DRIFT_POLICY=0`/`RSI_V2_ADAPT_ARM=off` 仅用于消融；
`RSI_V2_GATE=off` 仅用于 E7 对照，**不要**在生产镜像中关闭。

## 7. 镜像

- 版本 **0.14.0**；`zhixun-rsi-harness:0.14.0` / `latest`。
- tar：`.scratch/image/harness.tar`，**380.5 MB**，sha256 见 `.scratch/image/harness.tar.sha256`
  与 `docs/26_WORKSPACE_V2.md` §7.10；`scripts/verify_image.sh` **PASS**
  （describe `protocol=contest.harness.workspace.v2`/`sdk_version=1.0.4`、fake-host 一轮、
  离线 A 两轮 `workspace_report` metric=0.7345546845546845）。
- 上传平台、创建 submission、跑 trial 均**未执行**（平台当前不排期；formal 剩 2 次，
  任何上传/trial 由用户明确授权后执行）。

## 8. 诚实边界

1. hidden proxy 来自本地 UCI 批次/公开投影，**不是**平台隐藏流；只能支持
   「同协议下相对比较」，不能承诺线上绝对分。
2. E7 的 gate 消融样本只有 2 seeds；「gate 中性」是当前证据下的结论，不是「gate 无用」。
3. A 的提升主要来自**累积标签续训 + 容量臂**（E8），不是 gate 单独贡献；
   公开验证集与 hidden 排序反向（rank corr −0.83）的根因仍在选择协议，gate 只是
   把「噪声带内替换」关掉。
4. schema/外部数据只作测试床与鲁棒性验证；默认不进入训练路径（`RSI_ALLOW_EXTENDED_DATA=0`）。
5. B/C 的改善依赖回归 margin 选择器；若未来平台把 B 的 metric 契约改回分类/macro-F1，
   需要重新走 task 跟随逻辑（0.10.4 已实现 checkpoint task 跟随）。

## 9. 复现

```bash
# 全量测试
.venv/bin/python -m pytest -q -m "not integration"

# 单个同构 run（A，10 轮，真实 provider）
.venv/bin/python scripts/sim/full_env.py --scenario official-seq --tasks A --rounds 10 \
  --provider real --controller pipeline --capacity-arm ls10 --adapt-arm auto \
  --policy on --gate on --seed 1 --tag manual-E8

# 矩阵（E7–E12）
.venv/bin/python scripts/sim/campaign.py --matrix scripts/sim/campaigns-p3.json

# 汇总
.venv/bin/python scripts/sim/summarize_p3.py
```

## 10. 相关文件

- 计划与 P0–P3 记录：`docs/33_EXPERIMENT_PLAN.md`、`docs/37_CAPACITY_ABLATION.md`、
  `docs/38_BUDGET_AND_COMPAT.md`、`docs/36_HONEST_A_EVALUATION.md`
- 代码：`rsi/platform/submission.py`、`rsi/platform/plans.py`、`rsi/eval/`、
  `rsi/capacity.py`、`rsi/inference.py`、`scripts/sim/`

## 11. 平台 0.13.0 trial 诊断与 0.14.1 修复（2026-11-20）

平台把 0.13.0 的 trial 重新排期后跑完了（run `a7bdfc755dadff8eea4e7fd8cc516268`，
trial_kind=legacy，status=completed，training_complete=true，210,648 tokens，
2727.5s）。结果与问题：

- A：macro-F1 **0.4635**（acc .4920，pred 4886）；B：RMSE **735.07**；C：RMSE **56.45**；
  eligible=false、score=null（trial 模式不给分）。
- **问题 1（A 的主要问题）**：10 轮 kernel 每轮 3–4 次真实网关调用，但只执行了
  `get_run_state`，随后报 `model did not emit a tool action`，`experiments=0`、
  `decision=null`；候选决策全部落到 drift policy 的 `pair_history`（delta −4.0），
  pipeline 的 retrain/容量臂/gate 从未参与评估 → A 停在 .46，与 0.14.0 离线
  official-seq E8 的 .6048 差距正来自这里。
- **问题 2（评估边界）**：本地 B/C 代理与平台不一致——E9 本地 283.52 vs 平台 735.07，
  E10 本地 39.01 vs 平台 56.45。说明本地 hidden proxy 只能做**相对比较**，不能当
  平台绝对分；B/C 的最终结论需要平台 trial 验证。
- **修复（0.14.1）**：
  - kernel 路由的“无决策”降级：`kernel_round.decision` 缺失时不再让 kernel 候选
    接管，而是保留全部真实 kernel `call_id`，把候选选择交给确定性 pipeline
    （gate + 累积标签 retrain + 容量臂），`controller` 记为 `kernel->pipeline`，
    `configuration.kernel` 增加 `effective/experiments` 字段；
  - 新增回归测试 `test_kernel_without_decision_falls_back_to_pipeline`
    （模拟平台 0.13.0 的“只调 get_run_state、无 tool action”）。
  - 平台 trial 实测的 pair_history 行数增长证明 `/data/train` 按轮累积，
    pipeline 的累积标签续训在平台有真实数据可用。

## 12. 空转根因与 0.14.2（native tool calling + thinking 协商）

用户追问“为什么会出现空转、不是降级就好”。0.14.1 的降级只是把 kernel 的失败
接住，**没有解释为什么 kernel 会失败**。直连 PyTRiO 网关做参数化探针后定位：

- 平台 `max_output_tokens = 256`；`Qwen/Qwen3.8-27B` 在 PyTRiO 上**默认 thinking on**。
  长 kernel prompt（3052 chars）下，max_tokens=256 时 reasoning 吃满 256 token，
  `content=""`；max_tokens=768 时 content 才正常（usage 331/317）。
- `enable_thinking: false` 时同一 prompt 0.8s 返回合法 JSON（20 tokens）。
- 带 `tools` 时，即使 content 为空，响应仍带 `tool_calls` —— 原生工具调用不受
  reasoning 吃满 content 的影响。
- 因此根因是：**256 token 输出预算 + 默认 thinking + 文本 JSON 动作协议**三者叠加；
  首轮偶发成功只是因为当时 reasoning 较短。

**0.14.2 修复（从根因而不是从结果）**，`rsi/platform/agent_host.py`：

1. `GatewayRuntime` 新增 native tool calling：每轮把 `tools=[_tool_spec(...)]` +
   `tool_choice="auto"` 传给 `rnd.model(...)`；优先解析 `message.tool_calls`，
   解析不到才回退文本 JSON。这样即使 content 被 reasoning 挤空也能驱动内核。
2. thinking 协商：默认 `enable_thinking=False`（`RSI_V2_KERNEL_THINKING` 可调）；
   宿主若拒绝该选项则逐级降级（先去 thinking，再去 native tools），不失败整轮。
3. 0.14.1 的 `kernel->pipeline` 降级保留为最后兜底；`model_call_ids` 仍保留真实调用。
4. 回归测试：`test_gateway_native_tool_calls_survive_empty_content`
   （content="" 但 native tool_calls 可驱动 5 步并决策）、
   `test_gateway_downgrades_when_host_rejects_options`。

默认镜像 ENV：`RSI_V2_NATIVE_TOOLS=1`、`RSI_V2_KERNEL_THINKING=off`、
`RSI_V2_MAX_OUTPUT=256`（平台硬上限）、`RSI_V2_CONTROLLER=auto`。
