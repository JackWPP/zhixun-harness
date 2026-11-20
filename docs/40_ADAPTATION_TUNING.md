# 40 · PyTRiO 适配与策略调优实验报告（0.14.3）

> 目标：回答“还需不需要调优”的问题——对 PyTRiO 网关、kernel prompt/工具/轮次/skill/上下文、
> native tool-calling 与 thinking 做全量同构实验，并把结论落到默认配置与安全网代码里。
> 协议：official-seq（A：train=b1+已揭示 drops、validation=b2 固定、逐轮预测下一 drop、固定 6 类）；
> B/C 用 official（10 轮同构）；真实 PyTRiO `Qwen/Qwen3.8-27B`；SwanLab online 记录。

## 1. 先发现的问题：kernel “空转”不是模型不听话

0.13.0 平台 trial 与本地同构里，kernel 10 轮每轮 3–4 次真实调用，但 trace 只有
`get_run_state`，随后 `error="model did not emit a tool action"`，experiments=0，决策全部落到
drift policy 的 `pair_history`。

根因由探针拆成三层：
1. PyTRiO 的 `Qwen/Qwen3.8-27B` 默认 thinking=on；平台 `max_output_tokens=256`，reasoning
   吃满 256 → `content=""`，文本 JSON 动作协议解析失败；
2. 打开 native tool-calling 后 `tool_calls` 仍能返回，但 kernel 把 10 轮预算花在
   `search_knowledge` / 重复 `propose_experiment` / `get_context`，**轮次耗尽**而不是模型拒答；
3. 报错文案“no tool action”具有误导性（实际是 turn budget exhausted），把问题指向了模型。

## 2. 修改清单（0.14.2 → 0.14.3）

| 改动 | 文件 | 目的 |
|---|---|---|
| native tool-calling + thinking 协商 + 逐级降级 | `rsi/platform/agent_host.py` | content 被 reasoning 挤空时仍能驱动 kernel；宿主拒绝选项不失败整轮 |
| prompt 效率铁律 + 本轮工具预算 | `rsi/agent/prompts_kernel.py` | 信息类工具 ≤2 次，propose 后立刻 run，最后两次必须 decide+finish |
| `RSI_V2_KERNEL_TOOLS=core/full` 工具子集 | `rsi/agent/actions.py` | 减少每轮 prompt/token，保留核心闭环 |
| kernel 默认 `max_turns` 6→10 | `rsi/platform/submission.py`、`agent_host.py` | 给真实决策留预算 |
| **确定性安全网** | `rsi/platform/submission.py`、`rsi/platform/plans.py` | kernel 可能漏掉的累积标签 retrain / 容量臂，每轮自动评估，gate promote 才覆盖 |
| `plans.evaluate` 返回 predicted | `rsi/platform/plans.py` | 让安全网候选可做分块 bootstrap |
| `continue_train_pair` 独立 kind | `rsi/platform/plans.py` | 修复 pair 变体被去重吞掉 |
| campaign 透传 controller/kernel 参数 | `scripts/sim/campaign.py` | 修正“伪重复”与臂参数未生效 |

安全网设计要点：候选集为 `retain + continue_train + continue_train_pair (+ 容量臂)`，
用同一套 paired block-bootstrap gate（`rsi/eval/gate.py`）与父模型比较；只有 `promote`
才覆盖 agent 的决策，因此不会把 agent 变成“点估计选择器”，只是兜住漏掉的高价值臂。

## 3. K1：prompt / 工具 / 轮次 / native / 上下文消融（A official-seq，10 轮，seed 0）

| arm | 实际轮数 | hidden 均值 | 末轮 | 真实调用 | wall | prompt tok | 决策 |
|---|---|---|---|---|---|---|---|
| base（full, native, t10） | 8 | .4250 | .4468 | 63 | 367s | 373,226 | 全 pair_history |
| text protocol（关 native） | 8 | .4250 | .4468 | 60 | **139s** | **204,873** | 全 pair_history |
| core tools | 8 | .4250 | .4468 | 59 | 315s | 255,321 | 全 pair_history |
| max_turns=6 | 8 | **.6061** | **.7079** | 49 | 179s | 277,084 | r6 起 continue_train |
| minctx（core+minimal skills） | 8 | .4898 | .4503 | 57 | 201s | 252,387 | pair_history |
| thinking=on | — | 不可用 | — | — | 超时 | — | 256 输出预算下不可用 |

读法：
- native 与 text 决策相同；text 更快、token 更少。保留 native 是为了对抗“宿主忽略 thinking
  开关、content 被挤空”的环境，而不是为了分数。
- minctx 明显更差 → **skills/knowledge 上下文值得保留**，不要为了省 token 砍掉。
- t6 的 .61 不是“6 轮更好”的普适结论，而是“更早被迫进入 retrain”的副作用；这暴露了
  agent 会过度探索、错过累积标签臂。安全网就是把这个副作用变成确定性策略。

## 4. K3 / K3c：安全网 + 容量臂的 6-seed 验证（A official-seq，10 轮）

| 配置 | seeds | hidden 均值 | 逐 seed |
|---|---|---|---|
| K3 安全网，capacity off | 5 | .5871 | .6535/.5638/.5715/.5717/.5752 |
| K3c 安全网 + capacity ls10 | 6 | **.6278** | .6656/.5759/.6625/.6603/.6609/.5418 |

对比基线：同构 pipeline E8（capacity ls10 + retrain_pair，4 seeds）均值 **.6048**；
K1 无安全网 kernel 均值 .4250。结论：安全网把 kernel 从“只会 pair_history”拉回 .59–.63，
容量臂 ls10 再补约 +.04；6 seeds 均值 .6278，满足 ≥.58 的验收线，但 2/6 seeds 在
.54–.58（后段 b8–b10 drop 难度差异），必须如实标注方差。

## 5. K2b：B/C 全量对比（official，10 轮，seed 0）

| task | pipeline | kernel(core) | 结论 |
|---|---|---|---|
| B RMSE | **283.52** | 414.73 | 回归任务 kernel 决策更差 → 保持 pipeline |
| C RMSE | 38.72 | **37.54** | kernel 略好但优势小、耗时 8× |

默认 `RSI_V2_CONTROLLER=auto`（分类→kernel、回归→pipeline）与数据一致，维持不变。

## 6. 默认配置结论（进入 0.14.3 镜像）

```
RSI_V2_CONTROLLER=auto        # A 分类走 kernel，B/C 回归走 pipeline
RSI_V2_CAPACITY_ARM=ls10      # 分类容量臂（单隐层 tanh，官方 v1）
RSI_V2_ADAPT_ARM=auto
RSI_V2_DRIFT_POLICY=1
RSI_V2_GATE=on                # 安全网/选择共用 paired block bootstrap
RSI_V2_MAX_TURNS=10
RSI_V2_NATIVE_TOOLS=1
RSI_V2_KERNEL_THINKING=off    # 256 输出预算下 thinking 不可用
RSI_V2_KERNEL_TOOLS=full      # 保留 skills/knowledge 能力；core 仅供省 token 实验
RSI_V2_MAX_OUTPUT=256
```

## 7. 诚实边界

1. 代理集来自公开 validation 的留出 drop，不是平台隐藏流；只能做相对比较，绝对分不可外推。
2. 安全网提升与容量臂提升都依赖“平台按轮揭示历史标签”这一观测（trial#3 校准行数
   346→992→1020→1065）；若真实平台不揭示，安全网的 retrain 臂会退化为 retain，不会掉点。
3. 6 seeds 中 2 个偏低，均值达标但方差大；不得宣称“稳定 >.6”。
4. B/C 的 kernel 对比只有 seed 0 一次，且代理切片与平台不同，仅用于 controller 路由判断。
5. K1 的 t6 异常是行为副作用，不作为“6 轮优于 10 轮”的结论。

## 8. 复现

```bash
# K1 矩阵（prompt/tool/native/turns/skill 消融）
.venv/bin/python scripts/sim/campaign.py --matrix scripts/sim/campaigns-adapt.json --timeout 1800
# 安全网 + 容量臂 6 seeds
for s in 0 1 2 3 4 5; do
  .venv/bin/python scripts/sim/full_env.py --scenario official-seq --tasks A --rounds 10 \
    --provider real --controller kernel --policy on --gate on --tag K3c-ls10-s$s --seed $s \
    --swanlab auto --hidden-every-round --capacity-arm ls10
done
# B/C controller 对比
.venv/bin/python scripts/sim/full_env.py --scenario official --tasks B,C --rounds 10 \
  --provider real --controller pipeline --policy on --gate on --tag K2b-pipeline --swanlab auto
.venv/bin/python scripts/sim/full_env.py --scenario official --tasks B,C --rounds 10 \
  --provider real --controller kernel --policy on --gate on --tag K2b-kernel-core \
  --kernel-tools core --swanlab auto
```

SwanLab：项目 `zhixun-rsi`（online）；汇总表由 `scripts/sim/summarize_adapt.py K1` 与
`.scratch/sim/runs/*/summary.json` 生成。测试：`207 passed, 3 deselected`。

## 6. 0.14.4：陈旧验证集问题与 recent-block holdout（A 从双峰 .60 到稳定 .72）

### 6.1 问题
0.14.3 的 A official-seq 结果呈双峰：好 seed 每轮均值 ~.66，坏 seed ~.54。
诊断发现：一旦后续轮次用累积揭示的漂移批训练，**冻结的公开验证集 b2 已不再代表下一批**。
继续用 b2 做选择时，retrain 的 b2 分数会从 .968 掉到 .94–.95，于是 gate/policy 拒绝它，
模型被冻结，后续 drop 的 hidden 直接掉到 .29–.44。

### 6.2 改法（能力级，不是调参）
- `rsi/platform/submission.py` 新增 **recent-block holdout**：训练行 ≥800 且漂移为 high 时，
  把最新 25%（200–2000 行）揭示批留作 holdout，用更早的行训练候选，在该新鲜块上做配对
  block bootstrap；promote 才覆盖内核决策；若无显著 promote 但 retrain 在该块上
  **non-inferior**（≥ parent − margin），也允许 retrain（冻结的代价远大于一次小退步）。
- `_drift_brief` 增加 **arrival drift**：用最新揭示块相对模型参考分布计算漂移，
  避免“train 与 val 的聚合统计”在累积后把漂移洗掉。
- 内核仍是决策主体：holdout 只提供无偏证据；确定性安全网只在没有决策/gate 否决时兜底。

### 6.3 结果（A official-seq，真实 PyTRiO，kernel，10 轮，11 seeds）
| seed | hidden 逐轮均值 | 末轮 |
|---|---|---|
| 0 | .7081 | .7008 |
| 1 | .7053 | .6956 |
| 2 | .7198 | .7125 |
| 3 | .7203 | .6994 |
| 4 | .7508 | .7164 |
| 5 | .7120 | .7011 |
| 6 | .7242 | .7054 |
| 7 | .7055 | .7149 |
| 8 | .7219 | .7004 |
| 9 | .7229 | .7217 |
| 10 | .7206 | .7017 |
- **均值 .7192，sd .0121，最小 .7053，最大 .7508**（对比 0.14.3 的 6 seeds 均值 .6278、sd≈.05）。
- 早期轮次仍保留 pair/adaptive，r5–r6 起 holdout 触发累计 retrain，后续批稳定在 .68–.72。

### 6.4 B/C 多 seed（official，10 轮，pipeline vs kernel+回归 guardrail，3 seeds）
| 任务 | pipeline | kernel+guardrail |
|---|---|---|
| B RMSE | 283.52 / 283.52 / 283.52 | 283.52 / 283.52 / 283.52 |
| C RMSE | 38.72 / 38.71 / 39.32 | 38.00 / 38.04 / 38.88 |
- B 上 agent 路由与 pipeline 打平（回归 guardrail 阻止了越训越差）；C 上略好（均值 38.30 vs 38.92）。
- 结论：`auto` 路由（分类 kernel、回归 pipeline）仍成立；回归 guardrail 已让 kernel 路径安全可用。

### 6.5 边界
- 仍是本地同构代理（official-seq 用公开数据按批揭示）；平台隐藏流不同，绝对值不可外推。
- 11 seeds 均值 .719 是“公开数据代理”的估计，不是平台分数；平台 trial 仍排队。
- 改动包含确定性 holdout/gate，agent 仍是提假设、选实验、写记忆的主体；安全网只否决/兜底。
