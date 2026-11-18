# 32 · 全真模拟环境：平台协议 + PI-Agent 内核 + 真实模型网关

> 目标：在本地把「比赛平台 v2 协议 → 选手 harness → 漂移数据按轮投放 → Agent 自主实验 → 候选/状态/父产物 → 隐藏批评测」整条链路跑起来，
> 用真实模型（Qwen3.8-27B，OpenAI 兼容端点）驱动，而不是只跑确定性 pipeline；并把每轮记录写入 SwanLab。
> 本文档记录设计、场景、复现命令与实测结果。

## 1. 为什么需要这层模拟

官方平台只暴露 `workspace.v2` 的 RPC：`/data` 每轮投放当前阶段的公开投影，`/previous` 挂父产物，`/state` 跨轮，
`rnd.model(...)` 是唯一的大模型通道，`rnd.submit_candidate(...)` / `rnd.save_state(...)` / `rnd.finish(...)` 结束一轮。
平台之外的人无法看到宿主日志，trial 失败时也拿不到完整 stderr。因此我们要在本地复刻同一套协议，做到：

- **同一套选手代码**：镜像里的 `docker/app/harness.py` + `rsi/platform/submission.py`，不改一行即可跑；
- **真实的 PI-Agent 运行时**：kernel 路由通过 `rsi/platform/agent_host.py` 把 `Kernel` 的 tool-calling 循环接到宿主的 `rnd.model`；
- **真实模型网关**：本地宿主扮演平台网关，转发 OpenAI 兼容请求（PyTRiO `Qwen/Qwen3.8-27B`），返回真实 `call_id`；
- **可审计**：每轮的 hypothesis / change_summary / decision / parent sha / candidate sha / 真实 model_call_ids 全部落盘；
- **可重复**：数据按轮切片、候选计划可复算，`summary.json` 与 SwanLab 日志一一对应。

## 2. 组成

| 模块 | 作用 |
| --- | --- |
| `rsi/platform/agent_host.py` | `GatewayRuntime` 把 `rnd.model` 包装成 PI 运行时面（`arun_agent` / `collect_agent` / `Tool` 调用）；`PlatformKernelHost` 把 kernel 的 proposal/experiment 映射到平台候选计划；`AgentState` 把 memory/lessons/strategy_stats/SQLite 存入 `/state`，下一轮恢复 |
| `rsi/platform/plans.py` | 平台候选计划：retain / recenter / shrink / damage_shrink / pair_threshold / continue_train；`PlanContext` 统一特征展开与验证评估 |
| `rsi/platform/submission.py` | 训练轮入口：`train()` 默认走 pipeline 路由，`RSI_V2_CONTROLLER=kernel` 时走内核路由（失败自动回退 pipeline）；`finish()` 传真实 `model_call_ids` |
| `scripts/sim/full_env.py` | 本地宿主：start 帧、history/model/candidate/state RPC、父产物链、按轮数据投放、预测与指标、SwanLab 记录 |
| `scripts/sim/datasets.py` | S2 外部数据包（UCI gas 漂移批次）与 S3 私有 schema 包（A 权重 + 输入字段改名） |
| `scripts/sim/strategy_lab.py` | 策略实验室：同一批数据投放上对比 gen1（现有候选目录）与 gen2（漂移感知 + 新臂）的隐藏批代理指标 |

### 2.1 协议对齐（workspace.v2）

- `describe` / `train` / `predict` 三个入口与镜像完全一致；
- 一轮 = start 帧 → 读 `/data` 当前投影 → 一次或多次真实 `rnd.model` 调用 → `submit_candidate`（checkpoint + meta）→ `save_state` → `finish`；
- 每轮 `model_call_ids` 非空且都来自真实网关调用，`history_ids` ⊆ 已读历史；
- 预测端只用 `CONTEST_ARTIFACT_PATH` / `CONTEST_PREDICTION_CONTEXT` / 逐行输入，输出满足分类唯一最高分、回归非负有限；
- 宿主侧对候选做 ≤64 文件 / ≤64MiB、checkpoint_paths / code_paths 必须存在的校验。

## 3. 场景

| 场景 | 数据 | 轮数 | 说明 |
| --- | --- | --- | --- |
| S1 `official` | 官方 A/B/C 包 | 3 任务 × 3 轮 | 公开训练投影按轮增长，validation 固定；隐藏批代理 = validation 留出段 |
| S2 `external` | UCI gas 漂移批次（batch1–2 训练，3–7 按轮投放，8–10 隐藏） | 3 轮 | 模拟「赛题 A 之外的新工况」 |
| S3 `schema` | A 权重 + 输入字段改名（`probe` / `fNNNN`） | 2 轮 | 模拟「私有数据 schema 与训练时不兼容」 |

## 4. 实测结果（本机，真实 Qwen3.8-27B）

### 4.1 官方 S1（9 轮，kernel 路由，真实网关）

| 任务 | 每轮决策 | 最后一轮验证 | 隐藏批代理 |
| --- | --- | --- | --- |
| A（macro-F1） | retain | 0.7374 | **0.3330** |
| B（RMSE↓） | retain | 941.38 | **504.43** |
| C（RMSE↓） | retain | 38.97 | **42.45** |

每轮均有 6 次真实模型调用、真实 call_id、candidate 与父链 sha 校验通过；
`summary.json` 与 SwanLab run 目录见 `.scratch/sim/runs/rsi-sim-official-*/`。
结论：**协议全通，但「公开验证好」与「隐藏批好」并不同步**——尤其 A 的漂移使公开验证段与隐藏段分布差异巨大，
这正是策略迭代要解决的核心问题。

### 4.2 外部数据 S2（3 轮，kernel 路由，真实网关）

| 任务 | 每轮决策 | 最后一轮验证 | 隐藏批代理 |
| --- | --- | --- | --- |
| A（UCI gas 漂移） | retain | 0.7149 | **0.4554** |

3 轮均通过：6 次真实调用/轮、候选/父链/state 校验；运行目录
`.scratch/sim/runs/rsi-sim-external-20261118-134727/`。
说明：外部数据可作为「持续迭代测试床」，但不进入默认官方训练路径（`RSI_ALLOW_EXTENDED_DATA=0`）。

### 4.3 私有 schema S3（2 轮，kernel 路由，真实网关）

| 任务 | 每轮决策 | 最后一轮验证 | 隐藏批代理 |
| --- | --- | --- | --- |
| A（字段改名 probe/sensor_XXXX） | retain | 0.7343 | **0.1663** |

2 轮均通过协议（6 次调用/轮）。隐藏明显下降说明「私有数据 schema 与训练字段不一致」时，
只靠后处理无法救回，需要字段对齐/适配层——这正是模拟环境要提前暴露的风险。
运行目录 `.scratch/sim/runs/rsi-sim-schema-20261118-135047/`。

### 4.4 策略实验室：gen1 vs gen2（同一投放下）

gen1 = 现有候选目录（retain / recenter-mean-0.4 / pair-fit；B 的 shrink；C 的 continue-40），按「验证集最优 + margin」选择。
gen2 = 新增 recenter-std、pair-delta-history，并加入漂移感知规则：高漂移时若「历史校准的延迟标签臂」验证不劣于父模型（容差内），
优先选它，而不是什么都不做。

| 任务 | gen1 隐藏代理 | gen2 隐藏代理 | gen2 选择 |
| --- | --- | --- | --- |
| A | 0.3996 | **0.5762（+0.1766，+44%）** | pair-delta-history |
| B | 419.99 | 419.99 | shrink-0.3 |
| C | 51.52 | 51.52 | retain |

说明：A 的 gen2 收益不是靠「调参调出来的」，而是把「上一轮 campaign 里验证过、验证集上看似中性」的延迟标签经验
固化成一条可解释的选择规则；这正是「策略层自进化」的最小证据。
B/C 两代相同，说明规则没有在所有任务上乱动——这也是我们想要的：**只在有漂移证据的任务上改变策略**。

复现：

```bash
.venv/bin/python scripts/sim/strategy_lab.py --scenario official --rounds 3 --compare
```

### 4.5 平台问题复现与漂移策略接入（已生效）

平台 trial #3（0.10.4 镜像）暴露的问题：公开验证上选出的后处理在隐藏批上变差
（A 选 `pair_threshold` delta=-4.3955，平台隐藏 macro-F1 仅 0.4768；B 选 shrink 后隐藏 RMSE 735.07）。
本地模拟用同一 host 契约复现了「retain 不进化」基线，然后把 gen2 规则接回真实提交路径
（`rsi/platform/submission.py:_apply_drift_policy`）：

- 高漂移 + 分类：优先 `pair_threshold(delta=-4.0)`（历史先验，不重拟合当前验证集），
  只要其公开验证不劣于父模型 2*margin 就采用；
- 中漂移：只用 recenter；低漂移：retain；
- 规则同时写进 kernel brief 与 `catalog_text()`，agent 能看到这条策略提示。

| 场景 | 轮数 | 基线（retain） | 加漂移策略 | 变化 |
| --- | --- | --- | --- | --- |
| S1 官方 A | 3 | hidden 0.3330 | **0.4802** | +44% |
| S1 官方 A | 10（平台同构） | hidden 0.3330 | **0.4802** | +44% |
| S3 私有 schema A | 2 | hidden 0.1663 | **0.3190** | +92% |
| S2 外部 A | 3 | 0.4554 | 0.4554 | 持平（未触发） |
| S1 官方 B/C | 10 | 504.43 / 42.45 | 504.43 / 42.45 | 持平（无漂移证据不乱动） |

平台 trial #3 的 A 隐藏分是 0.4768，与本模拟 10 轮 A 的 0.4802 同一量级；
说明模拟能复现平台现象，而策略修复把「不进化」的 0.3330 拉回到平台水平。
**诚实边界**：模拟隐藏代理来自公开 validation 留出段，不是平台真实隐藏流，
所以不能声称线上一定提升；要验证必须再跑一次平台 trial（需用户确认）。

## 5. 复现命令

```bash
# 官方 9 轮（真实 provider + kernel 路由 + SwanLab online）
.venv/bin/python scripts/sim/full_env.py --scenario official --tasks A,B,C --rounds 3 \
    --provider real --controller kernel --swanlab auto

# 平台同构 30 轮（3 任务 × 10 轮）
.venv/bin/python scripts/sim/full_env.py --scenario official --tasks A,B,C --rounds 10 \
    --provider real --controller kernel --swanlab auto

# 外部数据 S2
.venv/bin/python scripts/sim/full_env.py --scenario external --rounds 3 \
    --provider real --controller kernel --swanlab auto

# 私有 schema S3
.venv/bin/python scripts/sim/full_env.py --scenario schema --rounds 2 \
    --provider real --controller kernel --swanlab auto

# 不需要真实端点时（CI / 单元测试）
.venv/bin/python scripts/sim/full_env.py --scenario official --provider fake --controller kernel --swanlab off
```

Provider 环境变量（与镜像一致）：`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`；
未设置时从 `.scratch/pytrio.env` 读取（0600，gitignore，不进入镜像）。
Qwen3.8-27B 必须带 `enable_thinking: false`，否则 27B 会把 token 全耗在 reasoning 上、`content` 为空。

## 6. SwanLab

- `--swanlab auto`：有 `SWANLAB_API_KEY` 时 `mode=online`，否则 `mode=offline`；
- 记录内容：每任务每轮的 `validation_primary` / `hidden_proxy_primary` / `model_calls` / `prompt_tokens` / `completion_tokens` / `wall_s`；
- 项目：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi>（公开项目 `zhixun-rsi`）；
- API key 只存 `.scratch/swanlab.env`（0600，gitignore，不进入镜像）；
- 已上传的 run：
  - 基线 9 轮：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi/runs/q6803l60>
  - 基线外部 S2：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi/runs/j6705h90>
  - 基线 schema S3：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi/runs/7xxhxw2y>
  - 漂移策略 9 轮：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi/runs/emdfnxxo>
  - 漂移策略 30 轮（平台同构）：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi/runs/ue3wrmap>
  - 漂移策略外部 S2：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi/runs/0phkv73l>
  - 漂移策略 schema S3：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi/runs/nvl19xzu>
- 记录内容：每任务每轮的 `validation_primary` / `hidden_proxy_primary` / `model_calls` / token 用量 / wall_s。

## 7. 边界与诚实说明

- 本模拟的「隐藏批代理」是官方公开 validation 的留出段，不是平台真正的隐藏流；它能暴露「公开验证过拟合」，但不等于线上分数；
- provider 是 PyTRiO 的 `Qwen/Qwen3.8-27B`（OpenAI 兼容）；若「轨迹流动」指硅基流动，只需改 `LLM_BASE_URL` / `LLM_MODEL`；
- 本模拟不验证镜像大小、CPU 配额、SwanLab 平台侧上传等容器外因素；
- 所有本地实验都没有上传平台、没有新建 submission、没有消耗 trial 额度。

## 8. 下一步

1. ✅ gen2 规则已接回 `submission.py` / `plans.catalog_text()`，并在三个场景中实测；
2. ✅ 真实 SwanLab key 已接入，7 个 run 已公开上传；
3. ✅ 版本 0.11.0、镜像重建并 verify PASS（380,149,760 B，
   sha256 `25c264746e2c4e95c87cefb646083e53c4311db4b4b2606061783e8ba2eec92d`，
   `RSI_V2_CONTROLLER=kernel` 已写进镜像 env）、150 项 pytest 全绿；
4. 平台 trial 需要用户确认后再上传/重跑（当前 3/10，剩 7 次）。
