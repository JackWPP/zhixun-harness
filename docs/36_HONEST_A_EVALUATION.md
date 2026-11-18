# 36 · A 任务的诚实评估与更正（0.12.0）

> 本文是对 `docs/34_EXPERIMENT_REPORT.md`、`docs/35_PLATFORM_LIKE_SIM.md` 中若干结论的
> **正式更正**，并给出在“与实际比赛管线一致”的口径下重新测得的 A 任务结果。
> 结论一句话：**A 的真实杠杆是“用累积揭示的标签持续更新权重”，不是调损失函数、调结构或调 pair 阈值；
> 但公开验证集无法用来选这个更新——选择协议本身才是当前最大的风险。**

## 1. 需要撤回的结论

| 旧结论 | 状态 | 原因 |
|---|---|---|
| A hidden proxy 0.3330→0.4802（gen2 +0.147，+44%） | **撤回为“3 类伪代理上的数字”** | `scripts/sim/full_env.py` 旧版取 `validation[600:1244]` 共 644 行，仅有 Acetone/Acetaldehyde/Ammonia 三类；固定 6 类口径上限 0.5。0.4802 = (1.0+0.998+0.883)/6，是 3 类切片被 6 类平均后的数值，不是可外推的 A 增益。 |
| “0.4802 复现平台 0.4768” | **撤回** | 平台 trial#3 的 4886 行是 UCI b3–b10 的 40% 子样本（固定 6 类口径下同一策略实测 **0.4808**）；本地 644 行 3 类代理上的 0.4802 与之数值接近属于巧合。 |
| “10 轮 × 2 seeds 一致、结论稳健” | **撤回** | 旧 `full_env` 没有真实 world seed：`campaign.py` 的 seed 只进了 tag，`submission.py` 固定 seed=20261117，s1/s2 逐位相同，n_eff=1（伪重复）。 |
| B/C 在旧模拟中“分布接近、可对比” | **撤回** | 旧 lab 用 1079/22449 行、旧 full_env 用 800/1500 行，不同切片下的数字不可直接比较（同 arm B shrink 419.99 vs 283.52）。 |
| 离线 loss/结构 sweep 的“提升” | **仅作离线证据** | 未走 train→candidate→predict 管线；用户已明确要求以管线为准。 |

另外三处已修的协议缺陷：`rsi/metrics.py` 使用 observed-union 计算 macro-F1（A 缺类时比宿主固定类口径乐观 **0.067**；
现已新增 `rsi/eval/metrics.py::macro_f1_fixed`）；旧模拟子进程 `PYTHONPATH` 含仓库根，理论上可读到验证集答案；
旧 `full_env` 对 official 只做 public-train 前缀增长，不模拟“上一轮漂移批的标签进入下一轮训练”。

## 2. 忠实协议（official-seq）

`scripts/sim/full_env.py --scenario official-seq`：

- 数据包：`.scratch/sim/datasets/sim-a-official-seq-v1`（`public/train`=b1 445 行、`public/validation`=b2 1244 行、
  `stream/b03..b10.jsonl` 8 个 drop）。
- 每轮 r：训练集 = b1 + **前 r−1 个 drop 的已揭示行**；验证集固定 b2（**不并入训练**，避免用选择集拟合）；
  预测目标 = 第 r 个 drop（**从未进入训练或选择**）；指标 = 固定 6 类 macro-F1。
- 这是“公开数据里能诚实构造的最接近比赛的多轮协议”；平台 `/data` 是否把历史验证批也并入训练仍未知
  （trial#3 的 pair 校准行数 346→992→1020→1065 超过官方静态上限，暗示平台确实累积标签）。
- 平台隐藏流大概率是 UCI b3–b10（12221 行的 40% 子样本与 trial#3 prediction_count 4886 吻合），
  因此 b3–b10 作为“未来漂移”代理比旧 3 类切片合理；但它仍是外部数据代理，不是平台隐藏流本身。

## 3. 结果（A）

### 3.1 管线级（真实 Qwen3.8-27B 驱动，pipeline 路由，官方同构）

| 臂 | 8 轮 hidden（b3…b10）均值 | 说明 |
|---|---|---|
| baseline（保留父模型 + 原始后处理候选） | **0.4009 / 0.4204**（两个 world） | 冻结权重，性能随漂移单调下滑 |
| adapt（policy on + 累积标签续训） | **0.5637 / 0.5643**（两个 world） | 每轮用已揭示标签续训 20 epoch，再在 b2 上验证选择 |

→ **管线级增益约 +0.15**，且 hidden 曲线在 b6–b9 不再塌到 0.19–0.27（baseline 最低 0.185，adapt 最低 0.363）。

### 3.2 离线全量回放（oracle 上界，仅作天花板参考）

每轮用 b1..b_r 全量可见标签训练、预测 b_{r+1}：

| 方案 | hidden 均值 |
|---|---|
| frozen | 0.4224 |
| 从零重训 200 epoch | 0.7296 |
| 续训 40 epoch | 0.7338 |
| 续训 120 epoch | 0.7115 |

→ 若平台确实把 b2 及历史验证批也放进 `/data/train`，累积续训的**上界约 0.73**；本地诚实协议 0.56 与
上界的差距，主要来自“b2 标签是否可见”这一平台细节。

### 3.3 无标签流式自适应（不重训）

顺序回放 b2–b10（固定 6 类，b3–b10 均值）：none .3859 / pair_median .4538 / **recenter_std+pair .5318** /
pair_hist raw .4250。batch-mode 全批统计可达 .69，但这是“整批可见”的非因果数字，线上不可用；
因果实现受参考统计滞后限制。结论：无标签 TTA 能救一部分（+.15），但打不到有标签续训的水平。

### 3.4 损失函数 / 结构（管线外证据，默认不改）

- 离线 sweep（`scripts/sim/capacity_lab.py`，`.scratch/sim/cert/capacity.md`）：只有 h128 的 CI 下界 > 0
  （Δ+0.0151，CI [+0.0019,+0.0285]）；loss 系全部跨 0（label smoothing +0.0148、focal +0.003、class-balanced +0.0027）。
- 结构 agent 复核：无结构变体达到“≥0.015 且 CI 下界>0”；单隐层 tanh ≤128 官方 loader 可加载且预测逐位一致，
  relu 可加载但官方 forward 硬编码 tanh（预测错 28.5%），两层 npz 被拒；扩宽续训不如从零重训。
- 结论：**默认保持官方 CE + 单隐层 tanh 32**；若尝试容量，只在管线内做单隐层 tanh h64 的 A/B，且必须过预注册门。

## 4. 仍然没解决的问题（诚实边界）

1. **选择协议**：b2 上的验证分在第 3 轮后饱和（0.95+），但 hidden 继续下滑；用 b2 选“最佳后处理/续训轮数”
   会系统性选反（340+ 策略上 val↔hidden rank corr ≈ −0.83）。需要 blocked/rolling 验证、预注册、CI 门控。
2. **代理集**：A 的隐藏代理是 UCI b3–b10（外部数据），不是平台隐藏流；所有分数只能用于相对比较。
3. **world seed**：仍只有两个 world（0/1），且差分很小；不能声称统计显著。
4. **平台数据细节**：`/data/train` 是否累积历史验证批、是否按轮揭示标签，只有平台侧确认；trial 只能做合规认证，
   拿不到 benchmark 分数（主办方新规则）。

## 5. 复现

```bash
# 构建同构 A 数据包（幂等）
.venv/bin/python scripts/sim/datasets.py

# baseline vs adapt（8 轮；fake 用于冒烟，real 需要 provider）
.venv/bin/python scripts/sim/full_env.py --scenario official-seq --rounds 8 \
  --provider fake --controller pipeline --policy off --adapt-arm auto --seed 0 --tag seq-off-s0 --swanlab offline
.venv/bin/python scripts/sim/full_env.py --scenario official-seq --rounds 8 \
  --provider fake --controller pipeline --policy on --adapt-arm retrain --seed 0 --tag seq-on-s0 --swanlab offline

# 离线全量回放 / 无标签流式探针
.venv/bin/python scripts/sim/capacity_lab.py --seeds 3
.venv/bin/python scripts/research/probe_a_stream.py
```

## 6. 0.12.0 镜像与合规认证

- 镜像：`.scratch/image/harness.tar`，**380,293,120 B**，sha256
  `90e8b067d01b19fb0e5bd96add79e8e09f58f479e85b65c5a800617744e0a4de`；
  `scripts/verify_image.sh` PASS（selftest/workspace_report 与 0.11.x 一致）。
- 本轮新增内容进镜像：`rsi/eval/metrics.py`（固定类口径）、`rsi/capacity.py`（实验性，不启用）、
  `rsi/platform/submission.py` 的累积标签续训候选与 `RSI_V2_ADAPT_ARM` / `RSI_V2_SEED` 门控、
  `rsi/inference.py` 的因果后处理流（recenter/pair）、`scripts/sim/*` 实验脚本。
- 平台：image archive `23a8fef463d7b076f00435e744ed9d4a`（ready）；submission
  `c95fba98a47f0d4299e1bd1880611db0`（model Qwen/Qwen3.8-27B、source base）。
- 合规 trial：主办方新规则下 trial **只验证合规、不给分数**；首次 validate 返回
  `模型服务暂不可用，请稍后重试。`（PyTRiO 端点当时超时），由
  `.scratch/platform/certify_0.12.0.py` 自动重试 validate → 启动 trial → 轮询到终态。
- 正式评测（formal，剩 2 次）**未触发**，需用户确认。

## 7. 已落地但尚未接线的“可信选择”积木

- `rsi/eval/stats.py`：block/cluster bootstrap CI、paired bootstrap、Holm 校正、覆盖率检查。
- `rsi/eval/gate.py`：`evaluate_evidence(...)` 用“paired CI 下界 > 0 且 Δ ≥ max(margin, 1.5·SE)”
  才 promote，支持 retain / abstain / rollback；`select_arm(...)` 在多个候选间做保守选择。
- 7 个单测覆盖 promote / rollback / retain / abstain / 少块弃权 / 选择 / Holm。
- 这些模块已进仓库与测试（162 passed），但**尚未接入 `submission._candidate_plans` 的选择路径**，
  也未进入 0.12.0 镜像（0.12.0 镜像 = 本次认证所用）；下一步是把 gate 接到
  “候选→选择”的唯一入口，并配合 blocked validation（最新揭示批不得同时用于拟合与选择）。
