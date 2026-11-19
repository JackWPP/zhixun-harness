# 37 · 容量/损失/结构消融（官方数据，诚实累积协议）

> 状态：离线消融已完成；管线级 E1 矩阵在跑（见 §4）。所有结论以“同协议、配对比较”为准，
> 不使用旧的 3 类 644 行代理，不声称隐藏流泛化。

## 1. 协议

- 数据：UCI gas（A）原始 batch1..b10。`batch1`=官方公开 train（445 行），`batch2`=公开
  validation（1244 行），`b3..b10` 为按轮的“未来揭示范畴”（本地用原始 UCI 复现，非平台隐藏流）。
- 离线累积前向：第 r 轮用 `b1..b_{r-1}` 训练、预测 `b_r`，报告 `b3..b10` 的固定 6 类 macro-F1 均值；
  配对 block bootstrap（按 batch 分块）给 95% CI。
- 管线同构（E1）：official-seq 场景，`train=b1+已揭示 drops`、`validation=b2` 固定，
  每轮经 kernel/pipeline 产生 candidate、predict 下一 drop；选择只看公开验证集（不偷看 hidden）。
- 固定 6 类顺序 `["Ethanol","Ethylene","Ammonia","Acetaldehyde","Acetone","Toluene"]`，
  缺类记 0（与平台宿主口径一致）。

## 2. 损失函数（离线，3 seeds，b3–b10 均值）

| 变体 | b2 | hidden | Δ vs CE | 95% CI |
|---|---|---|---|---|
| CE（默认） | .7219 | .7487 | +.0003 | [−.0042,+.0045] |
| focal γ=1 | .7172 | .7501 | +.0017 | [−.0042,+.0077] |
| focal γ=2 | .7116 | .7514 | +.0030 | [−.0048,+.0108] |
| label smoothing .05 | .7080 | .7535 | +.0051 | [−.0093,+.0208] |
| label smoothing .10 | .6952 | .7632 | +.0148 | [−.0057,+.0368] |
| class-balanced β=.999 | .7201 | .7511 | +.0027 | [−.0055,+.0114] |
| logit-adjusted τ=1 | .7042 | .7580 | +.0095 | [−.0036,+.0242] |
| pair-aux λ=.3 | .6973 | .7427 | −.0057 | [−.0125,+.0004] |

**结论**：没有任何 loss 变体同时满足“Δ≥+.01 且 CI 下界>0”；class-balanced 仅在从零重训 200ep
的 scratch 协议下过门（+.0213 [+.0038,+.0410]），相邻 β 不复现、continue-40ep 臂也不复现。
→ **默认保持 CE**；若现场时间允许，只做管线内 `label_smoothing=.10` 单臂 A/B（预期 ≤+.015、可能为 0），
且必须走官方同构协议。

## 3. 隐藏层/结构（离线 + 兼容性实测）

| 变体 | hidden | Δ vs CE | 95% CI |
|---|---|---|---|
| h64 | .7550 | +.0066 | [−.0029,+.0160] |
| h128 | .7635 | +.0151 | [+.0019,+.0285] |
| h64+h2（两层） | .7537 | +.0053 | [−.0083,+.0181] |
| relu32 | .7403 | −.0081 | [−.0187,+.0008] |
| relu64 | .7359 | −.0125 | [−.0216,−.0030] |
| two_relu | .7389 | −.0096 | [−.0247,+.0049] |

独立结构 agent（流式协议、3 seeds）：基线 .7456 → h64 **.7563**（+.0107，CI [+.0001,+.0228]）、
h48 **.7550**（+.0094，CI [+.0040,+.0155]），宽度峰值 48–64；relu 系/两层全部 ≤0；
离线 iid 与流式排名 Spearman **−0.86**（离线选结构会选反）。

**兼容性红线**（官方 loader 实测）：
- 单隐层 tanh、任意宽度 ≤128：可加载，预测逐位一致；
- relu：能加载但官方 forward 硬编码 tanh（不一致率 28.5%）→ **禁止**；
- 两层自定义 npz：官方 loader 拒绝（`invalid checkpoint archive`）→ **禁止**；
- 从零重训同结构 > checkpoint 扩宽续训（扩宽零初始化仅 +0.0015/+0.0050）。

**结论**：结构不是主杠杆（上限约 +.01–.015，且 CI 边缘）；若采用，只选 **单隐层 tanh h64 +
contest.cpu-mlp.v1** 官方格式，且必须先在管线内验证（E1）。

## 4. 管线级容量实验（E1，official-seq A，pipeline，8 轮 × 3 seeds）

official-seq A、pipeline、8 轮、6 seeds（E1 seeds 1–3 + E1b seeds 4–6）；配对 bootstrap CI：

| 臂 | hidden 均值 ± sd | n | Δ vs cap-off（配对） | 95% CI |
|---|---|---|---|---|
| cap-off | .4906 ± .0899 | 6 | — | — |
| cap-h64 | .4483 ± .0388 | 3 | −.1026 | [−.1907, +.0523] |
| cap-ls10 | .5828 ± .1095 | 6 | **+.0923** | **[+.0110, +.1806]** |

逐 seed 配对差：+.2670 / +.0310 / 0 / +.0352 / 0 / +.2204（6/6 非负）。

**判定**：ls10 的配对 CI 下界 >0、均值 ≥+.01 → **按预定验收规则采用为 opt-in 默认**
（镜像 ENV `RSI_V2_CAPACITY_ARM=ls10`，仅分类任务生效；只输出官方 v1 格式 checkpoint）。
h64 为负收益（CI 跨 0 但点估计明显为负）→ 从默认目录移除，仅保留实验开关。
注意：ls10 提升幅度方差较大（sd .12），在 6 seeds 下只能声称“配对比较为正”，
不能声称线上必涨；B/C 回归任务不启用容量臂。

同时 E2 累积标签适配（`continue_train`）4 seeds：off .3806（确定性）、retrain **.5214**，
配对差 **+.1408** CI **[+.0557, +.2259]** → 保留默认（`RSI_V2_ADAPT_ARM=auto` + policy on）。
pair-only（.3936）远弱于 retrain → 默认优先 retrain，pair 作为可选叠加。

## 5. 复现

- 离线：`.venv/bin/python scripts/sim/capacity_lab.py --seeds 3` → `.scratch/sim/cert/capacity.{json,md}`
- 管线：`.venv/bin/python scripts/sim/campaign.py --matrix scripts/sim/campaigns-p2.json --only E1`
- 原始记录：`.scratch/sim/runs/*/summary.json`、SwanLab 项目 <https://swanlab.cn/@wppjkwbuct/zhixun-rsi>

## 6. 边界

- hidden proxy 来自公开 validation 留出/原始 UCI 后续批，不等于平台隐藏流；只做配对比较。
- 旧代理实验（644 行 3 类、+0.147、2.8×、+73%）已撤回（见 docs/36），不再引用。
- 所有实验 seed 真实进入行序与候选初始化；旧 campaign 的“伪重复”已修（`RSI_V2_SEED`）。
