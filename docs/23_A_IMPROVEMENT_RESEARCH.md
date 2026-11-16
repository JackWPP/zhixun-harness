# 任务 A 提升调研（UCI Gas Sensor Array Drift #224）

> 调研时间：2026-11-16。所有实验只用官方 `public/train` 拟合；`selection/report` 仅用于选型与证据，**部署模型未在验证集上训练**。外部 UCI 批次仅用于模拟平台的“批次流协议”与扩展验证（Track X），不计入提交口径。

## 0. TL;DR

1. **A 的瓶颈非常集中**：全量 validation 上 `Acetaldehyde` F1 = **0.000**（109 个样本里 108 个被判成 Ethanol）；其余 5 类 F1 已达 0.67–0.997。macro-F1 是 6 类平均，只要 Acetaldehyde 从 0 回到 0.8，macro-F1 就能从 0.714 抬到约 0.85。
2. **这个类不是训练坏了**：官方 baseline 在官方 train 上 Acetaldehyde F1=0.824，说明它学到了 batch1；到 batch2 发生了针对该类的严重漂移/混淆。官方 baseline 输出里 Acetaldehyde 的分数几乎是常数（均值 0.020），rank 大多在第 2 位，**无监督 TTA 无法把它翻上来**（见 §2 的 CORAL/EM/温度/专家分类器实验）。
3. **静态（单批）可提升空间约 +1.9% 相对**：class-weighted CE + recenter w=0.3 全量 validation **0.7279**（当前部署 0.7199）；5 种子集成 + recenter 0.7263。剩余瓶颈是 Acetaldehyde（0 分）与 Ethanol 混淆。
4. **真正的杠杆是平台的批次流协议**：官方按批下发数据、历史批带标签（会议纪要：“每批都带标签…每次会把此前全部数据一起再给”）。在 UCI batch3–10 上模拟“用已到手的历史标签继续训练 + 当前批 recenter TTA”：平均 macro-F1 **0.399（冻结）→ 0.508（仅 recenter）→ 0.775（固定配方 ep60/lr5e-4 + recenter）**，Acetaldehyde 在后几批恢复到 F1 0.6–0.99。这是 A 最值得投入的方向，也是 harness 自身该做的事。
5. 建议优先级：**P0 强化 stream 模式的历史标签自适应策略**（harness 改动，隐藏流收益最大）；**P1 把 weighted-CE 续训纳入策略库**（静态 +1–2%）；P2 集成（收益小、checkpoint 契约风险）；P3 外部数据预训练只保留为 opt-in 消融（不计分）。

## 1. 现状与瓶颈定位

官方 base（checkpoint 在全量 validation 复算，与声明值一致）：

| 类别 | 样本数 | F1 | recall | 主要流向 |
|---|---|---|---|---|
| Ethanol | 164 | 0.684 | 0.963 | Ethanol→Ethylene 5 |
| Ethylene | 334 | 0.942 | 0.904 | Ethylene→Ethanol 32 |
| Ammonia | 100 | 0.995 | 1.000 | — |
| **Acetaldehyde** | **109** | **0.000** | **0.000** | **→Ethanol 108** |
| Acetone | 532 | 0.996 | 0.992 | →Toluene 4 |
| Toluene | 5 | 0.667 | 1.000 | — |
| macro-F1 | 1244 | **0.714030** | acc 0.8786 | |

进一步解剖（`.scratch/a_research/probe2.py`）：

- 官方 base 在官方 train 上：macro-F1 0.9558，Acetaldehyde F1 **0.824** → 训练代码没有把类 3 弄丢。
- 全量 validation 上，Acetaldehyde 的预测分数均值仅 **0.0201**（各类别行都差不多），即模型对“是不是 Acetaldehyde”几乎无输入敏感性；rank-of-class3 直方图 `[0, 677, 85, 46, 344, 92]`（第 1 位=0 次，第 2 位=677 次）。
- 一个二分类专家（训练集 tanh 特征 + 平衡权重 logistic）在 validation 上对 Acetaldehyde 的分数均值 0.589，Ethanol 0.636——**目标域上按特征几乎不可分**。
- 若 Acetaldehyde F1 能到 1.0、其余不变，validation macro-F1 上限约 0.887；到 0.8 约 0.853。这就是 A 的天花板结构。

**结论**：A 静态提升必须解决 Acetaldehyde 在 batch2 的漂移；只靠官方 batch1 无解。要么平台流里拿到目标域的带标签历史数据（协议允许），要么放弃该类的提升（当前 0.72 已接近静态上限）。

## 2. 静态（单批）方法实测

全部在 `.scratch/a_research/grid.py` / `probe2.py` / `probe3.py`，结果 `grid_results.json`。选择集=selection（35% val），证据列 report（35%）与 full（1244）。

| 方法 | selection | report | full | 备注 |
|---|---|---|---|---|
| 官方 base（无后处理） | 0.7032 | 0.6864 | 0.7140 | 声明基线 |
| base + recenter w=0.3 | — | — | **0.7268** | 静态最强单参数 |
| base + recenter w=0.5（现部署） | 0.7346 | 0.6834 | 0.7199 | 报告集上反而略降 |
| class-weighted CE（β=0.5, lr5e-4, ep30） | 0.7186 | 0.6894 | 0.7244 | 未修好 Acetaldehyde |
| 上面 + recenter w=0.3 | 0.7211 | 0.6864 | **0.7279** | 静态最佳单模型 |
| + 输入增益/偏移/噪声增广 | 0.7191 | 0.6894 | 0.7246 | 无额外收益 |
| 过采样少数类 | 0.6712 | 0.6855 | 0.7040 | 反而变差 |
| 标签平滑 0.05 | 0.7171 | 0.6864 | 0.7213 | 无感 |
| h32→h64 零初始化迁移 + 续训 | — | — | 0.7242 | 加宽无显著收益 |
| CORAL（目标白化→源重着色） | — | — | 0.0952 | 被 label shift 破坏，禁用 |
| 目标批 z-map 到源统计 | — | — | 0.5221 | 严重伤少数类 |
| EM label-shift 校正 | — | — | 0.4129 | 先验估计发散，禁用 |
| 温度缩放 0.5–3.0 | — | — | 0.7140 | 排序不变，无收益 |
| 熵最小化选 recenter w | — | — | 选到 w=0（0.714） | 熵不是好选择器 |
| 5 种子 weighted-CE 集成 | 0.7191 | 0.6894 | 0.7246 | |
| 集成 + recenter w=0.5 | — | — | **0.7263** | 收益约 +0.9% 相对 |
| Acetaldehyde 专家分类器 | — | — | 0.7140 | 分数不可分，无效 |

**读法**：静态最优约 **0.727–0.728**（相对 base +1.9%），再往上必须动 Acetaldehyde，而这需要目标域标签。recenter 权重在 selection 与 report 上方向不稳定（selection 偏好 0.3–0.5，report 偏好 0），提示它是在拟合当前批统计，跨批泛化有限。

## 3. 批次流协议：真正的提升来源（UCI 模拟）

协议模拟（`.scratch/a_research/stream_sim.py` / `stream_perclass.py`）：官方 train 为初始池；UCI batch3–10 按序到达；第 k 批可训练的数据 = 官方 train + 已到达的 b3…b(k-1) 的**带标签**数据；当前批只做预测（其标签仅用于打分）。这正是平台“历史数据带标签、逐批决策”的协议。

| 批次 | 冻结 base | 仅 recenter | 固定配方 ep60/lr5e-4 + recenter | Acetaldehyde F1（自适应后） |
|---|---|---|---|---|
| b3 | 0.594 | 0.608 | 0.606 | 0.000（此批前无该类历史标签） |
| b4 | 0.489 | 0.498 | 0.757 | 0.727 |
| b5 | 0.627 | 0.579 | 0.990 | 0.989 |
| b6 | 0.323 | 0.488 | 0.633 | 0.091 |
| b7 | 0.327 | 0.499 | 0.864 | 0.598 |
| b8 | 0.266 | 0.448 | 0.895 | 0.938 |
| b9 | 0.185 | 0.547 | 0.751 | 0.613 |
| b10 | 0.381 | 0.400 | 0.706 | 0.324 |
| **均值** | **0.399** | **0.508** | **0.775** | — |

其它固定配方的均值（全 batch3–10，raw/recenter）：ep20/lr5e-4 0.729/0.745、ep60/lr5e-4 0.747/0.775、ep150/lr3e-4 0.751/0.771、ep300/lr2e-4 0.748/0.766；“用上一批胜者选配方”的诚实策略 0.756（batch 3 用默认配方）。oracle-per-batch 上界 0.780。

**关键点**：
- Acetaldehyde 只有在历史标签里出现过后才被学回来（b4 起），这说明“该类完全可由目标域标签恢复”，且不需要外部数据。
- 必须**始终保留官方 train 做 replay**；只训当前批/近期批会灾难性遗忘（另一次把官方 validation 按 8 段顺序切分的模拟显示：段内类别极偏时自适应反而伤，均值 0.481 < 冻结 0.613——所以流式策略要“官方 train + 历史批 replay”，不要纯在线）。
- 该结果对 A 的意义：**A 的隐藏流是分批的，harness 在拿到历史批标签后持续微调 + 当前批 recenter，预期能把 A 的批次平均从 ~0.4–0.5 抬到 ~0.75**（UCI 代理），远大于静态 +1.9%。

> 合规说明：这使用的是“平台已返回标签的历史批”，不是当前/未来批的标签；本地实验用 UCI 外部数据做代理，仅作 Track X 证据。若正式规则最终只允许用平台数据，本策略在评测容器内仍成立（历史批就是平台数据）。

## 4. 文献方法地图（与本任务约束对齐）

| 方法族 | 代表工作 | 是否适配 | 备注 |
|---|---|---|---|
| 基线/组件校正 | Vergara et al., *Sensors & Actuators B* 2012（UCI 224 原始论文）；MOS 漂移补偿综述 [MDPI 14(6):143](https://www.mdpi.com/2227-9040/14/6/143) | 部分 | recenter 属于其轻量近似；组件校正需参考气/片段 |
| 分类器集成 | Vergara 2012（多分类器加权集成） | 是 | 成本低；我们 5 种子集成 +0.9% 相对 |
| 域适应（TCA/CORAL/JDA/DANN/AdaBN） | 综述同上；[UMons 半导体气体传感器 AI 综述](https://orbi.umons.ac.be/bitstream/20.500.12907/56868/1/Artificial%20intelligence%20for%20signal%20processing%20in%20semiconductor%20gas%20sensors_%20Advances%2c%20comparisons%2c%20and%20future%20perspectives.pdf) | 风险高 | 我们的 CORAL 实测 0.095（label shift + 特征几何被破坏） |
| 知识蒸馏/教师模型 | [MDPI 13(1):15](https://www.mdpi.com/2227-9709/13/1/15) | 否 | 需额外教师数据/模型，且镜像禁放大模型 |
| 测试时适应（BN 统计、TENT、动态场景 TTA） | [CVPR 2023 Robust TTA](https://openaccess.thecvf.com//content/CVPR2023/papers/Yuan_Robust_Test-Time_Adaptation_in_Dynamic_Scenarios_CVPR_2023_paper.pdf) | 部分 | 我们无 BN；recenter 是手工统计适应，熵最小化实测选错 w |
| label shift / logit adjustment | Saerens EM 等 | 实测失败 | EM 先验估计发散，且会归零少数类 |
| 类别不平衡（重加权/重采样/focal） | 常用实践 | 是 | weighted-CE 静态 +1.9%，但修不了完全漂移的类 |
| 持续学习/replay | 常见 continual learning | **是（最重要）** | §3 的 0.399→0.775 就来自 replay + 历史标签 |
| 原型/最近质心 | — | 实测 0.705 | 不如 softmax |

**警示**：UCI 224 的已发表精度存在“批次-气体混淆/调参泄漏”，不要直接引用文献数字作为可达上限；本报告的“可达”都以我们自己的代理协议为准。

## 5. 建议（按优先级）

**P0（最高，harness 改动，隐藏流收益最大）——stream 模式加入“历史标签 replay 自适应”策略**
- 在 `rsi/stream/runner.py` 增加 `--adapt-history` 模式：每当新批到达，先用“官方 train + 已到达历史批”按固定强配方（F1 上 ep60/lr5e-4；也可在 2–3 个配方间用**历史批**做选择，绝不用当前批标签）继续训练，再做当前批 recenter，最后输出预测/决策；
- 保持父 checkpoint 谱系（每轮从上一父继续），保留 `checkpoint.npz` 格式；Guard 数据白名单需允许“历史批”（标记为 `history`，仍拒绝验证/隐藏原始标签进入本地开发流程）；
- 验收：Track X 四臂中 full_harness 均值从 0.47–0.55 → 0.75±（UCI b3–10 代理），并写进 docs/12/21。

**P1（低成本，静态 +1–2%）——weighted-CE 纳入策略库**
- `continue_train` 增加 `class_weight_beta`（0..2）参数；A 种子计划加 `beta=0.5, lr=5e-4, epochs=30` + recenter w=0.3；预期 selection 0.7211、full 0.7279（vs 当前 0.7199）；
- 需全量测试 + 重跑 A canonical，若 gate 通过则重生成 submission（A 小幅 +0.008 绝对）。

**P2（不建议上线）——集成/加宽**：5 种子集成 full 0.7263，与 P1 相当但 checkpoint 契约风险高（平台按单 checkpoint 约定）；加宽无收益。

**P3（仅消融）——外部数据预训练**：A 可到 0.8884，但属于 opt-in 外部数据路径，正式口径默认关闭，只作为 extended_data_validation 加分证据。

## 6. 复现

```bash
PY=./.venv/bin/python
$PY .scratch/a_research/grid.py            # 静态网格 → .scratch/a_research/grid_results.json
$PY .scratch/a_research/probe2.py          # 混淆/特征空间/TTA/加宽/专家
$PY .scratch/a_research/probe3.py          # label-shift/熵/温度/集成
$PY .scratch/a_research/stream_sim.py      # 批次流模拟 → stream_sim.json
$PY .scratch/a_research/stream_perclass.py # 逐类证据 → stream_perclass.json
```

## 7. 局限

- 本地官方 validation 只有 batch2 一个批，静态分数无法体现流式收益；隐藏流的具体分批/标签返回时序未知，P0 效果依赖平台协议（会议纪要支持“历史批带标签”）。
- UCI batch3–10 是外部代理，类别覆盖不全（部分批缺 Toluene/其他类），macro-F1 与官方隐藏流不可直接对比；b3 之前没有 Acetaldehyde 历史标签，故该类前几批无法恢复。
- 文献部分为快速综述（含子代理未产出、由 Lead 补查的部分），未做完整系统性检索；引用数字仅作方法线索。
- 未在平台实测（无 PyTRiO Key，正式评测未开放）。

## 8. 实施结果（2026-11-16，P0 + P1 落地）

### 8.1 P0：stream 历史标签自适应（已实现，主收益）

- 代码：`rsi/stream/runner.py::_adapt_on_history`；CLI `stream --adapt-history --adapt-epochs --adapt-lr --adapt-weight`。
  每批（index>0）先把父 checkpoint 在「官方 train + 之前各批带标签数据」（即 `Batch.train`）上按连续训练 30 epoch / lr 5e-4 适配，再用 `recenter(batch_mean, w=0.5)` 校准当前批；随后仍完整走 `Runner` 一轮（Guard、记忆、records、门控不变）。不改变任何 checkpoint 结构/归一化/格式。
- 同时修复 Track X 的标签泄漏：`external_a_batches` 现在只把 batch i 的验证行交给当前批评分，batch i 的 train 行在**下一批**才进入训练（meta 新增 `pending_rows`、`labels_released_after_batch`）。
- 证据（`.scratch/stream-adapt`，UCI b3–b10、holdout 0.5、exp cap 1、leak-free；8 批全部完成、0 error、wall 333.7s）：

| batch | n | frozen | recenter w.5 | **history-adapt** |
|---|---|---|---|---|
| b3 | 793 | .5946 | .6098 | .6085 |
| b4 | 80 | .4837 | .4952 | .7722 |
| b5 | 98 | .6195 | .5822 | .9925 |
| b6 | 1150 | .3209 | .4799 | .6280 |
| b7 | 1807 | .3395 | .5074 | .8669 |
| b8 | 147 | .2563 | .4097 | .8416 |
| b9 | 235 | .1868 | .5494 | .7541 |
| b10 | 1800 | .3804 | .4032 | .7227 |
| **macro mean** | — | **.3977** | **.5046** | **.7733** |
| micro | 6110 | .3797 | .4851 | **.7417** |

  与 §3 的离线模拟（0.399 → 0.508 → 0.775）几乎一致，说明改善不是切分/拟合噪声。
- 决策序列：batch0 promote、batch1 promote、batch2 retain、batch3 retain、batch4 rollback（门控对“候选 vs 已适配父模型”保守回滚，但部署的已适配模型仍强）、batch5 retain、batch6 promote、batch7 retain。
- 口径说明：Track X 只用于证明 harness 的持续迭代能力，外部数据不进入正式提交谱系；当前批标签从未在当前批预测中使用。

### 8.2 P1：weighted-CE 静态配方（已实现，但门控未采纳——负结果，如实记录）

- 代码：`Model.fit(..., class_weight=...)`（可选参数，官方训练路径默认行为不变）、`continue_train.class_weight_beta ∈ [0,2]`、`_class_weight_vector`（逆频率、均值归一）、`tests/test_a_improvements.py`。
- 实验：A 10 轮 canonical 中于 r3 尝试 `class_weight_beta=0.5, lr=5e-4, epochs=30 + recenter w=0.3`：
  - selection 0.7226 vs 父模型 0.7346 → 门控 rollback；
  - 该候选在**全量 validation** 上为 0.7279（w.3）/ 0.7289（w.4），对比部署模型 0.7199；
  - 更强变体（beta=1.0、ep60、w.4）full 0.7368，但 selection 仅 0.7224。
- 结论：selection split（435 行）与全量 validation 方向冲突，属于 split 噪声/选择偏差；按受控协议**不能用全量 validation 反向选型**（否则就是对验证集调参）。因此 P1 保留为 opt-in 能力，不进入默认计划库，也不改提交模型。若后续把 calibration+selection 合并门控（65% validation，不参与训练）或 k-fold 门控，可重新评估；当前版本不改协议、不改 submission。
- 复现：`.scratch/a-p1/rsi-20261116-222915`（19 实验、wall 627s）；全量对照脚本见本节命令。

### 8.4 官方 validation 顺序 8 批上的对照（更保守的协议）

同一套 `--adapt-history` 在 **official source**（官方 validation 按顺序切 8 批，训练池只含官方 train + 已返回的前序批标签；无外部数据）上运行（`.scratch/stream-official-adapt`，wall 304s，0 error）：

| batch | n | frozen | recenter w.5 | history-adapt |
|---|---|---|---|---|
| 0 | 155 | .9676 | .4774 | .9676 |
| 1 | 155 | .5408 | .5421 | .7037 |
| 2 | 155 | .4884 | .4781 | .4918 |
| 3 | 155 | .5998 | .5112 | .6809 |
| 4 | 155 | .3333 | .2500 | .2479 |
| 5 | 155 | 1.0000 | .4968 | 1.0000 |
| 6 | 155 | .4984 | .3091 | .4984 |
| 7 | 159 | .4984 | .4984 | .4984 |
| **macro mean** | — | **.6158** | **.4454** | **.6361** |
| micro | 1244 | .6155 | .4456 | **.6356** |

读法：官方 validation 只有 1244 行、切 8 批后每批 155 行且类别极偏，单批 macro-F1 噪声很大；自适应带来 +0.020 绝对值（+3.3% 相对），小于 UCI 代理的 +0.376。**recenter-only 在官方小批次上反而伤指标**（.4454），这与 §3 的教训一致：小批次/类别偏斜时不要只依赖批统计校准。平台上隐藏流的批次构成未知，因此我们同时保留两个证据：外部代理证明“能力上限”，官方流证明“下限不回归”。

### 8.5 默认策略与交付影响

- `stream` 模式默认启用历史自适应（entrypoint `RSI_STREAM_ADAPT=1`，可通过环境变量关闭）；只影响批次流路径，不影响单次 `run`/submission 的 checkpoint 与 predict 协议。
- 不改变 `checkpoint.npz` 结构、归一化、父谱系规则与 Guard 数据白名单（训练数据仍是平台返回的 train/历史批标签）。
- P1 未进入默认计划库；需要时可显式在计划里传 `class_weight_beta`。
