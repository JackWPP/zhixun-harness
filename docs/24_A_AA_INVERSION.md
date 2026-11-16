# A 任务瓶颈新发现：Acetaldehyde 判定方向在批次 2+ 系统性反转

**日期**：2026-11-16  
**状态**：研究完成（未改提交代码）；建议下一步在 stream 协议下实现“历史标签校准的阈值后处理”。  
**口径**：仅用官方 train 拟合；官方 validation 只做一次性报告；UCI 外部批次仅用于诊断/验证，不进训练。

## 1. TL;DR

1. A 的 macro-F1 卡在 0.714，几乎完全由 **Acetaldehyde（class3）F1=0** 造成：validation 的 109 行 AA 中 108 行被判成 Ethanol。
2. 这不是普通的协变量漂移。用**在官方 train 上训练的 AA-vs-Ethanol 判别方向**评分，在 batch1（train）上 AUC=1.0，但在 **batch2（官方 validation）及 UCI batch3–10 上 AUC 全部 <0.55（多数 0.04–0.32）**——方向**系统性反转**。其余 5 类的方向在 train/val 之间相关 0.79–0.999，只有 class3 是 **−0.886**。
3. 在 validation 上用 oracle 阈值把 AA/Eth 判定方向翻转，macro-F1 从 **0.7140 → 0.8728**（AA F1 0.74、Eth 0.86），这是 A 目前发现的**最大单一杠杆**（+0.159）。
4. 该阈值不能靠无监督方法可靠恢复（两簇/分位数规则都失败），但**平台批次协议会返回每批标签**——用历史批标签校准阈值完全合规。离线模拟：官方 validation 切 8 批，历史校准使均值 **0.6132 → 0.6535**；UCI b3–10 **0.3990 → 0.4398**；UCI b6–10 **0.2963 → 0.3668**；含 AA 历史的批次单批最高 +0.32。
5. 结论：A 要突破，需要把“**类别判定方向漂移的检测与在线校准**”做成 harness 的一等能力，而不是继续加宽/加权/域适应。

## 2. 证据

### 2.1 官方 validation 的混淆矩阵（baseline checkpoint）

行=true，列=pred；pred 分布 `[298,307,101,0,528,10]`，true 分布 `[164,334,100,109,532,5]`。

| true\pred | Eth0 | Ethy1 | NH3 2 | AA 3 | Ace 4 | Tol 5 |
|---|---|---|---|---|---|---|
| Eth0 | 158 | 5 | 1 | 0 | 0 | 0 |
| Ethy1 | 32 | 302 | 0 | 0 | 0 | 0 |
| NH3 2 | 0 | 0 | 100 | 0 | 0 | 0 |
| **AA 3** | **108** | 0 | 0 | 0 | 0 | **1** |
| Ace 4 | 0 | 0 | 0 | 0 | 528 | 4 |
| Tol 5 | 0 | 0 | 0 | 0 | 0 | 5 |

### 2.2 判定方向反转（AA vs Ethanol）

评分 `s = logit_3 − logit_0`；用官方 train 训练（逻辑回归/LDA）的方向在目标批上算 AUC（>0.5 表示方向正常，<0.5 表示反转）：

| 批次 | batch1(train) | val(batch2) | b3 | b4 | b5 | b6 | b7 | b8 | b9 | b10 |
|---|---|---|---|---|---|---|---|---|---|---|
| AUC | **1.000** | **0.140** | 0.136 | 0.439 | 0.526 | 0.271 | 0.216 | 0.085 | 0.044 | 0.319 |

- 逐特征类均值差向量的相关：`corr(delta_train, delta_val)` = **−0.886**（class3 vs 0）；其他类 vs 0：Ethylene 0.999、Ammonia 0.990、Acetone 0.912、Toluene 0.793。
- 物理/数据解释（待外部确认）：Acetaldehyde 与 Ethanol 的传感器响应在 batch1 之后整体反向，可能是传感器极性/老化，也可能是数据集标签语义漂移；无论哪种，**平台 metric 用官方标签**，所以利用该关系是有效的。

### 2.3 Oracle 阈值上限（仅用于量化潜力）

规则：对 `argmax∈{0,3}` 的行，若 `s < delta` 判 AA，否则判 Eth。扫描 delta：

| delta | macro-F1 | AA F1 | Eth F1 | n_AA |
|---|---|---|---|---|
| 0（不做） | .7140 | .000 | .684 | 0 |
| −3.417 | .8479 | .703 | .784 | 184 |
| −3.947 | .8660 | .739 | .857 | 140 |
| **−3.875（最优）** | **.8728** | .74 | .86 | ~150 |
| −2.886 | .7790 | .594 | .480 | 241 |
| −1.294 | .6885 | .531 | .000 | 298 |

注意：这是 **oracle 上界**（delta 用 validation 标签选），只证明上限；不能直接作为提交配置（validation 不参与拟合）。

### 2.4 失败的尝试（重要负结果）

| 方法 | 结果 | 结论 |
|---|---|---|
| 逐段/滑窗 recenter（oracle 真边界 / W=15–200） | val macro .09–.42 | 局部重中心化破坏类别信息，**不是** gain/offset 型漂移 |
| 官方 train 内 5-fold CV 选配方/TTA | CV ~0.99，val ~0.72 | train 内部同批，**无法暴露跨批概念漂移**；不能用于此问题的选型 |
| 无监督 1D 两簇（k-means）翻转 | val .6885（Eth 归零） | 两簇切的是“全负/全正”，不是 AA 尾巴 |
| 分位数/先验比例翻转（旧实现方向写反） | 无增益 | 需要正确的方向+阈值，纯先验不够 |

### 2.5 流式协议下的历史标签校准（合规路径）

平台会议纪要明确：**每批下发时带标签（私有 B 包也有标签，无需配额），并且会把以前所有数据一起再给**。因此“用第 0..i−1 批的标签校准第 i 批的判定阈值”是协议允许的（不是使用隐藏标签，而是已返回标签）。

离线模拟（每条流按顺序，用历史标签在 `s` 上搜索最优 delta，AA 历史样本 ≥10 才启用；应用时不使用当前批标签）：

| 流 | 批数 | baseline macro 均值 | 历史校准均值 | 代表性单批 |
|---|---|---|---|---|
| 官方 validation 顺序 8 批 | 8 | .6132 | **.6535** | chunk4 .3333 → **.6558** |
| UCI batch3–10 | 8 | .3990 | **.4398** | b4 .4893→.5642、b7 .3267→.4160、b10 .3806→.4484 |
| UCI batch6–10 | 5 | .2963 | **.3668** | b7 .3899、b10 .4445 |

（这些是“只做阈值校准、其余全部冻结”的保守结果；叠加现有的 recenter/历史连续训练应更高。）

## 3. 推荐方案（按优先级）

1. **P0（推荐，合规）**：在 `rsi/stream` 中实现 `pair_threshold` 后处理：
   - 每批完成后把“本批特征 + 已返回标签”写入历史池（现有 stream 已有 history 机制）；
   - 下一批推理前，用历史池在 `s=logit_a−logit_b` 上搜索最优 delta（限制搜索范围/正则，AA 样本数不足时禁用）；
   - 与现有 recenter/damage_shrink 组合；写入 decision.json（`calibration_source="history_labels"`、`history_rows`）；
   - 在 UCI Track X 上做端到端验证（预期 macro 均值 +0.04~0.07，含 AA 批次 +0.1~0.35）。
2. **P1（研究）**：把“类别判定方向翻转”做成通用诊断：对每个类 c 检查 `corr(delta_c(train), delta_c(target_unlabeled))` 或 `pred_c=0 且 logit gap 双峰`；命中则把该类加入 `pair_threshold` 待校准列表。这可以做成 harness 的 `drift_type=label_semantics_flip` 诊断 + 实验候选。
3. **P2（提交静态路径，谨慎）**：静态 `predict.py` 拿不到隐藏标签，无法校准 delta。可选：
   - 用官方 validation 标签校准一次并固化（**validation 参与拟合**，有合规风险，不建议作为默认）；
   - 或用 UCI 外部批次校准一个保守 delta（外部数据仅 extended-validation，存在口径争议）；
   - 或只保留 stream 路径（平台评测本来就是批次流）。
4. **不改**：继续加宽/加权/域适应/局部 recenter 已被证明无法解决该瓶颈。

## 4. 给外部调研的需求（若需要）

> 如果队外能帮忙查证/给方法，优先这四件事：

1. **数据语义确认**：UCI Gas Sensor Array Drift（#224）中 class 3（Acetaldehyde）与 class 1（Ethanol）的响应在 batch2 之后是否已知存在**方向反转/标签语义变化**？是否有人报告过“batch1 与后续批次类内特征符号相反”或“部分气体传感器极性随老化反转”？UCI 原 README、数据集论文、以及 gas-sensor-drift-ml-benchmark（Zenodo 22820993）里是否有说明。
2. **方法文献**：概念漂移中“**类别条件方向反转 / label semantics drift**”（不只是 prior shift）的检测与在线校准方法；有延迟标签（delayed labels）的在线学习；test-time adaptation 在不使用目标标签时能否恢复方向信息。
3. **评测协议确认**：正式评测是否确实按批次返回标签（会议纪要说有）；如果静态预测是唯一评分入口，我们的 P0 方案需要改为“冻结校准”，请确认。
4. **baseline 对比口径**：官方静态基线（A 0.7140）在隐藏流上是否也是同一 checkpoint 直接预测？如果是，我们的流式校准优势会更明显。

## 5. 复现

```bash
PY=./.venv/bin/python
$PY scripts/research/diag_aa.py      # 混淆矩阵/AA 可分性/oracle 偏置
$PY scripts/research/inversion.py    # 各批 AA-vs-Eth AUC
$PY scripts/research/feat.py         # 逐特征方向相关性
$PY scripts/research/flip.py         # oracle 阈值上限扫描
$PY scripts/research/seg.py          # 逐段/滑窗 recenter（负结果）
$PY scripts/research/stream_thr.py   # 历史标签校准流式模拟
```

局限：官方 validation 仅 batch2 一个批；UCI 外部批与官方隐藏流分布未必一致；oracle 上限使用了 validation 标签，仅作潜力量化；历史校准模拟未接完整 Runner（尚未包含实验/门控/记忆），因此是保守下界。

## 6. 外部调研结论（2016-11-16 晚）

调研方核查了 UCI #224/#270 说明、Vergara 2012、Zenodo 22820993 基准仓库及若干复现论文，结论：

1. **没有公开记载**过 “Acetaldehyde(class 3) 对 Ethanol 在 batch2 后方向反转”或该两类标签语义互换；UCI 只记录了 Batch10.dat 末尾 120 行损坏的**文件更正**，与类别语义无关。→ 我们的观测是一个**未被公开报告的数据现象**，可以按“发现”写进技术报告，但需要自行给出可复现证据与谨慎表述（不能断言传感器物理极性翻转或标签错置）。
2. UCI 原始标签是 **1-based**（Ethanol=1、Acetaldehyde=4）；我们的接口用 0-based，class3=Acetaldehyde 与此一致。
3. 方法侧建议与本方案一致：**延迟标签校准**（上一批标签→下一批）、**滚动验证门控**、**类条件监控**；并明确“只有无标签目标批时，类别方向在原理上不可识别”（Kong 2023 部分可识别性、Ben-David 2010 不可能性定理），与我们无监督实验失败一致。
4. 评估规范：延迟标签场景应分别报告“预测当时分数”和“标签返回并更新后的下一批分数”（Grzenda 2020），不能把后者算作前者提升。
5. 基线口径：本地 `training.json` 为 `purpose=development_reference`、`official_scoring_ready=false`，`baselines/predict.py` 无任何校准逻辑，因此本地静态基线就是 checkpoint 直接 predict；平台隐藏评分是否有额外处理无法从公开材料确认。

## 7. P1 实现与端到端证据（已落地）

按调研建议实现了 `pair_threshold` 延迟标签校准：

- `rsi/inference.py`：新增 `fit_pair_threshold(x, y, model, pair=(0,3), min_left=30, min_right=10)` 与 `pair_threshold` 后处理。规则：`score = logit[a] − logit[b]`，模型判为 a/b 的行中 `score < delta` 改判 b，否则 a；可 `base` 叠加 recenter。delta 只能由**已返回的历史批标签**拟合。
- `rsi/stream/runner.py`：每批预测前用已揭示历史（累计 `X/y`）重新拟合 delta（要求 Eth≥30、AA≥10、历史校准相对基线提升≥0.02，否则禁用）；将该后处理叠加到父模型与候选模型上重新计分，再走正常门控；`decision.json` 增加 `pair_calibration: {delta, score, rows, ...}`。
- `rsi/cli.py`：`--no-pair-calibration`、`--pair 0,3`；默认开启。`tests/test_pair_threshold.py` 5 用例。

**官方 validation 顺序 8 批对照**（`--experiments-cap 1 --retention-every 0`，无外部数据）：

| 配置 | 均值 macro-F1 | 逐批 |
|---|---|---|
| pair off | 0.7269 | .981 / .948 / .481 / .577 / **.333** / 1.0 / 1.0 / .495 |
| **pair on** | **0.7680** | .981 / .948 / .481 / .577 / **.662** / 1.0 / 1.0 / .495 |

- 增益 +0.041，全部来自含 105 行 AA 的第 5 批（.333→.662），其余批次零回归；delta 从第 5 批起触发（−3.756 → −3.830），与此前离线模拟一致。
- 叠加 `--adapt-history` 时，历史连续训练已重新学到 AA 方向，pair 校准不再触发（delta=None），两者互补而非叠加：adapt 均值 0.7406（本机同协议）。
- 复现：`python -m rsi.cli stream --source official --tasks A --batches 8 --state-dir .scratch/stream-pair-frozen --experiments-cap 1 --retention-every 0 --no-memory`。

## 8. 局限与后续

1. 首批没有历史标签（delta 不可用），只能裸模型；竞赛若首批即含足够 AA 且允许先训练再预测，可先用 batches=1 的 train split 做初始校准（需按协议确认标签时序）。
2. 若隐藏流方向与公开流不同（例如始终不反转），pair 校准会因历史提升不足自动关闭，不会硬套。
3. delta 的拟合目标是 AA/Eth 二类平衡准确率；若官方 metric 对某些类有更高权重，可把目标换成加权 F1。
4. 静态 `predict.py` 无隐藏标签，不启用该机制；提交模型仍是 checkpoint + 现有 postprocess。
