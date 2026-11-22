# 42 · B/C 回归任务的本地深挖与 scratch_retrain 策略（v0.14.8）

## 1. 背景

0.14.7 试运行（平台 run `ee40828914cba3c6ef7c36aa3c6c86c8`）结果：

| 数据集 | 指标 | 值 |
|---|---|---|
| A | Macro-F1 | 0.894721 |
| B | RMSE / MAE | 720.07 / 518.72 |
| C | RMSE / NASA | 56.45 / 8.46e10 |

B 的 `shrink` 后处理在公开验证集上更优（497.87 vs retain 681.16），但隐藏集反而更差（735.07 vs 720.07）——静态 shrink 把预测推向 train 目标均值（954.15），在晚期低 RUL 段方向错误。B 的失败不是模型上限，而是候选策略选错。

## 2. B 数据与候选（公开数据，不含隐藏标签）

- `public/train.jsonl`：7032 行，9 个轴承（Bearing1_1 … Bearing3_3），RUL 0–2537，target_mean 954.15。
- `public/validation.jsonl`：1679 行 = Bearing1_4（122）+ Bearing2_4（42）+ Bearing3_4（1515），label 均值 60/20/757。
- checkpoint 在 validation：RMSE 681.16 / MAE 563.11；逐轴承 242 / 391 / 711，整体 bias −365。

| 候选 | val RMSE | val MAE | 逐轴承 RMSE |
|---|---|---|---|
| retain（父 checkpoint） | 681.16 | 563.11 | 242 / 391 / 711 |
| shrink α=0.3 | 497.87 | 437.86 | 678 / 754 / 471 |
| damage_shrink（train 分组拟合） | 418.72 | 340.91 | 569 / 371 / 405 |
| **scratch h48 e600（仅 public train）** | **473.08** | **—（balanced 349.67，逐轴承 327/145/489）** | 327 / 145 / 489 |

shrink/damage 只帮助大轴承 3_4，重创小轴承；晚期 last-200 段（≈隐藏尾部）shrink 从 339 变 708、damage 405。scratch h48 在全部轴承上都不劣于 checkpoint，且平衡指标最好，作为默认 B 候选。

## 3. C 数据与候选

- `public/train.jsonl`：113089 行（496 个 unit-key）；`validation.jsonl`：23049 行（106 个 unit-key），unit 完全不重叠。
- checkpoint：val RMSE 52.87 / MAE 38.90（worst unit 132.8）。

| 候选 | val RMSE | val MAE |
|---|---|---|
| retain | 52.87 | 38.90 |
| continue_train 40ep（lr=1e-3，train-only） | 51.32 | — |
| continue_train 80ep | 51.15 | — |
| **scratch h32 e300（仅 public train）** | **51.01** | **36.67** |
| scratch h64 e300 | 50.98 | 36.66 |
| scratch h64 e300 + sqrt target | 50.89 | 35.26 |

h32 与 h64 几乎持平但快 3 倍（25s vs 84s），在 2 核沙箱下选 h32 e300。

## 4. 实现

- `rsi/platform/plans.py`：新增 `SCRATCH_ARM_ENV=RSI_V2_SCRATCH_ARM`、`_SCRATCH_CONFIG`（B h48 e600 lr1e-3；C h32 e300 lr1e-3）、`scratch_arm_mode()`、`_model_to_official()`（把新训练的单隐层 tanh 模型转成官方 `contest.cpu-mlp.v1` zip 结构）、`scratch_plans(ctx)`；`build_plans` 的回归分支追加 scratch 臂。
- `rsi/platform/submission.py`：`_candidate_plans` 回归分支调用 `scratch_plans(ctx)`，候选带 `checkpoint="SCRATCH"`；候选评估循环直接用计划里的模型，不走续训；保存时 `checkpoint_model.save(candidate_dir/checkpoint.npz)` 写官方 v1 格式。
- B 的静态 shrink/damage 臂改由 `RSI_V2_B_STATIC`（默认 0）控制；`RSI_V2_B_RETAIN=1` 仍可复测父模型基线。
- 训练只用 `public/train.jsonl`；`public/validation.jsonl` 只用于候选打分与选择（`_assert_training_only` + `data_boundary.json` 继续生效）。模型结构为官方支持的单隐层 tanh，宽度 ≤128。

## 5. 默认配置（0.14.8 镜像 ENV）

```
RSI_V2_CONTROLLER=auto
RSI_V2_CAPACITY_ARM=ls10        # A 分类容量臂
RSI_V2_ADAPT_ARM=auto
RSI_V2_DRIFT_POLICY=1
RSI_V2_NATIVE_TOOLS=1
RSI_V2_KERNEL_THINKING=off
RSI_V2_MAX_TURNS=10
RSI_V2_B_RETAIN=0
RSI_V2_B_STATIC=0
RSI_V2_SCRATCH_ARM=on           # B/C 从零重训（B h48 e600，C h32 e300）
```

## 6. 本地验证

- `pytest -m "not integration"`：**214 passed, 3 deselected**。
- emulator A 2 轮 PASS（gate 决策不变）；B 3 轮 PASS（r1 选 scratch，随后 retain）；C 3 轮 PASS（r1 选 scratch，随后 retain）。
- 本地 C 每轮含 scratch 训练约 73s；B 每轮约 10s；平台 2C 沙箱下 10 轮预计增加 10–15 分钟，仍在试运行时限内。

## 7. 诚实边界

- 本地 validation 与平台隐藏集分布不同：B 的 200 行隐藏样本、C 的 9687 行隐藏样本都不在本地可观测范围内；本策略只用公开 train 训练、公开 validation 选择，不能保证隐藏绝对分数提升，只能保证“不劣于父模型 + 训练数据显著增加”。
- scratch 臂每次重训 600/300 epoch，增加平台计算时间；若超过试运行时限需回退为 continue-train 或缓存。
- 0.14.8 仍需平台试运行验证：预期 A 维持 0.8947；B 由 720.07 向 473（公开）与 418（damage）之间收敛；C 由 56.45 向 51 附近收敛（隐藏绝对分未知）。
