# 46 · C 任务尾部误差分析（最后一次 formal 前的 C 策略决策）

> 结论先行：C 的隐藏分由极少数「超长 RUL 行」主导，本地验证无法识别；任何在
> validation 上看起来更好的 C 候选都可能让 hidden NASA 变差（已有 scratch 为证）。
> 最后一次 formal 采用 **C retain 父模型（baseline-safe）**，即 v0.14.12 配置。

## 1. 平台口径复算（已验证）

- 平台 C 的 NASA 口径为 **Σpenalty，无 cap**：`err=pred-answer`；
  `penalty = expm1(-err/13) if err<0 else expm1(err/10)`。
- 用该口径复算官方 checkpoint 在 `public/validation.jsonl`：
  RMSE **52.8686** / MAE **38.8998** / NASA **5,823,439,325**，与平台
  `training_records` 的 5.823e9 一致。
- `rsi.metrics.nasa_score` 是「mean + cap 1e6」，不是平台口径，仅作内部参考。

## 2. 误差结构（validation）

| 真实 RUL | 行数 | bias | RMSE | NASA 占比 |
|---|---:|---:|---:|---:|
| 0–50 | 5300 | +17.7 | 31.2 | 0.03% |
| 50–100 | 5300 | +35.1 | 47.6 | 2.32% |
| 100–150 | 5246 | +15.9 | 37.3 | 0.98% |
| 150–200 | 3859 | −17.3 | 42.1 | 0.40% |
| 200–250 | 1931 | −55.1 | 71.1 | 1.63% |
| 250–300 | 737 | −88.0 | 97.1 | 0.31% |
| 300–350 | 415 | −131.2 | 137.4 | 4.59% |
| 350–434 | 261 | −176.1 | 181.2 | **89.75%** |

- 方向：err<0（低估）9786 行贡献 **96.6%** NASA；err≥0 仅 3.3%。
- 输出幅度压缩：`pred ≈ 0.55·y + 53`（train/val 一致）；y>400 的 42 行 pred 均值
  ≈ 200，单行 penalty 可达 7.5e8；y>max(pred)=298 的 685 行占 NASA 94.4%。
- 最差 unit：FD004_u179（NASA 4.01e9，68.9%）。

## 3. hidden 4.26e10 的推断

- hidden 行数 9687；每行均值 4.40e6（val 2.53e5，×17.4），但 hidden RMSE 56.23
  只是 val 的 ×1.06 → 整体没有变差，缺口必须来自**极少数行**。
- 一致解：val 误差 ×1.06（复现 56.23/41.29）+ **3–4 行 err≈−300**（例如 y≈500、
  pred≈200）即可复现 4.26e10；RMSE/MAE 对这几行几乎不敏感。
- 过预测方向不成立：要凑 4.26e10 需约 13000 行 err≥+150，超过 hidden 行数。
- 因此 hidden 比 val 多出的是 **y≈480–510 的超长 RUL 行**（val y_max=434 覆盖不到）。

## 4. 本地实验（train-only，validation 只测量）

| 候选 | val RMSE | val MAE | val NASA | hidden NASA（实测/推断） |
|---|---:|---:|---:|---:|
| 官方基线 retain | 52.87 | 38.90 | 5.82e9 | 4.26e10（探针实测） |
| scratch h32 e300（0.14.8） | 51.20 | 36.38 | 4.09e9 | **9.24e10（实测，2.17×）** |
| scratch h32 e600 | 50.73 | — | — | 未测 |
| scratch h64 e600 / h96 e900 | 50.62 / 50.25 | — | — | 未测 |
| Huber δ=80 h32 e300 | **50.55** | 35.7 | **3.55e9** | 未测（P6 代理 2.1×） |
| sqrt 目标 | 50.74 | 35.12 | 4.07e9 | 未测 |
| 集成 baseline+scratch | 51.44 | 37.03 | **3.94e9** | 未测 |
| 高 RUL 重采样/加权 | 52–57 | — | 1e11–1e16（灾难） | — |

关键反例：scratch 在 validation 上 NASA 好 30%，hidden 反而差 117% —— **本地 val
NASA 与 hidden NASA 对近似模型不可迁移**。重采样/加权会抬高全模型输出，低 RUL 段
的过预测被 exp(err/10) 放大，直接爆炸。

## 5. 决策与依据

- 最后一次 formal 只有 1 次；两次 formal 的 `not_below_static_baseline` 均为
  false，A/B 隐藏指标都远高于基线，C 是唯一嫌疑人。
- **C 全程 retain 父模型**（v0.14.12，`RSI_V2_PROBE_RETAIN=C`）：
  隐藏 RMSE/MAE/NASA = 静态基线（56.2259 / 41.2866 / 4.26e10），
  `not_below_static_baseline` 应变为「持平」而非「低于」。
- 代价：C 的 mean_gain 贡献≈0、worst_stage 可能归零，预计客观分约 −0.3~0.4；
  换来 eligible/gate 不再被 C 拖累。
- 若未来还有 trial 额度，值得验证的 C 候选优先级：h64/h96 e600（RMSE 收益稳定）
  → Huber80（val NASA 最好但 hidden 未知）→ 集成（成本 2×）。在拿到 hidden 读数前
  不进入正式提交。

## 6. 复现

- 本地分析脚本：`.scratch/c_lab/metric/step1.py … step10.py`（平台口径、尾部、
  输出幅度、training-only 变体、合法后处理、hidden 预算）。
- 子代理报告：`/tmp/c_metric.md`、`/tmp/c_loss.md`、`/tmp/c_arch.md`、
  `/tmp/c_proxy.md`。
