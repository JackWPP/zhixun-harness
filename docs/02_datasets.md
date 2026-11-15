# 02 · 三份数据与任务定义

三个数据包结构完全一致：

```
datasets/<package>/
├── TASK.md              任务说明（人读）
├── START.md             快速开始
├── TRAINING_GUIDE.md    续训协议说明（重要，含"不得把新模型声明为原模型续训"的约束）
├── training.json        基线训练报告（含 validation_before / validation_after、摘要、参数）
├── baseline-info.json   基线元信息（label_status、parameter_count、checkpoint_sha256）
├── checkpoint.npz       平台基线模型
├── history.jsonl        基线逐 epoch 训练曲线
├── public/
│   ├── train.jsonl
│   └── validation.jsonl
└── baselines/           基线代码（model.py / data.py / train.py / predict.py / configs/ / tests/）
```

> 三个包里的 `baselines/model.py` 与 `baselines/data.py` **字节完全相同**，
> 因此 `audit/scripts/_common.py` 只加载一份。

---

## 任务 A · 气体传感器漂移下的六类识别

| 项 | 值 |
|---|---|
| 数据来源 | UCI Machine Learning Repository — Gas Sensor Array Drift Dataset (#224) |
| 输入 | `inputs.features` = 128 个 float |
| 输出 | 6 类：Ethanol / Ethylene / Ammonia / Acetaldehyde / Acetone / Toluene |
| 主指标 | **macro-F1**（辅助：accuracy、混淆矩阵） |
| 训练 | 445 行（**仅 batch 1**） |
| 验证 | 1244 行（**仅 batch 2**） |
| 基线模型 | 128 → 32 (tanh) → 6 (softmax)，**4326 参数** |
| 基线分数 | macro-F1 **0.714030**（训练集 0.955754） |
| label_status | `draft` · `official_scoring_ready: false` |

**关键陷阱：这不是随机切分。** 训练集来自第 1 批测量，验证集来自第 2 批测量，
中间隔着数月的时间漂移。所以"验证集掉分"是**设计使然**，不是过拟合。

类别分布极不均衡，且验证集里有两个坑：

| 类别 | 训练 | 验证 | 备注 |
|---|---|---|---|
| Ethanol | 90 | 164 | |
| Ethylene | 98 | 334 | |
| Ammonia | 83 | 100 | |
| **Acetaldehyde** | 30 | 109 | **F1 恒为 0**，跨批次完全不可迁移 |
| Acetone | 70 | 532 | |
| **Toluene** | 74 | **5** | 只有 5 个样本，macro-F1 量化步长约 0.008 |

**后果**：macro-F1 有 1/6 的分量来自一个只有 5 个样本的类，
任何"提升"都可能是这 5 个样本里翻面 1 个造成的。

---

## 任务 B · 滚动轴承剩余寿命预测

| 项 | 值 |
|---|---|
| 数据来源 | XJTU-SY 轴承全寿命加速退化数据集（官方完整运行至失效记录） |
| 输入 | 34 维 = 17 个特征 × 2 个通道（horizontal / vertical） |
| 输出 | `RUL_minutes` = (N − i) × 1 分钟，最后一次观测 = 0 |
| 主指标 | **RMSE（分钟）** ↓（辅助 MAE） |
| 训练 | 7032 行 / **9 个轴承** |
| 验证 | 1679 行 / **3 个轴承** |
| 基线模型 | 34 → 24 (tanh) → 1，**865 参数** |
| 基线分数 | RMSE **681.160**（MAE 563.106） |
| label_status | `approved` |

17 个特征（每通道）：rms、peak、peak_to_peak、mean_abs、kurtosis、skewness、
crest_factor、shape_factor、impulse_factor、margin_factor、spectral_centroid、
spectral_rms、spectral_kurtosis、dominant_freq、band1_energy_0_2k、
band2_energy_2k_6k、band3_energy_6k_12k。

分组分布（**这才是真正的样本量**）：

| 训练轴承 | 行数 | | 验证轴承 | 行数 |
|---|---|---|---|---|
| Bearing3_1 | 2538 | | **Bearing3_4** | **1515（90.2%）** |
| Bearing3_2 | 2496 | | Bearing1_4 | 122 |
| Bearing2_3 | 533 | | Bearing2_4 | 42 |
| Bearing2_1 | 491 | | | |
| Bearing3_3 | 371 | | | |
| Bearing1_2 | 161 | | | |
| 其余 3 个 | <160 各 | | | |

**两个致命事实**：

1. 有效样本量是 **9 个轴承**，不是 7032 行。行之间高度自相关（同一轴承相邻时刻）。
2. 验证集 **90.2% 来自 Bearing3_4 一个轴承**，所以"验证 RMSE"基本等于"Bearing3_4 上的 RMSE"。

**并且基线劣于常数对照**（见 `docs/05_findings.md` §2）。

---

## 任务 C · 涡扇发动机剩余寿命预测

| 项 | 值 |
|---|---|
| 数据来源 | NASA C-MAPSS（FD001–FD004） |
| 输入 | 24 维 = 3 个操作设置 + 21 个传感器 |
| 输出 | `RUL_cycles` = T − t |
| 主指标 | **RMSE（循环）** ↓（辅助 NASA 非对称分数） |
| 训练 | 113,089 行 / **182 台发动机** |
| 验证 | 23,049 行 / **62 台发动机** |
| 基线模型 | 24 → 32 (tanh) → 1，**833 参数** |
| 基线分数 | RMSE **52.869**（MAE 38.900） |
| label_status | `draft` · `official_scoring_ready: false` |

四个子集的行数分布（训练 / 验证）：

| 子集 | 训练 | 验证 |
|---|---|---|
| FD001 | 14,130 | 3,210 |
| FD002 | 38,034 | 7,581 |
| FD003 | 17,529 | 3,555 |
| FD004 | 43,396 | 8,703 |

**关键事实：C 的验证集是同分布随机切分，不是漂移测试。**
四个子集在训练/验证中的比例几乎相同，且验证集标准化后 |z| 最大值只有 **3.0**
（对比任务 A 的 299.6）。所以在 C 上"漂移诊断"没有对象——
harness 在 C 上只报出 `snapshot_protocol` 和 `no_causal_labels` 两个无条件结论，是正确的。

**注意**：C 是纯快照任务（`window_sizes: [1]`），没有时间窗口，
所以"用历史序列"这条路在平台给定的协议下是被关掉的。

---

## 数据质量结论（见 `audit/scripts/01_data_quality.py`）

| 检查项 | A | B | C |
|---|---|---|---|
| NaN / inf | 0 | 0 | 0 |
| 重复行 | 0 | 0 | 0 |
| 常量列 | 0 | 0 | 0 |
| 训练∩验证 精确重叠 | 0 | 0 | 0 |
| 验证集标准化 \|z\| > 10 的行 | 30 / 1244 | 26 / 1679 | **0** |

**所以"数据质量不够"在缺失/重复/噪声意义上不成立。**
真正的结构性问题在别处：独立单元太少（1 个批次 / 9 个轴承 / 182 台发动机）、
A 跨批次不可迁移、B 的标签与分组强相关。详见 `docs/05_findings.md`。
