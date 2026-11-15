# datasets/ — 赛题数据包（原样保存，未修改）

> 这三个目录是从 `../archives/*.zip` 解出来的**原始赛题包**，
> 除删除 `__pycache__` 外**未做任何改动**。
> 校验：`shasum -a 256 -c ../archives/SHA256SUMS`

## 三个包

| 目录 | 任务 | 主指标 | 基线 | 数据来源 |
|---|---|---|---|---|
| `cpu-a-guide-20261110-v1/` | 气体传感器漂移·六类识别 | macro-F1 | 0.7140297 | UCI Gas Sensor Array Drift (#224) |
| `cpu-b-rul-20261110-v1/` | 滚动轴承·剩余寿命（分钟） | RMSE | 681.160 | XJTU-SY |
| `cpu-c-guide-20261110-v1/` | 涡扇发动机·剩余寿命（循环） | RMSE | 52.8686 | NASA C-MAPSS FD001–FD004 |

## 包内结构（三个包一致）

```
cpu-x-guide-20261110-v1/
├── TASK.md                 任务定义（输入/输出/指标/约束）—— 权威来源
├── START.md                快速开始
├── TRAINING_GUIDE.md       训练协议详解（续训规则、安全边界）
├── training.json           训练超参 + validation_before/after + 元信息
├── baseline-info.json      基线元信息（参数数量、label_status、hash）
├── checkpoint.npz          平台基线模型（含 Adam 状态、归一化统计量）
├── history.jsonl           基线训练的逐 epoch 记录
├── public/
│   ├── train.jsonl         训练集（唯一允许拟合的数据）
│   └── validation.jsonl    验证集（只允许评估，不允许拟合）
└── baselines/
    ├── model.py            模型定义（200 行，本项目的"真相"）
    ├── data.py             数据加载与扁平化
    ├── train.py            参考训练脚本
    ├── predict.py          推理协议（stdin JSONL）
    ├── configs/            三个任务的配置 json
    ├── tests/              平台自带的测试
    ├── README.md
    └── requirements.txt
```

## 三个必须知道的坑

### 坑 1：`configs/B.json` 在 A 和 C 的包里是错的

`cpu-a-guide-*/baselines/configs/B.json` 和 `cpu-c-guide-*/baselines/configs/B.json`
写着：

```json
{"task": "classification", "classes": ["Normal", "Degrading", "SevereFault"]}
```

但同一个包里的 `TASK.md` / `baseline-info.json` / `training.json` 都说 B 是 **regression**，
单位是 **minutes**。**这是平台打包时的版本漂移 bug。**

✅ 只有 `cpu-b-rul-*/baselines/configs/B.json` 是正确的。
**判断谁是对的：以 `TASK.md` 和 `training.json` 为准。**

### 坑 2：A 和 C 是 `draft` 状态

| 包 | `label_status` | `official_scoring_ready` |
|---|---|---|
| A | `draft` | `false`（`purpose: development_reference`） |
| B | `approved` | — |
| C | `draft` | `false`（`purpose: development_reference`） |

**正式评测若更换数据版本，本仓库所有数字需重新验证。**

### 坑 3：A 的 train/val 是跨批次切分，不是随机切分

- train 只有 **batch-1** 的 445 行
- validation 只有 **batch-2** 的 1244 行

这是**故意的域偏移测试**。任何"随机切分"的实验都会严重高估性能。
同理：B 的验证集 90.2% 是 Bearing3_4；C 的验证集是按发动机随机切的（同分布）。

## 数据质量（已审计，见 `docs/05_findings.md` §5）

| 检查项 | A | B | C |
|---|---|---|---|
| NaN / Inf | 0 | 0 | 0 |
| 重复行 | 0 | 0 | 0 |
| 常量列 | 0 / 128 | 0 / 34 | 0 / 24 |
| 训练∩验证精确重叠 | 0 | 0 | 0 |
| 验证集 \|z\|>10 的行数 | 30 / 1244 | 26 / 1679 | **0 / 23049** |

**结论：数据是干净的。**"数据质量不够"这个说法不成立。

## 不要做的事

- ❌ 不要修改 `public/validation.jsonl`（它是评测基准）
- ❌ 不要用验证集标签训练（赛题明令禁止，且会毁掉所有结论）
- ❌ 不要把 `checkpoint.npz` 的结构改掉再声称"续训"
- ❌ 不要 `chmod` 改动 `../archives/` 里 zip 的权限（444 是刻意的）

## 规模

`du -sh` 结果：三个包合计约 **58 MB**，64 个文件。
其中 `public/*.jsonl` 是大头（C 的 train.jsonl 约 40 MB）。

如果仓库要公开，请注意：这三个包是**赛事方提供的**，
**对外发布前请确认授权**。见 `docs/01_competition.md`。
