# 15 · 数据与评测切分

## 15.1 官方数据包（唯一训练/选型来源）

| 任务 | 数据包 | 训练集 | 官方验证集 | 输入 | 输出 | 基线指标 |
|---|---|---|---|---|---|---|
| A 气体分类 | `datasets/cpu-a-guide-20261110-v1` | 445 | 1,244 | 128 维传感器响应 | 6 类（Ethanol/Ethylene/Ammonia/Acetaldehyde/Acetone/Toluene） | macro-F1 0.714030 |
| B 轴承 RUL | `datasets/cpu-b-rul-20261110-v1` | 7,032 | 1,679 | 34 维（统计特征 + 轴承上下文） | RUL（分钟） | RMSE 681.160 |
| C 发动机 RUL | `datasets/cpu-c-guide-20261110-v1` | 113,089 | 23,049 | 24 维传感器 | RUL（循环） | RMSE 52.869 |

- 训练集 sha256：A `c724014f…`、B `71b2663f…`、C `fcfaeaf1…`；
- 验证集 sha256：A `6be13703…`、B `23673a66…`、C `0cc747f9…`；
- 数据内容字段只有 `id / inputs / answer`；平台禁止在 inputs 中放 RUL 派生字段，
  官方包已满足，我们的切分只复制原始行、不增删字段（`rsi/splits.py::_write_subset`）；
- A/C 包内 `label_status=draft`（开发参考版），B 为 `approved`；正式评测前需以平台
  最终数据为准，系统对数据包只做 sha 校验与读取，不含硬编码分布。

## 15.2 三层互斥切分（`rsi/splits.py`）

官方验证集在每次运行开始时被确定性地切成三份互斥子集（seed=42），写入
`run_dir/<task>/splits/`：

| 子集 | 用途 | A 行数 | B 行数 | C 行数 |
|---|---|---|---|---|
| calibration | 只能用于拟合推理期校准（shrink 中心）或显式标记的训练扩展 | 374 | 504 | 6,922 |
| selection | 门控（promote/retain/rollback）唯一依据 | 435 | 588 | 8,215 |
| report | 只用于报告，绝不参与接受/拒绝 | 435 | 587 | 7,912 |

切分协议（同时写入 `manifest.json`，逐文件 sha256 可核验）：

- **A**：按类别分层行切分 30/35/35。验证集是单批次 batch2，Toluene 只有 5 条，
  分层保证每份都含 6 类（稀缺类可能只有 1–2 条，报告中如实标注方差大）。
- **B**：验证集只有 3 个轴承（`1_4/2_4/3_4`），按轴承分组后组内行切分。行级
  泄漏无法完全避免，已在实验报告中用 LOBO/按轴承校准实验给出保守结论。
- **C**：按发动机单元整组切分（`fd00x_uNNNN` 不跨组）：先取 30% 单元作
  calibration，剩余再 50/50 分 selection/report。
- 任何策略都拿不到 report 的标签；`runner.py` 只在 promote 后用 report 复算指标
  写摘要，不参与决策。

## 15.3 外部扩展验证数据（只验证，不训练）

### A · UCI Gas Sensor Array Drift（#224）

- 来源：<https://archive.ics.uci.edu/dataset/224/gas>，CC BY 4.0；
- 下载：`archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip`；
- 10 个批次共 13,910 行（原始 `.dat` 每行首列为类别 1–6，后 128 项
  `编号:数值`），由 `rsi/data.py::read_uci_gas_dat` 转换为与官方包一致的
  `{"features":[128]}`；
- 官方 A 包 public train/validation 分别对应 batch1/batch2（已逐值核对一致）；
  batch3–10 是漂移更严重的后续批次，用作**开发期 hidden 代理**；
- 探针文件 `run_dir/A/splits/probe.jsonl`（13,910 行，sha `7a080439…`），
  仅由 `worker.py` 在评估时读取，永不进入 `train`/`calibration` 路径；
- 证据：基线 raw probe macro-F1 0.4054 → 最终 0.5239（+0.1185），逐批结果见
  `docs/12_EXPERIMENTS.md` §12.6。

各批构成（类 1–6 计数）：

| batch | 行数 | 1 Ethanol | 2 Ethylene | 3 Ammonia | 4 Acetaldehyde | 5 Acetone | 6 Toluene |
|---|---|---|---|---|---|---|---|
| 1 | 445 | 90 | 98 | 83 | 30 | 70 | 74 |
| 2 | 1,244 | 164 | 334 | 100 | 109 | 532 | 5 |
| 3 | 1,586 | 365 | 490 | 216 | 240 | 275 | 0 |
| 4 | 161 | 64 | 43 | 12 | 30 | 12 | 0 |
| 5 | 197 | 28 | 40 | 20 | 46 | 63 | 0 |
| 6 | 2,300 | 514 | 574 | 110 | 29 | 606 | 467 |
| 7 | 3,613 | 649 | 662 | 360 | 744 | 630 | 568 |
| 8 | 294 | 30 | 30 | 40 | 33 | 143 | 18 |
| 9 | 470 | 61 | 55 | 100 | 75 | 78 | 101 |
| 10 | 3,600 | 600 | 600 | 600 | 600 | 600 | 600 |

> batch3–5 不含 Toluene（类 6），探针的 per-class F1 对类 6 只在含该类的批次上
> 有意义；报告中按 batch 分层给出，避免用总体均值掩盖。

### 其他已调研但未进仓库的数据（候选）

| 数据 | 用途 | 未采用原因 |
|---|---|---|
| XJTU-SY 原始全量 | B 的扩展验证 | 与官方 B 同源；体积大；官方包已含 3 个验证轴承，行级重采样证据弱 |
| NASA C-MAPSS 全量 FD001–004 | C 的扩展验证 | 官方 C 已覆盖同源；平台禁止把答案派生字段放 inputs；跨子集评测需重建特征，边际收益低 |
| PRONOSTIA/FEMTO | B 跨域验证 | 传感器/工况与官方不同，需重做特征，时间成本高于加分收益 |
| Zenodo 15681119 电子鼻长期漂移 | A 跨设备验证 | 原始时序需特征工程；与 UCI 同任务但分布不同，留作后续 |

结论：**只把 A 的 UCI 全量批次作为正式外部验证证据**，其余列为后续工作，避免
"堆数据"稀释合规性与时间预算。

## 15.4 许可证与引用

| 资源 | 许可 | 引用 |
|---|---|---|
| UCI Gas Sensor Array Drift (#224) | CC BY 4.0 | Vergara et al., *UCI ML Repository*, 2012 |
| XJTU-SY bearing | 学术使用（仓库 README 声明） | Wang et al., *IEEE Trans. Reliability*, 2020 |
| NASA C-MAPSS | 公开数据（NASA PCoE） | Saxena et al., 2008 |
| NumPy | BSD-3 | — |
| pi-agent-core-lc / pi-agent-harness-lc | MIT | zy1233（社区移植，非官方） |
| langchain-openai / openai | MIT / Apache-2.0 | — |
| SwanLab | Apache-2.0 | — |

仓库内 `extra/uci-gas/` 只包含 UCI 原始 `.dat`，不再分发官方赛事数据包之外的
受限内容；官方包来自赛事平台，随镜像打包仅用于评测运行。

## 15.5 私有/隐藏风格数据与合成漂移（v0.4）

正式评测使用隐藏测试流，本地开发需要一个"看不到标签也能自证"的通道：

1. **私有 JSONL 探针（`rsi/adapters/`）**
   - `python -m rsi.cli probe --task A --path private.jsonl --checkpoint ...`
   - 复用官方 `read_dataset` 的全部严格校验（唯一 id、数值特征、禁止
     `rul/true_rul/total_snapshots` 等答案派生字段、非负 RUL）；
   - 输出整体 primary + 按 group 分解的指标 + `training_isolation` 声明；
   - 数据只进入推理，不进入 executor：Guard 的训练白名单只有
     `train` / `train+calibration`，路径参数不可能被当作训练集。
2. **合成漂移（`python -m rsi.cli synth`）**
   - 从官方 train 出发，按维施加"增益 + 偏移 + 噪声"，逐批增强；
   - 固定 seed，输出标准 JSONL（id 形如 `synbatch2_r00007_xjtu...`），
     `answer` 保留，`meta.synthetic=true`；
   - 默认 drift=2.5 / noise=0.25，对应 severe 漂移（baseline macro-F1≈0.72）；
   - 用途：回归测试 harness 的漂移检测与恢复能力，不参与选型、不进训练。
3. **外部独立验证（`scripts/external_validation.py`）**
   - UCI Gas Drift batch1–10 逐批报告（0.4252→0.5281 微加权）；
   - B/C 的 XJTU-SY / C-MAPSS 全量目录接口保留，开发机可选，镜像不含；
   - 所有外部数据在 `manifest`/报告中标记 `used_for_training=false`。

> 平台正式评测时，模型只通过 `predict.py` 读隐藏集 inputs；训练/选择阶段
> 从未接触隐藏标签，也不需要接触。

## 5. 外部预训练数据（任务 A，opt-in）

- 文件：`extra/uci-gas/Dataset/batch3..7.dat`（随镜像打包，仓库内可见）。
- 用途：仅作为任务 A 的**外部预训练**输入，`extended_pretrain` 策略先在其上
  训练少量 epoch，再回到官方 `public/train.jsonl` 微调；不读取官方 val/hidden 标签。
- 开关：`RSI_ALLOW_EXTENDED_DATA=1`（v0.5 正式口径**默认关**，仅用于消融/扩展验证演示）；未开启时策略目录
  中的扩展计划会被 Guard 拒绝并回退到合法候选。
- 隔离证明：`rsi/policy.py` 只允许 `train`/`extended` 两种训练数据；校验/隐藏
  split 路径不进 executor；`workspace_report.json` 与 `docs/12` 记录实际用量。
- 证据：官方 selection 0.7346→0.8645；外部逐批（见 docs/evidence/uci_gas_drift.md）
  micro 0.4253→0.7515，其中**未参与预训练的 batch8–10** 加权 0.352→0.593。
- 披露：UCI Gas Sensor Array Drift 数据集为公开学术数据（CC BY 4.0 兼容引用），
  不是官方比赛数据；报告中明确标注 `used_for_training=true (pretrain only)`。
