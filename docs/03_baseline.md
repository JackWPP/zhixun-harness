# 03 · 基线模型与续训协议

> 代码：`datasets/<package>/baselines/{model.py,data.py,train.py,predict.py}`
> 复现：`python3 audit/scripts/02_baseline_metrics.py`

## 1. 一句话

基线就是一个 **NumPy 手写的小 MLP + Adam**，checkpoint 里存了**恢复续训所需的全部状态**
（参数、Adam 一阶/二阶矩、steps、归一化参数、目标缩放），因此"续训"与"从头训"在数学上等价。

## 2. 结构

```
输入 x (d 维)
  → 标准化: z = clip((x − mean) / scale, −10, +10)
  → 隐层:   h = tanh(z @ w1 + b1)          (hidden 维)
  → 输出:   y = h @ w2 + b2                (1 维回归 / 6 维分类 logits)
```

| 任务 | d | hidden | 输出 | 参数量 | 激活 |
|---|---|---|---|---|---|
| A | 128 | 32 | 6 (softmax) | **4326** | tanh |
| B | 34 | 24 | 1 | **865** | tanh |
| C | 24 | 32 | 1 | **833** | tanh |

参数 = `w1(d×h) + b1(h) + w2(h×out) + b2(out)`。
`Model.__init__` 里有硬上限：**参数数必须 < 100,000**。

## 3. 训练细节（`model.py`）

```python
Model(task, paths, classes, x, y, *, hidden=32, seed=42,
      learning_rate=.003, batch_size=128)
```

- **优化器**：Adam，β1=0.9，β2=0.999，ε=1e-8，带偏差修正；`steps` 持久化，续训时偏差修正连续。
- **梯度裁剪**：`grad *= min(1.0, 5.0 / max(‖grad‖, 1e-12))`。
- **shuffle**：`rng = default_rng(SeedSequence([seed, epochs]))`
  —— 用 **epoch 计数**做子种子，所以"分两次各训 5 轮"与"一次训 10 轮"结果**逐位相同**。
  这是它能宣称"续训确定性"的真正原因（不是什么高深机制）。
- **归一化**：`mean = x.mean(0)`、`scale = x.std(0)`（`scale<1e-12 → 1`），**只在首次训练时拟合，之后冻结**。
- **回归目标**：内部按 `target_mean / target_scale` 标准化，输出时还原并 `np.maximum(0, ·)` 截断为非负。
- **线程**：`BASELINE_CPU_THREADS`（1–8，默认 1），在 import numpy **之前**设置
  `OPENBLAS/OMP/MKL_NUM_THREADS`。

各任务配方（`training.json`）：

| 任务 | lr | batch | epochs | steps |
|---|---|---|---|---|
| A | 0.003 | 128 | 20 | 80 |
| B | 0.003 | 128 | 15 | 825 |
| C | 0.002 | 256 | 15 | 6630 |

## 4. checkpoint 格式

- zip（`ZIP_STORED`），成员：参数数组 + `first/*` + `second/*` + `meta.json`。
- 固定 `date_time=(1980,1,1,0,0,0)` → **同样的权重产生同样的字节**，可做摘要比对。
- `allow_pickle=False`；**上限 16 MiB**。
- `save()` 用 `"xb"` **独占写**，不能覆盖已有文件（写 checkpoint 前必须先删/换路径）。
- `load()` 会拒绝：符号链接、超限、张量集合不符、非 float64、Adam 二阶矩为负、meta 非法。
- `meta["paths"]` 是 **list[str]**，每个元素是 JSON 字符串（如 `'[["key","features"],["index",0]]'`）。

## 5. 续训协议（`TRAINING_GUIDE.md` 的硬约束）

```
python -m baselines.train --profile B \
  --train public/train.jsonl --validation public/validation.jsonl \
  --from-baseline checkpoint.npz --output round-1 --stage-id round-1

python -m baselines.train --profile B ... \
  --resume round-1/checkpoint.npz --output round-2 --stage-id round-2
```

- `--from-baseline` 与 `--resume` **都恢复权重 + Adam 状态**，没有"重新初始化"分支。
- 归一化与目标缩放**冻结**，跨轮不变。
- 明确禁止：**"改变网络结构、输入维度或任务时，需显式实现兼容的权重迁移，不能将一个新模型声明为原模型续训"**。
  —— 这条直接限制了"换模型结构"这条创新路线。
- 明确说明：**文件名/命令行参数不是安全边界**，隔离要靠数据发布与挂载。

## 6. 推理协议（`predict.py`）

stdin 逐行 JSON：

```json
{"task": "A", "inputs": {"features": [...]}}
```
→ 分类：`{"outputs": [{"label": "Ethanol", "score": 0.97}, ...]}`
→ 回归：`{"outputs": {"value": 123.4}}`

会拒绝：多余字段、task 不符、`inputs` 展开后的路径与 `meta["paths"]` 不一致。

## 7. 已知陷阱

### 7.1 `configs/B.json` 版本漂移（真实 bug）

`datasets/cpu-a-guide-20261110-v1/baselines/configs/B.json` 与
`datasets/cpu-c-guide-20261110-v1/baselines/configs/B.json` 写的是：

```json
{"task": "classification", "classes": ["Normal", "Degrading", "SevereFault"], ...}
```

但 `TASK.md` / `baseline-info.json` / `training.json` 都说是 **regression，单位 minutes，classes 为空**。
只有 `datasets/cpu-b-rul-20261110-v1/baselines/configs/B.json` 是对的。

**影响**：任何按 `configs/B.json` 建模型的代码在任务 B 上会直接建错模型类型。
**建议**：以 `baseline-info.json` + `TASK.md` 为准，或在 `make verify` 里加一条断言。

### 7.2 A / C 的 `label_status: draft`

`baseline-info.json` 中 A、C 的 `label_status` 是 `draft`，`training.json` 中
`official_scoring_ready: false`、`purpose: "development_reference"`。
只有 B 是 `approved`。

**含义**：A / C 的公开基线分数不一定是最终评分口径，隐藏集上可能重训过。

### 7.3 基线分数可以直接复现

`audit/scripts/02_baseline_metrics.py` 用 `Model.load()` 加载 checkpoint，
在公开验证集上算出的分数与 `training.json` 的 `validation_after` **逐位一致**：

| 任务 | 记录 | 复算 |
|---|---|---|
| A | 0.7140297220788709 | 0.7140297221 ✅ |
| B | 681.1598326043113 | 681.1598326043 ✅ |
| C | 52.86857256611151 | 52.8685725661 ✅ |

所以"基线是什么"没有黑箱，可以放心在它上面做实验。
