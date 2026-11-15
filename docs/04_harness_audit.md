# 04 · Harness 源码审计：它是什么，不是什么

> 审计对象：`harness/`（13 个文件，原型 v0.2，依赖仅 numpy）
> 复现：`python3 audit/scripts/08_demo_forensics.py`、`make test`

## 0. 一句话结论

> **这是一个写得相当干净的"确定性策略路由器 + 受控续训器"，
> 但它不是 Agent，不是 RSI，也没有在任何一次运行里让"小模型"做过一次决策。**

它做的事，本质上是：

```
读诊断（4 个阈值 + 文本模板）
  → 从 6 个写死的配方里按 if/else 挑 ≤3 个
  → 每个候选都从同一个父 checkpoint 续训 5 个 epoch（lr = 父 lr × 0.3）
  → 用验证集主指标做门控，gain > 0 就接受
  → 最多两轮；挑不出更好的就 break
```

全部逻辑可在 10 分钟内读完。**没有一行代码涉及"理解"、"推理"或"生成"。**

---

## 1. 架构与模块

| 文件 | 行数 | 职责 |
|---|---|---|
| `strategies.py` | 485 | 6 个配方定义 + 规则路由 + 策略小模型（MLP 12→24→6） |
| `training.py` | 627 | 受控续训内核：`execute_epoch` / `execute_candidate` + 诊断探针 + 经验池 |
| `runner.py` | 303 | 两轮控制循环 + 门控 `select_candidate` + 摘要输出 |
| `diagnostics.py` | 160 | 阈值诊断 + 指标计算 + NASA 分数 |
| `data.py` | ~250 | 压缩包导入、摘要校验、train/validation 角色隔离 |
| `reporting.py` | ~200 | 静态 HTML 报告 |
| `tests/` | — | 30 项单元测试 |

数据流：

```
archives/*.zip
   │ prepare_bundle()  ← 摘要校验 + 只解必要成员 + 角色隔离
   ▼
Bundle(train, validation, checkpoint)
   │ diagnose()        ← 只读，不拟合验证集
   ▼
Diagnosis(findings[])  ──rule_select_candidates()──►  [Strategy × ≤3]
   │                                                     │
   │                    execute_candidate()  ×N  ◄───────┘
   │                    （逐 epoch 验证、早停、恢复最佳 epoch）
   ▼
select_candidate()  ── gain > min_improvement ? ──► 接受 / 保留父模型
   │
   └─► 下一轮（重新诊断）或 break
```

---

## 2. 它不是什么 —— 逐条证据

### 2.1 没有 LLM、没有 Agent、没有网络

```bash
grep -rniE "requests|urllib|http|socket|openai|anthropic|llm|gpt|prompt|subprocess|exec\(|eval\(" \
     harness/ --include=*.py
```

**只有两处命中**，都不是功能代码：

- `harness/reporting.py:154` —— HTML 的 CSP `<meta>` 标签；
- `harness/reporting.py:166` —— 一句自我声明：`"策略与诊断基于规则，不使用 LLM。"`

没有 prompt、没有 tool-calling、没有代码生成、没有基于模型输出的分支。

### 2.2 动作空间是 6 个写死的配方

```python
# harness/strategies.py:20
STRATEGY_IDS = ["conservative", "class_weighted", "robust_noise",
                "device_balanced", "regularized", "penalize_late_rul"]
```

每个配方的超参在 `build_strategy_by_id()`（`harness/strategies.py:111`）里**硬编码**：

| 配方 | lr | l2 | noise | 其他 |
|---|---|---|---|---|
| conservative | 0.3×父 | 0 | 0 | — |
| class_weighted | 0.3×父 | 0.001 | 0 | `balanced_classes=True` |
| robust_noise (A) | 0.3×父 | 0.003 | 0.04 | `balanced_classes=True` |
| robust_noise (B/C) | 0.3×父 | 0.001 | 0.02 | — |
| device_balanced | 0.3×父 | 0.002 | 0 | `balanced_groups=True` |
| regularized | 0.3×父 | 0.003 | 0 | — |
| penalize_late_rul | 0.3×父 | 0.002 | 0 | `asymmetric_rul_alpha=0.25` |

**学习率永远是父模型的 0.3 倍**（`harness/strategies.py:212`：

```python
low = max(float(model.meta["learning_rate"]) * 0.3, 1e-6)
```

），从不根据诊断结果推导。所以"自适应"只体现在**选哪个盒子**，不体现在**盒子里的数字**。

### 2.3 控制循环是 for-loop，卡住就 break

```python
# harness/runner.py:247
if exhausted or winner is None:
    break
```

没有逃逸分支、没有"换个角度重新分析"、没有扩大搜索、没有回溯。
所谓"两轮"就是 `for round_index in range(1, rounds+1)`。

### 2.4 诊断是 4 个魔法阈值 + 文本模板

`harness/diagnostics.py:81` 的 `diagnose()` 产出这些 finding code：

| code | 触发条件 | 性质 |
|---|---|---|
| `observable_input_shift` | `max(|Δmean|/scale) > 0.5` | 阈值 |
| `generalization_gap` | A: `train_f1 − val_f1 > 0.08`；B/C: `val_rmse > train_rmse × 1.3` | 阈值 |
| `train_class_imbalance` | `max/min 类计数 > 2` | 阈值 |
| `weak_classes` | 验证集某类 `F1 < 0.5` | 阈值 |
| `few_drift_examples` | 训练组数 < 3 | 阈值 |
| `worse_than_constant` | `val_rmse > 常数预测 RMSE` | 阈值 |
| `validation_device_dominance` | 最大组 > 50% 行 | 阈值 |
| `snapshot_protocol` | 恒真 | 无条件 |
| `rul_label_delay` | 恒真（B） | 无条件 |
| `no_causal_labels` | 恒真 | 无条件 |

每个 finding 的 `recommendation` 是一句固定的中文句子。**没有一条是根据数据形状生成的。**

---

## 3. 五个问题的直接回答

### Q1：是否真的实现了某种程度的 RSI？能否及时发现问题、自主迭代优化策略？

**不是 RSI，是"带门控的固定配方搜索"。**

| RSI 的必要条件 | 本实现 |
|---|---|
| 能修改自身的改进机制（不只是参数） | ❌ 只有 6 个固定配方 |
| 能生成新的假设/策略 | ❌ 假设来自源码里的 if/else |
| 能基于历史改进"如何改进" | ❌ 经验池只有 6 个样本，小模型从未被调用 |
| 能自主决定停止/继续/换方向 | ❌ `winner is None → break` |

**"及时发现问题"**：部分成立。诊断能正确报出 `weak_classes`（A 的乙醛 F1=0）、
`worse_than_constant`（B 的 681 > 536）、`validation_device_dominance`（B 的 90.2%）。
这些是**正确的观察**。

**"自主迭代优化策略"**：不成立。它做的是"在 6 个预设里试 3 个，取最好的"。
这更像 AutoML 的一层薄封装，而不是 RSI。

### Q2：训练了什么模型？结构？输入输出？想得到什么？

见 `docs/03_baseline.md`。三个任务共用同一套 128/34/24 → hidden → 6/1 的 tanh MLP。
**harness 不训练新结构**——它只在基线 checkpoint 上续训，结构被 `Model.load()` 锁定。

### Q3：基线是怎么做的？

平台用固定配方（Adam lr .003/.002，20/15/15 epoch）在公开 train 上训好，
把 checkpoint 连同 Adam 状态一起发下来。`docs/03_baseline.md` 有完整复现，
`audit/scripts/02_baseline_metrics.py` 可逐位复算。

### Q4：有没有大胆创新？卡在局部最优时有没有深度分析？

**没有。**

- 卡住时执行的是 `break`（`harness/runner.py:247`），不是分析。
- 唯一称得上"创新"的设计是**双 12 维诊断状态向量 + 策略小模型**，
  但这两个东西**都有严重缺陷，且从未生效**（见 §4）。
- 没有任何针对"为什么卡住"的机制：不看病灶样本、不看梯度、不看学习曲线形状、
  不比较多个随机种子、不做显著性检验。

### Q5：运行时是什么？

**纯 NumPy、单进程、无 GPU、无网络、无子进程。**

| 任务 | 总耗时 | 其中训练 | 候选数 |
|---|---|---|---|
| A | **1.399 s** | 0.059 s | 6 |
| B | **2.249 s** | 0.279 s | 6 |
| C | **23.536 s** | 5.181 s | 6 |

C 慢是因为基线诊断要在 113,089 行上跑一遍前向（约 15.9 s）。
资源效率这一项（客观分 5 分）**基本是白送的**。

---

## 4. 三个"看起来有、实际没有"的东西

这是本次审计最值得写进报告的部分——**它们是评分风险，不是加分项**。

### 4.1 策略小模型一次都没被调用过

`StrategyRecommenderMLP`（`harness/strategies.py:243`，12→24→6 ReLU）存在，
预训练权重 `strategy_recommender_pretrained.npz` 也在，`USAGE.txt` 宣称"双引擎"。

但三次 demo 的 `summary.json` 里：

```json
{"decision_mode": "rule", "rule_based": true, "trainer_engine": "RuleDrivenHarnessTrainer"}
```

**全部是 rule。** 小模型零决策。

更糟的是 `demo/harness_overview.html` 顶部写"集成 RuleDrivenHarnessTrainer 与策略小模型 v0.2"，
**这是事实性错误**，评委一查 `summary.json` 就会发现。

### 4.2 两个 12 维状态向量互不兼容，其中一个彻底是死代码

| | `DiagnosticState.to_vector()` | `diagnosis_to_state_vector()` |
|---|---|---|
| 位置 | `harness/training.py:161` | `harness/strategies.py:54` |
| 粒度 | 逐 **epoch** | 逐 **轮** |
| 特征 | 含 `grad_norm_mean/std`、`train_loss_slope` | 含 `med_shift`、`train_groups`、`weak_class_count` |
| 谁在用 | **无人**（只有 `test_harness.py:379` 引用） | 小模型 `select_candidates()` |

两者都是 12 维，但**含义完全不同**。`USAGE.txt` 说小模型吃的是 `DiagnosticProbe` 的向量，
**这是错的**。

### 4.3 梯度统计量恒为 0

```python
# harness/training.py:529  —— 唯一的生产调用点
probe.record_epoch(..., grad_norms=[], ...)
```

`grad_norms_out` 参数只在测试里被传入过（`test_harness.py:367`）。
所以每一次 `diagnostic_trace` 里的 `grad_norm_mean` / `grad_norm_std` 都是 **0.0**，
而 `USAGE.txt` 把"梯度统计量"列为诊断能力之一。

---

## 5. 代码级证据清单

| 结论 | 位置 |
|---|---|
| 6 个配方 | `harness/strategies.py:20` |
| 超参硬编码 | `harness/strategies.py:111-193` |
| lr 恒为 0.3×父 | `harness/strategies.py:212` |
| 规则路由 if/else | `harness/strategies.py:196-236` |
| 静默截断候选 | `harness/strategies.py:236` `return strategies[:max_candidates]` |
| 12 维轮级向量 | `harness/strategies.py:54` |
| 小模型定义 | `harness/strategies.py:243` |
| 卡住就 break | `harness/runner.py:247` |
| 门控严格大于 | `harness/runner.py:31`（`gain > min_improvement`） |
| 死代码 to_vector | `harness/training.py:161` |
| 梯度恒 0 | `harness/training.py:529` |
| 4 个魔法阈值 | `harness/diagnostics.py:81` |
| NASA 分数是求和 | `harness/diagnostics.py`（`np.expm1(exponent).sum()`） |
| 不用 LLM 的自述 | `harness/reporting.py:166` |
| 自承认的风险 | `harness/runner.py:16-24` WARNINGS |

---

## 6. 公平地说：真实存在的工程优点

不能只挑毛病。以下是**实打实做对了**的地方，答辩时应该讲：

1. **训练角色隔离**：`if train.role != "train": raise ValueError("optimizer accepts train-role data only")`
   （`harness/training.py:364`）。
2. **归一化冻结校验**：每个 epoch 后与选择后都验证 `mean/scale/target_mean/target_scale` 未被改动。
3. **父 checkpoint 不变性**：候选训练前后校验父模型摘要未变。
4. **预算记账**：`Partial epoch is never checkpointed or eligible for acceptance`
   （`harness/training.py:400`）——半轮不落地，避免"预算用尽但已污染模型"。
5. **交付前自检**：`runner.py:258` 重新加载交付的 checkpoint，指标不符就报错。
6. **确定性**：同种子同数据 → 逐位相同；`make test` 里有 3 项专门测这个。
7. **数据导入安全**：摘要校验、只解必要成员、拒绝重复成员、大小上限、符号链接拒绝。
8. **诚实的自述**：`harness/runner.py:16-24` 的 WARNINGS 明确写了
   "多轮针对公开验证集选策略可能过拟合；提升不保证迁移至隐藏集"、
   "诊断是统计关联与候选假设，不是真实漂移原因或因果结论"。
   **这句话本身就应该出现在技术报告里。**

---

## 7. 一句话给答辩用

> 我们做了一个**可复现、可审计、带门控的模型续训流水线**，
> 它在三个工业任务上把公开基线提升了 0.0126 / 29.0 / 1.11，
> 并且完整记录了每一次"诊断—候选—门控"的决策链。
>
> 它**不是** RSI，也**没有**用大模型；我们把它定位为
> "受控自改进的第一步：先把过程做对，再谈自动化"。
