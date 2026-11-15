# 06 · 风险清单与答辩应对

> 每一条都是**评委可能当场问到的**。按"杀伤力"排序。
> 左列是风险，右列是**已经准备好的应对**。

## 🔴 致命级（不处理会直接丢分）

### R1. 宣称"RSI / 自进化 Agent"，但代码里没有 Agent

**证据**：`grep -rniE "llm|gpt|prompt|openai" harness/` 只有 2 处命中，
且都是 HTML 里的字符串（`harness/reporting.py:154` 的 CSP、
`harness/reporting.py:166` 的自述"策略与诊断基于规则，不使用 LLM"）。

**评委问法**："你们的 Agent 用的是什么模型？prompt 在哪？"

**应对**：
> 我们不称它为 Agent，称它为**受控自改进流水线（controlled self-improvement pipeline）**。
> 它实现了 RSI 的两个必要条件——**自我诊断**和**受控更新**——
> 但**没有**实现第三个：**自主生成新假设**。
> 这是我们主动划定的边界，也是下一步要做的。

**不要做**：硬说"规则引擎也算 Agent"。评委见过太多这种话术。
**要做**：把"我们知道自己不是什么"写进技术报告第一页。这一条会加印象分。

---

### R2. 宣称"集成了策略小模型"，但三次运行全是 rule 决策

**证据**：`demo/harness-runs/{A,B,C}-v1/summary.json` 里
`"decision_mode": "rule"`、`"rule_based": true`、`"trainer_engine": "RuleDrivenHarnessTrainer"`。
小模型（`harness/strategies.py:243`）一次都没被调用。

更麻烦的是 `demo/harness_overview.html` **顶部**写
"集成 RuleDrivenHarnessTrainer 与策略小模型 v0.2"——这是**事实性错误**，
而同一页**底部**又诚实写了"第一版不接大模型、没有独立诊断小模型"。

**评委问法**："小模型在哪里起作用了？"

**应对（二选一，必须选一个）**：

- **方案 A（推荐，1 小时）**：现在就补一次 `--decision-mode small_model` 的运行，
  让 `v0.2` 名副其实。风险：小模型只有 6 条训练样本，结果可能更差——
  但那反而是**很好的实验素材**（"我们对比了规则 vs 小模型，见文档"）。
- **方案 B（30 分钟）**：删掉 overview.html 顶部那句话，全文统一口径为
  "规则路由为主，小模型为研究原型，未进入本次交付"。

**绝对不要**：保留现在的矛盾状态。评委只要打开两个文件对照，就是"夸大宣传"。

---

### R3. 没有留出集，报告的"after"分数是选择后的乐观估计

**证据**：候选在每个 epoch 用验证集选最佳 epoch（`harness/training.py:517`），
再用同一个验证集做候选门控（`harness/runner.py:31`），
最后把赢家的验证集分数报成"结果"。三次 demo 的 `validation_policy` 都是
`"evaluation_and_selection_only"`，但**没有任何 holdout**。

**评委问法**："你的验证集既用来选模型又用来报分数，这个数字可信吗？"

**应对**：
> 不可信为无偏估计，这一点我们在报告里明确标注了。
> 我们做了配对 bootstrap，三个任务的提升在 95% 置信区间下都显著，
> 但**幅度是乐观的**。A 的 +0.0126 来自 10 个样本的改变，
> B 的 +29 是在修一个比常数还差的基线，C 的第二轮只有 +0.0106。
> 下一步的第一件事就是切 holdout。

这个回答的关键是**主动说出数字的局限**，而不是等评委问。

---

### R4. B 的"提升"是在修一个坏基线，且修完仍然不可用

**证据**：常数预测 RMSE **536.271**，基线 681.160，harness 后 652.190。
**即使做完 harness，还是比"永远预测平均值"差 21.6%。**

**评委问法**："你们的模型比瞎猜好吗？"

**应对**：
> 在 B 上**没有**。这个任务的基线本身是坏的，我们把它拉回来了 29 分钟，
> 但真正的瓶颈是输入里缺少 bearing-level 上下文（哪个轴承、已运行多久），
> 模型只能靠振动特征跨轴承泛化，这在这个数据上做不到。
> 我们认为**如实报告这一点**比虚报一个好看的数字更有价值。

**加分动作**：在报告里放一张图——"常数 / 基线 / harness / 理想"四条横线。
这张图会让"实验分析"那 15 分很难不给。

---

## 🟠 高风险（会扣分但可挽救）

### R5. 测试在独立解包目录下跑不起来

**证据**：`harness/tests/test_harness.py:37` 与 `test_runner.py:19` 硬编码了
`C:/Users/zhao jiahao/Downloads/cpu-a-guide-20261110-v1.zip`。独立跑 → 0 测试、2 errors。

**已经解决**：两个文件都用了
`Path(os.environ.get("HARNESS_TEST_VENDOR_ARCHIVE", "C:/Users/zhao jiahao/..."))`，
所以**只需设环境变量**，不用改代码。仓库的 `Makefile` 已自动 export：
```make
export HARNESS_TEST_VENDOR_ARCHIVE := $(CURDIR)/$(ARCHIVE_A)
```
`make test` → 30 项全过。

**评委问法**："clone 下来能跑吗？"

**应对**：现场 `make test`。**必须提前验证一遍**（见 `docs/07_handover.md`）。

---

### R6. `harness_verification.json` 和 `harness_test_results.txt` 无法溯源

**证据**：`harness_prototype.zip` 里**没有任何代码会生成这两个文件**
（`reporting.py` 只写 `report.html`，`__main__.py` 只有 `run` 和 `train-recommender`）。
`harness_verification.json` 的 `checks` 全是裸字符串，没有 per-check 证据、时间戳或命令。

**评委问法**："这个 verification 是谁生成的？"

**应对**：**直接重新生成**。仓库里的 `make ci`（`test` + `verify`）会产出可复现的证据链，
`audit/scripts/00_verify_bundle.py` 会把 sha256 和逐位复算结果写进 `audit/results/`。
**用真实产物替换掉这两个来源不明的文件。**

---

### R7. `configs/B.json` 在 A/C 包里是错的（版本漂移）

**证据**：`datasets/cpu-a-guide-*/baselines/configs/B.json` 与 `cpu-c-guide-*/.../B.json`
声明 `"task":"classification","classes":["Normal","Degrading","SevereFault"]`，
但同一个包里的 `TASK.md`、`baseline-info.json`、`training.json` 都说是 **regression**。
只有 `cpu-b-rul-*/baselines/configs/B.json` 是对的。

**评委问法**："你们用哪个配置文件？"

**应对**：这是**平台发的包本身的 bug**，不是团队的错。
但**团队必须在报告里指出这件事**——它证明"我们真的读了数据"。
一句话："平台 A/C 包内的 `configs/B.json` 是版本漂移产物，与 `TASK.md` 冲突，
我们以 `baseline-info.json` 与 `training.json` 为准。"

---

### R8. A/C 的 `label_status` 是 `draft`

**证据**：`baseline-info.json` 里 A `"label_status": "draft"`、B `"approved"`、C `"draft"`；
A/C 的 `training.json` 还有 `"official_scoring_ready": false`、`"purpose": "development_reference"`。

**意味着**：A 和 C 的基线分数**可能不是最终官方分数**，正式评测时可能变。

**应对**：报告里注明"本文所有数字基于 `-v1` 开发版数据，
正式评测若更换数据版本，结论需重新验证"。这是**加分项**（安全意识）。

---

### R9. C 的候选计划与诊断结果毫无关系

**证据**：C 的诊断只产出 `snapshot_protocol` 和 `no_causal_labels` 两个**恒真**结论，
于是 `rule_select_candidates` 落到硬编码 else 分支。唯一由数值触发的
`penalize_late_rul`（`overestimate_fraction = 0.5719 > 0.50`）被
`return strategies[:max_candidates]`（`harness/strategies.py:236`）**静默截断，从未运行**。

**评委问法**："C 上你们根据什么选的策略？"

**应对**：**不要说"根据诊断"**。说："C 的诊断没有产出可操作的信号，
所以我们退回到默认配方组合。这暴露了一个设计缺陷：当诊断沉默时，
系统没有能力主动探索。这是我们 roadmap 的第一项。"

---

### R10. 经验池只有 6 个样本，小模型不可能学到东西

**证据**：3 任务 × 2 轮 × 3 候选 = 18 条轨迹，但**唯一 state_vector 只有 6 个**
（同一轮三个候选的状态向量字节相同）。`harness_verification.json` 自己写着
`unique_decision_states: 6`。

**应对**：这是 R2 方案 B 的补充论据——**"小模型不是我们没做，是这个数据量下做了也没用"**。
把这句话写进报告，比假装它有效要好得多。

---

## 🟡 中风险

| # | 风险 | 应对 |
|---|---|---|
| R11 | 门控 `min_improvement` 默认 0.0，严格 `> 0` 就接受，C 第二轮 +0.0106 也过 | 报告里写明，并给出"建议阈值 = 2×种子标准差"（A 应设 0.006，B 应设 9.4） |
| R12 | `DiagnosticState.to_vector()` 是死代码，与真正用的 12 维向量不兼容 | 报告里作为"我们发现的实现不一致"列出，体现审计深度 |
| R13 | `grad_norm_mean/std` 恒为 0（`harness/training.py:529` 传空列表） | 同上，列为已定位的缺陷 |
| R14 | NASA score 是 23049 个样本的 **求和**，得到 5.8e9 的无意义数字 | 报告里注明"本地 NASA 分数仅作相对比较，绝对值不可用" |
| R15 | A 的 `max_shift 11.4582` 是单个离群点造成的 | 改用中位数偏移，报告里说明两个数字的差异 |
| R16 | 三个任务都是一次配方实验，没有消融表 | 至少补一张"6 个配方 × 3 任务"的完整矩阵（1 小时，见 roadmap） |

---

## 一页纸答辩备忘

**开场（30 秒）**
> 我们审计并重做了一个"模型自进化"原型。先说结论：它**不是** RSI，
> 也**没有**用大模型；它是一个带门控的固定配方搜索器。
> 我们把它工程化、复现、量化，并给出了**下一步真正做 RSI 的路线**。

**三个数字（60 秒）**
> A +0.0126（翻了 10 个样本）、B −29（但在修一个比常数差 27% 的基线）、
> C −1.11（第二轮只 −0.0106）。
> 三个提升统计显著，但**幅度是乐观估计**，因为没有留出集。

**三个洞察（90 秒）**
> 1. A 的乙醛 F1 在所有容量、所有配方下恒为 0 → 问题是**域适应**，不是超参。
> 2. B 的基线比常数差 27% → 问题是**缺上下文特征**，不是正则化。
> 3. C 的容量曲线完全平、train≈val → 问题是**特征受限**，不是模型太小。

**收尾（30 秒）**
> 所以我们认为这个赛道"卷 0.01 指标"没有意义。
> 客观分只有 15 分给性能提升，45 分给方法论。
> 我们的交付是**可复现的审计 + 可执行的路线**，而不是一个虚高的数字。

---

## 红线（绝对不能说的话）

| ❌ 不能说 | ✅ 改成 |
|---|---|
| "我们实现了 RSI" | "我们实现了受控自改进的两个必要环节" |
| "集成了 LLM Agent" | "当前版本不含 LLM，这是设计选择" |
| "小模型提升了决策质量" | "小模型是研究原型，样本量不足，未进入交付" |
| "B 提升 4.25%" | "B 从 681 降到 652，但仍差于常数 536" |
| "代码完全可复现" | "30 项测试可通过 `make test` 复现，依赖一个环境变量" |
| "数据质量不够是瓶颈" | "数据量充足，瓶颈是独立单元数与特征表达" |
