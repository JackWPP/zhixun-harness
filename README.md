# 智训引擎 Harness —— 工业模型受控自进化（比赛工程仓库）

> 北京市大学生"人工智能+"创新大赛 · 企业赛道（软通计算机有限公司 / 华为技术有限公司）
> 赛题：**"智训引擎" —— 工业制造场景下的模型训练与超参数优化**
>
> 本仓库把团队交付的 5 个压缩包**全部解包、重新组织、可一键复现**，并附上第三方
> 独立审计（`audit/`）与移交文档（`docs/`）。目标：任何人 clone 下来就能跑、能验证、
> 能接着往下做。

---

## 1. 三十秒了解现状

| 维度 | 结论 | 证据 |
|---|---|---|
| 是不是 RSI / 自主 Agent | **不是**。没有 LLM、没有网络、没有工具调用，是一个确定性规则引擎 | `docs/04_harness_audit.md` §2 |
| 可选择的动作空间 | **6 个写死的配方**（conservative / class_weighted / robust_noise / device_balanced / regularized / penalize_late_rul），所有超参在源码里硬编码 | `harness/strategies.py:20`、`harness/strategies.py:111` |
| 能不能自主迭代优化 | 能"换配方 + 续训"，但**不能发明新配方**；lr 恒为父模型的 0.3×；卡住就 `break` | `harness/runner.py:247` |
| 策略小模型用上了吗 | **没有**。三次 demo 全部 `decision_mode="rule"`，小模型零决策 | `demo/harness-runs/*/summary.json` |
| 训练侧做了数据工作吗 | **没有**。只有训练集 z-score + ±10 裁剪，无清洗/去重/异常值/特征选择/窗口化 | `harness/training.py`、`docs/05_findings.md` §5 |
| 三个任务的"提升" | A +0.0126（10 个样本翻转）、B +28.97（基线本就劣于常数）、C +1.11 | `docs/05_findings.md` §1 |
| 运行时 | 纯 NumPy 单进程，A/B/C 分别 **1.40 / 2.25 / 23.54 秒** | `demo/harness-runs/*/summary.json` |
| 测试 | **30 项全部通过**（需设置 `HARNESS_TEST_VENDOR_ARCHIVE`） | `make test` |

> 详细判定与整改建议见 [`docs/06_risks.md`](docs/06_risks.md)。

---

## 2. 三分钟跑起来

```bash
git clone <this-repo> && cd <repo>

make setup          # 建 .venv 并安装 harness（可编辑模式）
make test           # 30 项单元测试
make sums           # 校验 5 个原始 zip 的 sha256
make verify         # 复算基线分数 + demo 分数（含 sums）
make run-a          # 在任务 A 上跑一轮完整 harness
make audit          # 复现全部审计结论（约 1–3 分钟）
```

不想建虚拟环境也可以直接用系统 python3（>=3.9，只需 numpy）：

```bash
python3 -m unittest discover -s harness/tests -v
python3 -m harness run --task C --archive archives/cpu-c-guide-20261110-v1.zip \
        --output /tmp/C-run --epochs 5 --max-candidates 3 --rounds 2
```

**测试必须先设环境变量**，否则会去读原开发者机器上的 Windows 路径：

```bash
export HARNESS_TEST_VENDOR_ARCHIVE="$PWD/archives/cpu-a-guide-20261110-v1.zip"
```

（`make test` 已自动设置；`harness/tests/test_harness.py:37` 与
`harness/tests/test_runner.py:19` 读的就是这个变量。）

---

## 3. 目录结构

```
.
├── archives/            原始交付物，只读存档（15 MB，5 个 zip）
│   └── SHA256SUMS       摘要清单，make verify 会比对
├── datasets/            从 archives 解包出来的三个数据包（58 MB）
│   ├── cpu-a-guide-20261110-v1/   任务 A：气体传感器漂移 · 六类识别
│   ├── cpu-b-rul-20261110-v1/     任务 B：滚动轴承 · 剩余寿命（分钟）
│   └── cpu-c-guide-20261110-v1/   任务 C：涡扇发动机 · 剩余寿命（循环）
├── harness/             harness 原型源码（13 个文件，依赖仅 numpy）
│   ├── strategies.py    6 个写死配方 + 规则路由 + 策略小模型（未被使用）
│   ├── diagnostics.py   阈值诊断（4 个魔法阈值 + 文本模板）
│   ├── training.py      受控续训内核（execute_epoch / execute_candidate）
│   ├── runner.py        两轮控制循环 + 门控
│   ├── data.py          数据导入与角色隔离
│   ├── reporting.py     静态 HTML 报告
│   └── tests/           30 项单元测试
├── demo/                团队交付的 demo 证据（153 个文件，8.6 MB）
│   ├── harness-runs/{A,B,C}-v1/   三次运行的完整留痕
│   ├── harness_overview.html
│   ├── harness_verification.json
│   └── harness_test_results.txt
├── audit/               第三方审计（本次新增）
│   ├── scripts/         10 个编号脚本（00–09）+ _common + run_all
│   └── results/         我独立重跑的 harness 输出（audit-{A,B,C}-r2）
├── docs/                移交文档（9 篇）
├── Makefile             唯一入口：make help
├── pyproject.toml       可 pip install -e .
└── .github/workflows/   CI：test + verify
```

---

## 4. 三个任务速查

| | A 气体识别 | B 轴承 RUL | C 发动机 RUL |
|---|---|---|---|
| 数据来源 | UCI Gas Sensor Array Drift (#224) | XJTU-SY 全寿命 | NASA C-MAPSS FD001–004 |
| 输入 | 128 维 | 34 维（17×2 通道） | 24 维（3 设置 + 21 传感器） |
| 输出 | 6 类 | 分钟数 | 循环数 |
| 主指标 | macro-F1 ↑ | RMSE ↓ | RMSE ↓ |
| 训练 / 验证 | 445 / 1244 | 7032 / 1679 | 113089 / 23049 |
| 模型 | 128-32-6 tanh | 34-24-1 | 24-32-1 |
| 参数量 | 4326 | 865 | 833 |
| 基线分数 | 0.714030 | 681.160 | 52.869 |
| **常数对照** | — | **536.271（基线差 27%）** | 79.591（基线好 33.6%） |
| demo 分数 | 0.726672 | 652.190 | 51.758 |

---

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/01_competition.md`](docs/01_competition.md) | 赛题要求、评分口径、提交材料清单 |
| [`docs/02_datasets.md`](docs/02_datasets.md) | 三份数据的字段、切分方式、已知陷阱 |
| [`docs/03_baseline.md`](docs/03_baseline.md) | 基线模型架构、checkpoint 格式、续训协议 |
| [`docs/04_harness_audit.md`](docs/04_harness_audit.md) | harness 逐模块审计：它是什么、不是什么 |
| [`docs/05_findings.md`](docs/05_findings.md) | 全部实验结论与数字（含"数据量/质量不够"是否成立） |
| [`docs/06_risks.md`](docs/06_risks.md) | 评分风险清单 + 按优先级的整改建议 |
| [`docs/07_handover.md`](docs/07_handover.md) | 服务器移交步骤、环境要求、常见问题 |
| [`docs/08_roadmap.md`](docs/08_roadmap.md) | 接下来该做什么（可执行的工作项） |
| [`docs/09_glossary.md`](docs/09_glossary.md) | 术语表（RSI / 门控 / 续训 / 漂移 …） |

---

## 6. 已知限制（请勿在答辩中夸大）

1. **没有 LLM、没有 Agent、没有 RSI。** 这是一条确定性流水线：诊断 → 从 6 个配方里挑 → 续训 → 门控。
2. **验证集既用于选择又用于汇报。** 所有 improvement 都是乐观估计，没有留出集。
3. **B 的基线比"永远预测训练集均值"还差 27%。** 在 B 上"降低 RMSE"首先是在修一个坏基线。
4. **A 的 Acetaldehyde F1 恒为 0。** 训练集 30 个样本、验证集 109 个，跨批次完全不可迁移；加宽到 128 维隐层也救不回来。
5. **C 的验证集是同分布随机切分**，不是漂移测试；`snapshot_protocol` 是唯一诊断结论。
6. **`harness_verification.json` / `harness_test_results.txt` 不是原型代码生成的**，是外部脚本的产物。

---

## 7. 许可与来源

- `archives/`、`datasets/`、`harness/`、`demo/` 来自比赛平台与团队原始交付，**未做任何逻辑修改**，仅重新组织目录。
- `audit/`、`docs/`、`Makefile`、`pyproject.toml`、CI 为本次移交新增。
- 数据集原始出处：UCI ML Repository、XJTU-SY、NASA PCoE。请遵守各自的使用条款。
