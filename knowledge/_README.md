# 离线知识库目录（人类导航，不参与索引）

检索入口：`rsi.knowledge`（Python）或 `./.venv/bin/python scripts/build_knowledge.py --query "..."`。
每篇文档为 Markdown + 简单 frontmatter（`title/slug/tags/applies/not_applies/data_needed/sources`），
正文固定小节：一句话结论、适用条件、方法步骤、与 harness 的映射、如何验证、数据需求、实验证据、风险与不适用条件、来源。

| # | 文件 | slug | 主题 |
|---|---|---|---|
| 01 | `01-drift-taxonomy.md` | drift-taxonomy | 协变量 / 标签先验 / 概念漂移分类与处置 |
| 02 | `02-drift-detection-statistics.md` | drift-detection-statistics | KS/PSI/MMD/PH/ADWIN 选择与阈值 |
| 03 | `03-residual-performance-monitoring.md` | residual-performance-monitoring | 有/无标签与标签延迟下的性能监控 |
| 04 | `04-feature-attribution.md` | feature-attribution | 特征归因与 top-k 定位 |
| 05 | `05-calibration-transfer.md` | calibration-transfer | recenter / shrink 推理期校准 |
| 06 | `06-domain-adaptation-coral-tca.md` | domain-adaptation-coral-tca | CORAL/TCA 适用边界与落地 |
| 07 | `07-rul-calibration.md` | rul-calibration | RUL 偏置、收缩、退化曲线校正 |
| 08 | `08-sensor-gain-baseline-drift.md` | sensor-gain-baseline-drift | 传感器增益/基线漂移识别与校正 |
| 09 | `09-experiment-design-falsification.md` | experiment-design-falsification | 单变量、噪声阈值、负对照 |
| 10 | `10-compliance-data-boundary.md` | compliance-data-boundary | 切分边界、外部数据、泄漏红线 |
| 11 | `11-model-transfer-continued-training.md` | model-transfer-continued-training | 受限续训、ridge/SWA、门控 |
| 12 | `12-swanlab-logging.md` | swanlab-logging | 记录、回放、SwanLab 离线优先 |

修改任一文档后重新运行 `scripts/build_knowledge.py`（幂等，内容签名不变则跳过）。
