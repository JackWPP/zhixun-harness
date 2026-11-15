---
title: 受限续训：从父 checkpoint 到候选模型的低成本迁移
slug: model-transfer-continued-training
tags: transfer, continued-training, finetune, swa, ridge, 续训, 迁移, 微调, 学习率
applies: 需要在不改变网络结构与加载协议的前提下，用官方 train 继续训练生成更强候选
not_applies: 任务要求从头训练；或需要改结构/归一化协议（属于禁止项）；训练集极小且有严重过拟合风险
data_needed: 官方 train（可含规则允许的 calibration 拟合）、父 checkpoint、selection 切分用于门控
sources: 本仓库 docs/12_EXPERIMENTS.md 与 docs/16_DECISIONS.md；Neyshabur et al. (2020) What is being transferred in transfer learning, NeurIPS；Loshchilov & Hutter (2019) Decoupled weight decay regularization, ICLR
---

# 一句话结论

续训 = **保留父模型全部结构与非参数状态，只更新参数**。先用低 lr 短程试探，再用分段 lr（先大后小）精修；参数预算内优先尝试"可解释、可回滚"的小步，如只换输出头（ridge_last_layer）、SWA 平均或多阶段续训。

## 适用条件

- 候选必须能被官方加载器直接加载：结构、归一化、类别表、meta 均不变。
- 漂移被归因为概念/表征不足，而非单纯输入覆盖问题（覆盖问题优先校准，不占 checkpoint 预算）。
- 官方 train 足够支撑一次短程续训；selection 可用于门控但不可用其标签拟合。

## 方法步骤

1. **建立父模型基线**：在 selection 上复算父模型指标，确认可复现；记录 sha。
2. **小步试探**：`continue_train(lr≈1e-4, epochs≈5)` 观察 loss 与 selection 是否同步；若 loss 降但 selection 不升，警惕过拟合。
3. **分段 lr**：`continue_train_staged` 先大 lr 阶段（例如 1e-3）再小 lr（1e-5），总 epoch 按预算；C 任务实测 1000+400 epoch 起步有效。
4. **输出头替换**：表征可能已好、头失配时用 `ridge_last_layer(alpha)` 闭式重拟合，成本极低；A 任务曾用其替代全量续训。
5. **权重平均**：`swa` 在多个快照上平均参数，通常改善泛化；需要足够 epoch 与合理快照数（2–6）。
6. **早停与门控**：只用 selection 决定是否继续；训练损失不是选择依据；连续无提升则 retain。
7. **限量与回滚**：参数预算（每任务 ≤40 轮）内规划；候选劣化 ≥2×margin 立即 rollback，保留父 checkpoint。

## 与 harness 的映射（可执行）

| 策略 | 关键参数 | 适用场景 |
|---|---|---|
| `continue_train` | lr 1e-5~0.05、epochs 1~300、data="train" | 快速验证续训方向 |
| `continue_train_staged` | 两段 lr/epochs | 长程精修（C 首选） |
| `ridge_last_layer` | alpha 1e-6~1e6、data="train" | 只换头，诊断表征质量 |
| `swa` | lr、epochs、snapshots 2~6 | 抖动大、泛化不足 |
| `retain` | — | 无提升或证据不足时的安全动作 |
| `extended_pretrain` | 仅 A、需 opt-in | 合规允许时的外部数据预训练 |

## 如何验证

- **可加载性**：用官方 loader 加载候选，比较结构摘要、normalize 协议、meta 与父模型一致，仅参数不同。
- **门控**：selection 主指标 ≥margin 才 promote；多 seed 复跑确认非噪声（A 曾出现 0/10 复现）。
- **对照**：与 retain、ridge、SWA 对照臂比较，而不是只与原始基线比。
- **泛化抽检**：report 只在决策后算一次；外部探针（如 UCI 后期批次）只作方向性参考。
- **成本**：记录 epoch/耗时/参数量，确认在预算内。

## 数据需求

官方 train 数据与标签；父 checkpoint；selection 切分（门控用）；规则允许时的 calibration 数据。续训不得使用外部数据（除显式 opt-in 的 extended_pretrain）与隐藏/报告标签。

## 实验证据（公开 + 本仓库）

- 本仓库严格官方口径：C 用 `continue_train_staged`（首轮 1000+400 epoch，lr 3e-4→1e-5）selection RMSE 44.75→40.52，之后 9 轮 retain；B 用 `damage_shrink` 等推理期手段 678.02→409.74，续训不是唯一路径。
- 本仓库：A 续训/ridge 类方案与 recenter 校准方案都曾提升；但 A 换种子出现 0/10 复现，说明小数据续训结论需谨慎。
- 公开：迁移学习中"保留父模型特征 + 轻量调整"常优于从头训练（Neyshabur 2020）；解耦权重衰减与合理 lr 调度影响泛化（Loshchilov & Hutter 2019）。
- 诚实提醒：本仓库未对 SWA 做完整评测，其参数范围来自策略目录，不宣称已在本任务胜出。

## 风险与不适用条件

- 小训练集续训易过拟合，selection 与 report 背离是警报。
- 改变归一化统计/结构/类别表会破坏官方加载协议，属于禁止项。
- 使用 selection/report 标签拟合参数是泄漏。
- 长时间续训可能遗忘父模型的稳健性（catastrophic forgetting），需与 retain 对照。
- 多 seed 成本高，预算不足时应在记录中声明不确定性。

## 来源

Neyshabur et al., NeurIPS 2020；Loshchilov & Hutter, ICLR 2019；Kirkpatrick et al. (2017) Overcoming catastrophic forgetting, PNAS；本仓库 docs/12_EXPERIMENTS.md、docs/16_DECISIONS.md。
