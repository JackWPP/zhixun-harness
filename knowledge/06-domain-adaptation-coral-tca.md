---
title: 域适应基础：CORAL 与 TCA 的适用边界及离线落地方式
slug: domain-adaptation-coral-tca
tags: domain, adaptation, coral, tca, transfer, 域适应, 迁移学习, 二阶统计
applies: 源域（训练批）与目标域（当前批）特征分布差异明确，且目标域无标签或标签极少
not_applies: 目标域标签充足（直接重训更简单）；标签先验变化主导；目标批样本极少
data_needed: 源域特征（必需）、目标域无标签特征（必需）；标签只在评估时使用
sources: Sun, Feng & Saenko (2016) Return of frustratingly easy domain adaptation, AAAI（CORAL）；Pan et al. (2011) Domain adaptation via transfer component analysis, IEEE TNN；Ben-David et al. (2010) A theory of learning from different domains, Machine Learning 79
---

# 一句话结论

CORAL 是"白化源域 + 重着色目标域"的**二阶统计对齐**，闭式、便宜、无需目标标签；TCA 用核 MMD 找共享子空间，更强但需调核、成本高。在本仓库的受控约束下，**recenter(batch_mean/batch_std) 是它们的低风险近似**；引入真正的 CORAL/TCA 需要改推理路径或特征层，属于跨模块变更，必须先申请。

## 适用条件

- 漂移被归因为协变量/传感器域差异（`feature_shift` 明显、标签分布相对稳定）。
- 目标域特征在推理期可获得（批量预测天然满足；逐条流式需要滚动统计）。
- 允许推理期变换特征或对特征层做对齐（本仓库当前只允许 postprocess 白名单）。

## 方法步骤

1. **对齐准备**：源域统计量来自训练集（必须固定，不能用目标标签重算）；目标域统计量用当前批次无标签特征估计。
2. **CORAL 闭式解**：源协方差 `Cs`、目标 `Ct`；白化源特征后乘 `Ct^{1/2} Cs^{-1/2}`（或对目标做重着色）。正则化协方差估计（加 εI）防止病态。
3. **TCA**：构造源+目标核矩阵，中心化后求 MMD 最小化的前 k 个特征向量，把两域投到共享子空间再训练/推理；核带宽与维度需要验证集选择（注意合规）。
4. **轻量近似**：先做逐特征均值/方差对齐（≈ diagonal CORAL），若 selection 有效再考虑全协方差。
5. **伪标签自训练**（更激进）：用当前模型给目标批打高置信伪标签加入训练；属于 transductive，正式评测是否允许必须依据规则声明，禁止使用目标标签。
6. **声明**：任何用到目标批特征的推理期方法都要在报告中写明"无标签、transductive、标签不可得"。

## 与 harness 的映射（可执行）

| 方法 | 本仓库可行方式 | 风险 |
|---|---|---|
| 均值对齐 | `recenter(mode=batch_mean, weight≤0.5)` | 低 |
| 均值+尺度对齐 | `recenter(mode=batch_std, weight≤0.4)`，批次足够大 | 中 |
| 对角 CORAL | 逐特征对齐，可用现有 recenter 近似 | 低-中 |
| 全协方差 CORAL | 需要改 `apply_postprocess`/特征层，跨模块变更 | 高，需申请 |
| TCA | 需要新的训练/推理组件与更多预算 | 高，不建议在 10 轮内尝试 |
| 伪标签自训练 | 需要新策略与合规审查 | 高，默认禁止 |

## 如何验证

- **对齐效果**：对齐后计算目标批与源域的 KS/MMD（用无标签特征），并检查 selection 指标是否越 margin。
- **负对照**：随机置换目标批特征后做同样对齐，收益应消失。
- **标签先验检查**：若标签分布 JS 高，先解决先验漂移，再做域适应，否则对齐会把类别差异抹平。
- **跨批泛化**：A 任务可用 UCI batch3–10 做外部探针（只验证不训练），观察方向一致性而不是追最高值。

## 数据需求

源域特征与固定统计量、目标域无标签特征（批量 ≥ 数百行更稳）；标签仅用于评估。全协方差方法要求特征维度不高（A 的 128 维尚可，B/C 的 24–34 维更友好），样本数应显著大于维度以避免协方差奇异。

## 实验证据（公开 + 本仓库）

- 公开：CORAL 在多个视觉域适应基准上以极低成本接近深度方法（Sun et al. 2016）；TCA 在无监督域适应中给出核子空间解（Pan et al. 2011）；域适应理论给出目标误差上界与域差异的关系（Ben-David et al. 2010）。
- 本仓库：A 的外部漂移探针中，逐批 recenter（对角对齐）把宏 F1 从 0.4054 提升到 0.5239；这属于"对角 CORAL 级"证据，**未实现/未验证全协方差 CORAL 与 TCA**。
- 本仓库：A 官方验证集是单批 batch2，跨批收益必须在外部探针上验证，不能只看单批 selection。

## 风险与不适用条件

- 目标域标签不可得时，域适应不能修复概念漂移（P(Y|X) 变）。
- 伪标签自训练会自我强化错误，且可能触碰"隐藏标签/外部标签"红线。
- 全协方差对齐在维度高、样本少时不稳定；正则化不当会毁掉特征。
- 目标批统计量随批次变化，逐条在线推理需要滚动窗口，与批量评测口径不同。
- 任何新推理路径都必须保持官方加载器兼容（checkpoint 不变、协议不变）。

## 来源

Sun, Feng & Saenko, AAAI 2016；Pan et al., IEEE TNN 2011；Ben-David et al., Machine Learning 2010；Ganin et al., ICML 2015（DANN，作为对照）；本仓库 docs/11_COMPLIANCE.md。
