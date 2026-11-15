---
title: 漂移检测统计量：KS / PSI / MMD / Page-Hinkley / ADWIN 的选择与阈值
slug: drift-detection-statistics
tags: drift, detection, ks, psi, mmd, ph, page-hinkley, adwin, 检测, 统计量, 阈值
applies: 需要为一个特征或一组特征选择检测统计量、设定报警阈值、控制误报
not_applies: 目标窗只有个位数样本；或特征含大量缺失/离散字典且无稳定分箱方案
data_needed: 参考窗口与当前窗口的原始特征值；时序在线检测需要按到达顺序排列的样本流
sources: Massey (1951) KS two-sample test, JASA；Siddiqi (2006) Credit Risk Scorecards（PSI 工业阈值）；Gretton et al. (2012) Kernel two-sample test, JMLR；Page (1954) Continuous inspection schemes, Biometrika；Bifet & Gavaldà (2007) Learning from time-changing data with adaptive windowing, SDM
---

# 一句话结论

**离线批量**先看逐特征 KS + PSI（便宜、可解释），需要检测"联合分布"的整体位移时用 MMD；**在线流式**用 Page-Hinkley（有明确漂移方向）或 ADWIN（窗口自适应、误报可控）。任何阈值都要按样本量与特征数校正，不能照搬。

## 适用条件

- 有参考窗口与目标窗口的**原始特征值**（不需要标签）。
- 需要解释"哪个特征在漂移"（KS/PSI 逐特征）还是"整体是否漂移"（MMD）。
- 需要在线逐条决策（PH/ADWIN）还是周期性批检（KS/PSI/MMD）。

## 方法步骤

1. **逐特征 KS**：`D = sup_x |F_ref(x) - F_tgt(x)|`。大样本下 p 值几乎必然显著，**不要用 p 值当报警器**，用效应量 D：D>0.1 关注、>0.2 报警（结合本仓库 `ks_max`、`ks_p95`）。做多重检验校正（Bonferroni/FDR）。
2. **PSI**：按参考分布分箱（通常 10 分位或固定区间），`PSI = Σ (p_t - p_r) ln(p_t/p_r)`。工业惯例：<0.1 稳定，0.1–0.25 中度，>0.25 显著。分箱必须固定自参考窗口，否则不可比。
3. **MMD**：RBF 核两样本检验，带宽取中位数启发式（median pairwise distance）；用置换检验定阈值。适合多维联合漂移；成本 O(n²)，窗口大时先下采样。
4. **Page-Hinkley**：对误差/残差序列检测均值上移，参数 δ（容忍漂移）与 λ（阈值）；适合"性能突然变差"的在线报警，需要方向假设。
5. **ADWIN**：维护自适应窗口，用 Hoeffding 界判断窗口内均值是否变化；误报有理论保证，适合不知道漂移时刻的流式场景。
6. **组合策略**：批量轮次用 KS/PSI 定位特征，MMD 判整体；在线用 ADWIN 报警、PH 确认方向；报警后统一进入归因流程。

## 与 harness 的映射（可执行）

| 需求 | 现成能力 | 建议动作 |
|---|---|---|
| 逐特征报警 | `rsi/drift.py::_ks_statistic`，报告 `ks_max`/`ks_p95`/`attribution[].ks` | ks_p95>0.2 且 attribution 前几集中 → 增益/基线漂移 |
| 分箱稳定性 | 暂无 PSI，可在监控脚本中按参考分箱复算 | 作为外部诊断补充，不改变门控 |
| 整体联合漂移 | 暂无 MMD，报告用相关性位移近似 | 多维相关结构变化时优先看 `correlation_shift` |
| 在线轮次 | 每轮读 `state.drift`；`window_rows` 逐轮增大 | 连续 2 轮 severity 升级 → 提前干预 |

## 如何验证

- **置换检验**：把两窗口混合后重采样，重复 200 次得到 KS/MMD 零分布，报警阈值取 95 分位；确认实际统计量超阈值。
- **固定分箱复算 PSI**：用参考窗口分位点存成 JSON，跨轮复用；分箱漂移会导致 PSI 虚高。
- **误报率回归**：在稳定窗口上跑检测器，要求 10 次运行误报 ≤ 1 次；否则调阈值。
- **已知漂移注入**：对目标窗加已知 gain（如 1.1×）或偏移（0.5σ），确认检测器在预期特征上报警。

## 数据需求

参考窗口建议 ≥ 500 行（KS 的 D 分辨率约 1/√n），目标窗 ≥ 200 行；在线检测需要按时间排序的流式样本与明确的报警成本（漏报 vs 误报）。PSI 需要稳定分箱方案；MMD 需选择核与带宽。

## 实验证据（公开 + 本仓库）

- 公开：KS 两样本检验（Massey 1951）；MMD 的一致性与核选择（Gretton 2012）；ADWIN 的误报界（Bifet & Gavaldà 2007）；PSI 0.1/0.25 阈值来自信用评分工业实践（Siddiqi 2006）。
- 本仓库：`rsi/drift.py` 已实现无 SciPy 依赖的 KS 与 JS；A 任务 KS 均值被纳入 severity 公式（权重 0.35），B/C 的输入维（34/24）适合逐特征 KS。
- 诚实提醒：本仓库没有对 PSI/MMD/ADWIN 做离线评测，本文档只给选择依据与阈值建议，不宣称已实现。

## 风险与不适用条件

- 大样本下 p 值失去意义；只用 p 值会天天报警。
- KS 对离散/并列值敏感，离散特征应先做分箱或换卡方类统计量。
- PSI 依赖分箱，分箱一变结论就变；空箱要加平滑。
- MMD 的 O(n²) 成本与核带宽选择会显著改变功效。
- 多个特征同时检测必然产生误报，必须做多重比较控制。
- 检测到 P(X) 变化不等于性能会下降，**报警之后仍要做归因与实验**。

## 来源

Massey, JASA 1951；Gretton et al., JMLR 2012；Bifet & Gavaldà, SDM 2007；Page, Biometrika 1954；Siddiqi, Credit Risk Scorecards, Wiley 2006；Rabanser et al., NeurIPS 2019（漂移检测方法的大规模评估）。
