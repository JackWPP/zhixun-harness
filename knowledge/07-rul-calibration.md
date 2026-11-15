---
title: RUL 校准：偏置、收缩与退化曲线校正
slug: rul-calibration
tags: rul, calibration, regression, shrink, damage, 剩余寿命, 校准, 回归, 退化
applies: 轴承/发动机等 RUL 回归任务出现系统偏置或 RMSE 劣化，需要推理期校准或续训
not_applies: 时间索引泄漏未排除；评估按行而非按单元/轴承分组；测试单元与训练单元不可比
data_needed: 训练目标均值与分布、按单元分组的验证切分、退化轨迹（时序）信息
sources: Saxena et al. (2008) Damage propagation modeling for aircraft engine run-to-failure simulation, PHM；Wang et al. (2008) A similarity-based prognostics approach for RUL estimation, PHM；Nectoux et al. (2012) PRONOSTIA bearing dataset, IEEE PHM 2012
---

# 一句话结论

RUL 误差通常不是对称的：早期高估、后期低估很常见。先用 **bias/分位残差** 判断是系统偏置还是退化段误差，再选 `shrink`（向训练均值收缩，稳但压缩动态范围）或 `damage_shrink`（用因果退化指标做温和校正）；任何方法都要按**单元/轴承分组**验证。

## 适用条件

- 评估协议按单元分组（C 按 unit、B 按轴承），避免同寿命轨迹泄漏。
- 已确认协变量漂移不是主因，或协变量方法已试过。
- 允许推理期后处理（checkpoint 不变）或受限续训。

## 方法步骤

1. **分层看残差**：按真实 RUL 分桶（例如 <25%、25–75%、>75% 剩余寿命）统计 bias 与 RMSE；若仅高 RUL 段偏大，是动态范围问题；整段偏置是校准问题。
2. **bias 修正**：`y' = y_pred + b`，b 由训练/校准切分估计，不能来自 selection/report 标签。
3. **收缩校准**：`y' = α·y_pred + (1-α)·center`，center 取训练目标均值（B 约 694.31 分钟），α 在 selection 上选；α 太小会退化成常数预测，务必与常数基线对比。
4. **退化曲线校正**：构造因果退化指标（如传感器相对初始基线的变化、累计最大值比、单位时间斜率），在训练集上拟合"退化指标→RUL"的单调曲线，再与模型输出混合；参数（权重 w、锚点 a）必须在 train 上拟合。
5. **单调性/物理约束**：预测随运行时间应总体下降；出现非单调跳变时，优先修特征或加后处理约束，而不是继续调模型。
6. **分组验证**：C 用按发动机单元整组切分；B 用留一轴承（LOBO）或按轴承校准，避免同轨迹泄漏。

## 与 harness 的映射（可执行）

| 工具 | 用途 | 关键约束 |
|---|---|---|
| `shrink(alpha, center)` | 整体偏置/尺度校准 | center 用 train 目标均值；α=0 是对照不是终态 |
| `damage_shrink(a, w)` | 用因果退化指标做曲线校正 | `fit=train`，不得用验证标签 |
| `continue_train_staged` | 长阶段+低 lr 精修 | 只用官方 train；保持结构与归一化 |
| `ridge_last_layer` | 只换输出头，诊断表征瓶颈 | 只用 train 特征与标签 |
| report 切分 | 最终报告 | 不参与任何选择 |

## 如何验证

- **分组留出**：按单元/轴承留出，报告 mean±sd；不要把同一轨迹的行分到两边。
- **基线对照**：与常数预测（训练均值）比较；B 官方基线 681 分钟很弱，常数约 536 分钟，因此提升必须同时报告相对常数。
- **残差分桶表**：低/中/高 RUL 段分别报告 RMSE 与 bias，确认方法不是只改善某一桶。
- **退化段检查**：末期误差是否被放大（工业代价不对称），必要时看 NASA score 的对称化版本。
- **重放**：用 run 记录复算每轮 selection/report 指标，确认无泄漏。

## 数据需求

run-to-failure 轨迹与真值 RUL、单元/轴承 id、按组切分 manifest、训练目标均值。时序特征必须在同一条轨迹内按时间顺序构造（只能用 t 及以前信息），禁止把"已运行帧数"直接当特征（RUL=N-i 泄漏）。

## 实验证据（公开 + 本仓库）

- 本仓库严格官方口径：B RMSE selection 678.02→409.74（damage_shrink，report 664.0→424.1）；C RMSE selection 44.75→40.52（continue_train_staged 首轮 1000+400 epoch，之后 9 轮 retain）。
- 本仓库诚实记录：B 的变差对照显示"按轴承泛化极难"，常数预测约 536 分钟；因此只看相对官方基线会夸大收益。C 的 report 与 selection 差异大（57.46 vs 44.75），说明单元间方差显著。
- 公开：C-MAPSS 的 run-to-failure 仿真与 PHM08 评测协议（Saxena 2008）；相似性匹配 RUL（Wang 2008）；PRONOSTIA 轴承退化基准（Nectoux 2012）。

## 风险与不适用条件

- **轨迹泄漏**：同单元的行同时出现在训练与验证，指标会虚高。
- 用验证/报告标签拟合校准曲线 = 泄漏。
- α=0 或强收缩会把预测压成常数，虽然 RMSE 未必变差，但失去排序能力，工业上不可用。
- 不同测试单元的工况/寿命先验不同，全局 center 可能不适合所有单元。
- 未来信息（整个轨迹的统计量）进入特征会在部署时不可得。

## 来源

Saxena et al., PHM 2008；Wang et al., PHM 2008；Nectoux et al., IEEE PHM 2012；本仓库 docs/12_EXPERIMENTS.md §12.5、§12.19。
