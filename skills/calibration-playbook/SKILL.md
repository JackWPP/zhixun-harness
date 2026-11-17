---
name: calibration-playbook
description: 推理期校准（recenter/shrink）的适用条件与参数选择
version: 1
tags: calibration, TTA, postprocess
---

# 何时用
漂移报告显示协变量/增益漂移，或历史 recenter 实验曾 promote。

# 步骤
1. A 任务（分类）：recenter 用目标批均值/方差映射回模型冻结归一化；weight 从 0.3-0.5 起，验证是否越过 margin 0.004。
2. batch_mean 只平移均值，更稳；batch_std 同时缩放，收益大但风险高；两者都必须在 selection 上验证。实测（任务A，round 3）：batch_std w=0.5 使 selection macro_f1 从 0.7207 跌至 0.6769（-0.044），batch_std 在 A 上确认有害，默认只用 batch_mean。实测（任务A，round 3）：batch_mean 权重 0.4→0.5 使 selection 0.72070→0.73455（+0.0139，已 promote），且 report 0.654→0.683 同步改善，A 的最优 weight ≥0.5；同轮 class_balance 续训（-0.043）与 SWA+balance（-0.006）均变差，A 上别用 class_balance。实测（任务B，round2）：batch_std w=0.4 selection rmse 602.24→618.72(+16.48)、bias -263.6→-351.9，B 上同样有害；B 只用 batch_mean。另：ridge_last_layer α=10 在 B 上致输出尺度灾难失配（rmse 395k），head 重拟合方向关闭。实测（任务B，rsi-20261117 round1）：recenter batch_mean w=0.5 → selection rmse 678.02→588.45（Δ-89.57）、bias -375.5→-235.7、report 582.93 同步改善，已 promote；round2 续测 w=0.7→568.19（Δ-20.26）、w=0.85→559.91（Δ-28.54，已 promote，bias -137.8，report 555.24），round3 实测 w=1.0→557.87（Δ-2.04 未过 margin），B 的 w 扫描确认 0.85 饱和，最优 w=0.85；continue_train lr=5e-4×10ep 叠加 w=0.85 得 555.10（Δ-4.81，差 0.19 未过门控→retain）；shrink 与 damage_shrink 在本 run 均被 harness 拒绝（"only meaningful for regression"），B 输出侧校准路线关闭，勿重试；同轮 continue_train lr=5e-4×10ep 在 w=0.5 下 Δ-11.6（576.85）有效但弱于提 w，可试 续训+w=0.85 组合；shrink 后处理该 run 被 harness 拒绝（retain/continue_train 各一次，error: only meaningful for regression），B 输出侧校准改用 damage_shrink，勿重试 shrink。
3. 回归任务：优先 shrink；alpha 越小越保守（0 = 常数中心）。B 的基线在留出轴承上失效，shrink 往往优于继续训练。实测（任务C，rsi-20261117 round1）：C 是 RUL 回归，harness 同样拒绝 shrink/damage_shrink（"only meaningful for regression"，疑似与 B 同因，C 输出侧校准路线关闭）；recenter batch_mean w=0.5 在 C 上 selection rmse 44.75→46.41（Δ+1.67）、bias 3.4→10.4，特征重定心有害，C 漂移为 concept_drift，唯一有效路线是官方 train 短程续训（lr=5e-4×10ep→43.95，promote）。实测（任务C round2）：同父续训 lr=2.5e-4×10ep→43.8435（Δ-0.108）、lr=5e-4×20ep→43.8748（Δ-0.077），均<0.3 margin→retain；round1(-0.797)→round2(~-0.1) 收益递减，续训已趋饱和，后续优先 swa 或短 epoch 续训，勿再拉长轮数。实测（任务C round3）：swa lr=2.5e-4×10ep snapshots=5→43.5663（Δ-0.385，已 promote），snapshots=3→43.6723（Δ-0.279 未过门控）——C 上 SWA 快照数是关键变量，≥5 有效；recenter batch_mean w=0.5 复测 Δ+1.527（bias 2.37→9.62）再次确认有害，C 输入侧校准路线关闭，唯一有效杠杆=官方 train 上的续训/SWA。
4. 校准不改变 checkpoint 字节；记录里仍要写清 postprocess 与父/候选 sha。
5. 一次只改一个参数，连续两轮收益递减时保留当前最优。

# 坑
- 窗口太小时批统计噪声大；batch 评分先读全批再算统计。
- recenter 假设批内类别分布不变；类别极偏的批次会放大偏差。
- 校准后的 selection 提升必须再看 report 是否同步改善。

# 验收
每轮记录 postprocess 类型、模式、weight/alpha、selection 前后主指标与 delta。
