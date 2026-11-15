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
2. batch_mean 只平移均值，更稳；batch_std 同时缩放，收益大但风险高；两者都必须在 selection 上验证。实测（任务A，round 3）：batch_std w=0.5 使 selection macro_f1 从 0.7207 跌至 0.6769（-0.044），batch_std 在 A 上确认有害，默认只用 batch_mean。实测（任务B，round2）：batch_std w=0.4 selection rmse 602.24→618.72(+16.48)、bias -263.6→-351.9，B 上同样有害；B 只用 batch_mean。另：ridge_last_layer α=10 在 B 上致输出尺度灾难失配（rmse 395k），head 重拟合方向关闭。
3. 回归任务：优先 shrink；alpha 越小越保守（0 = 常数中心）。B 的基线在留出轴承上失效，shrink 往往优于继续训练。
4. 校准不改变 checkpoint 字节；记录里仍要写清 postprocess 与父/候选 sha。
5. 一次只改一个参数，连续两轮收益递减时保留当前最优。

# 坑
- 窗口太小时批统计噪声大；batch 评分先读全批再算统计。
- recenter 假设批内类别分布不变；类别极偏的批次会放大偏差。
- 校准后的 selection 提升必须再看 report 是否同步改善。

# 验收
每轮记录 postprocess 类型、模式、weight/alpha、selection 前后主指标与 delta。
