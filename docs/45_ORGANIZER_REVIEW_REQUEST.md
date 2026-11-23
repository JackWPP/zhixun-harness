# 45 · 请求 organizer 核验材料与记录的沟通草稿

> 用途：最后一次正式评测前，请组委会核验 submission 上的
> `materials_complete` 与 `records_complete` 两个人工审核字段。

## 事实清单（可直接复制）

- 队伍：yxc200506（北京化工大学）
- 平台 submission：`304427233e4ce497ff6ac536f824d285`（最终候选，0.14.11）
- 镜像 archive：`ce8232a438c860e925b6efd82ebe9a8d`（381,091,840 B，ready）
- 材料文档：`6363516651f9866567987f843de27fa6`（`zhixun-rsi-harness-v0.14.11-materials.md`）
- 仓库：<https://github.com/JackWPP/zhixun-harness>，tag `v0.14.11`
- SwanLab：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi>
- 两次正式评测 run：`1b0e7803…`（45.74）、`ee8cb5c7…`（45.18）
- 隐藏基线探针 trial：`d401a95a…`（completed，无违规）
- 最终配置 trial：`05a5d838…`（0.14.11，见后续报告）

## 请求内容

1. 请核验该 submission 的 **实验记录完整性**（10 轮 × A/B/C 的
   hypothesis / change_summary / decision / parent+candidate sha /
   validation_metrics / history_ids / model_call_ids 是否满足规范）。
2. 请核验 **材料完整性与可复现性**（镜像、文档、仓库、SwanLab、复现命令）。
3. 如 `not_below_static_baseline` 是按隐藏流计算，请告知其计算口径
   （stage 粒度/数据集粒度、baseline 包版本），我们会据此对齐最终候选。

## 背景（一句话）

两次正式评测的客观分损失主要来自 `mean_gain`/`worst_stage`/`resource_efficiency`
三个计分项与上述三个人工 gate；代码侧的合规问题（FIT_PAIR、验证集入梯度、
gate 方向）已在 v0.14.6–v0.14.10 修复，并有一次隐藏基线探针实测（C 父模型
隐藏 RMSE 56.23，我们的候选 55.89）。
