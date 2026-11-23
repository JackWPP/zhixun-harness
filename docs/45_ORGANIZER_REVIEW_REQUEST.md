# 45 · 请求 organizer 核验材料与记录的沟通草稿

> 用途：最后一次正式评测前，请组委会核验 submission 上的
> `materials_complete` 与 `records_complete` 两个人工审核字段。

## 事实清单（可直接复制）

- 队伍：yxc200506（北京化工大学）
- 平台 submission：`2560f93aec8b5ba4e099ce86e96b4429`（v15，最终候选，0.14.12，valid，已选为正式版本）
- 镜像 archive：`1eae63b3990429855fd80fe4b38626e7`（381,050,880 B，ready）
- 材料文档：`6758ff3df369f201a1dfb05dbbed02e4`（`zhixun-rsi-harness-v0.14.12-final-materials.md`，含两次 formal 复盘、C retain 决策、记录字段说明）
- 最终正式评测 run：`1d679070f34badfdf0dd9a77086cdb96`（2026-11-23T07:02:29Z 创建，attempt 3）
- 仓库：<https://github.com/JackWPP/zhixun-harness>，tag `v0.14.12`
- SwanLab：<https://swanlab.cn/@wppjkwbuct/zhixun-rsi>
- 两次正式评测 run：`1b0e7803…`（45.74）、`ee8cb5c7…`（45.18）
- 隐藏基线探针 trial：`d401a95a…`（completed，无违规）
- 最终配置 trial：`05a5d838…`（0.14.11，A 0.894721 / B 155.17 / C 55.89）
- 最终候选 0.14.12：C 改为 retain（隐藏基线 56.2259/NASA 4.26e10，与静态基线持平），A/B 不变

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
