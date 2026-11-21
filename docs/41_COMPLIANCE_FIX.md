# 41. 平台合规判定修复（0.14.5）：移除训练内拟合的 pair 阈值

## 1. 平台判定

0.14.4 的试运行（run `44afba85…`）被判定 `failure_kind=contestant`，原因：
原生入口（`python3 /app/harness.py`，SDK 1.0.4）本身可用，describe / train / predict 签名正确，
A 父模型验证 macro-F1=0.71403、B/C 预测正常；但**训练策略违反续训数据边界**：

- `submission.py::_candidate_plans(A)` 注入 `postprocess="FIT_PAIR"`；
- 执行链调用 `fit_pair_threshold(x_train, y_train, model, pair=(0, 3))`（`inference.py`），
  在训练标签上做网格搜索最大化含标签正确率，提升 >0.02 才返回阈值；
- 返回的 delta 随最终 checkpoint 的 `submission_meta.json` 一起发布（`code_paths=[META_NAME]`，不再刷新），
  predict 的 `PostprocessStream(pair_threshold)` 逐行应用该阈值；
- 判定：属于用父/新揭示标注拟合推理期统计，而非纯推断校准，会随修复潜藏流代入隐藏评测。

修复建议（平台原文）：用历史校准 delta，或仅 train 后处理并在 predict 用固定 delta。

## 2. 修复（0.14.5）

| 位置 | 改动 |
|---|---|
| `rsi/platform/submission.py::_candidate_plans` | 删除 `postprocess="FIT_PAIR"` 候选；A 只保留 `pair_history`（固定 delta，历史先验） |
| `rsi/platform/submission.py` 评估循环 | 删除 `FIT_PAIR` 分支及 `fit_pair_threshold(x_train,y_train,...)` 调用 |
| `rsi/platform/plans.py::_resolve_postprocess` | `pair_threshold` 必须带显式 `delta`；缺省直接降级 `none`，**永不拟合** |
| `rsi/platform/plans.py::build_plans` | 移除 `delta=None` 的 pair 候选，只保留 `pair_history` 固定 delta |
| `rsi/inference.py::resolve_postprocess` | 移除 calibration 拟合分支；`pair_threshold` 缺 `delta` 直接报错（不再用任何标签拟合） |
| predict | 继续使用固定历史 delta（`PAIR_HISTORY_DELTA=-4.0`），不读取任何标签 |

保留的合规能力：
- 累积标签**训练**（`continue_train` / `continue_train_pair`，用 `/data/train` 已揭示标签）——训练数据边界内；
- recent-block holdout 选择（用留出的新鲜已揭示块做 gate）——不写回 predict 产物，只用于选 checkpoint；
- B/C 的 `shrink`（train target_mean）与 `damage_shrink`（train 标签）——平台认定合规。

说明：`pair_threshold` 不再是“拟合出的推理期统计”，而是**固定先验阈值**（历史 campaign 校准值），
与 predict 代码一起固定发布；`fit_pair_threshold` 仅保留在离线研究脚本/内部 stream 模拟器中，不在平台路径。

## 3. 验证

- 全量测试 **207 passed, 3 deselected**；
- 新增/更新回归测试：`tests/test_sim_protocol.py::test_v2_adapter_has_pair_history_candidate_for_task_a`
  （断言 A 候选含 `pair_history`、不含 `pair_threshold` 拟合臂，且 delta 为固定值）；
- 本地 official-seq A smoke：见 §4；
- 镜像 0.14.5：构建 + `scripts/verify_image.sh` PASS。

## 4. 结果

- A official-seq smoke（seed 3，8 轮，真实 PyTRiO，kernel+固定 delta）：
  逐轮 hidden [0.67, 0.64, 0.82, 0.61, 0.53, 0.91, 0.89, 0.70]，**均值 .7203**，
  选择 continue_train / continue_train_pair（固定 delta），与修复前 11 seeds 均值 .7192 持平
  → 去掉标签拟合阈值没有损失 A 的收益。
- 新的平台试运行：见 §6（修复后重新提交）。

## 5. 边界

- 本次修复只去掉“标签拟合阈值”这一条违规，不改变训练/选择/预测的其它逻辑；
- 平台判定为 `contestant`，属于提交内容问题而非平台故障，因此需要重新构建镜像并重新提交；
- 试运行额度仍为 7 次（已用 3/10），正式评测 2 次未动。
