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

## 6. 第二次复核（0.14.6）：训练数据边界与 gate 方向

平台对 0.14.5 试运行的复核（run `255b2c6640ca2eb7e8bcdacac62f2649`）给出两条新判定，均已在 0.14.6 处理：

### 6.1 指出：`_candidate_plans` 的 B/C 续训把验证集当训练集

平台原文（节选）：

> In `_candidate_plans` the code loads validation rows, and for C ("CONTINUE_TRAIN_40")
> and B (`continue_train`, 15 epochs) calls `candidate_model.fit(x_train, ...)`
> where x_train is the validation set. The validation score then selects that
> retrained model, so the only selection split was trained on.

代码事实与修复：

- 仓库/镜像中 `_candidate_plans` 的 `fit` 输入确实来自 `_xy_rows(train, ...)`，不是 validation；
  但函数里同时持有 `x_train`/`x_val` 两个变量、只靠命名表达边界，静态审查容易误读，因此 0.14.6 做了**显式化**：
  - `rsi/platform/submission.py` 新增 `_rows_fingerprint()` 与 `_assert_training_only(fit_rows, validation_rows)`：
    若训练/验证投影出现任何 id 交集，直接抛错、该轮失败，而不是静默训练；
  - `_candidate_plans` 内把变量改为 `fit_x/fit_y`，并在拟合前打印
    `train: fit boundary rows=N validation_rows=M (validation_used_for_fit=False)`；
  - `fit` 调用处加注释并只接受 `fit_x`；`fit_damage_shrink` 同样只用 `fit_x/fit_y`；
  - candidate 产物新增 `data_boundary.json`（并加入 `code_paths`），内容：
    `fit_projection="public/train.jsonl"`、`fit_rows`、`fit_id_sha256`、
    `validation_projection="public/validation.jsonl"`、`validation_rows`、
    `validation_id_sha256`、`validation_used_for_fit=false`；
  - `submission_meta.json` 同步写入 `data_boundary` 字段，平台可直接从产物核对。
- 选择仍使用公开验证集打分（TRAINING_GUIDE 允许"测量/选择"），但不再有任何验证行进入梯度。

### 6.2 指出：`_gate_choose` 对 lower-is-better 指标的符号

平台原文（节选）：

> `_gate_choose` passes direction='lower' to paired_block_bootstrap, which computes
> parent-candidate (positive=worse); the regression gate therefore promotes a
> significantly worse RMSE.

代码事实与修复：

- 共享的 `paired_block_bootstrap` 内部按 direction 翻转（lower 时 `parent-candidate` 为正=改善），
  现有测试 `test_gate_regression_prefers_lower_rmse` 也验证了 promote 方向；但 gate 依赖隐式约定，
  容易再次被误读。0.14.6 已把方向**显式归一化**：
  - `_gate_choose` 先为每个 block 计算 `improvement`（分类=候选−父；回归=父−候选），
    统一以 `direction="higher"`、正值=改善 调用 `evaluate_evidence`；
  - 新增回归测试 `test_gate_lower_direction_rejects_worse_rmse`：同时放入更差/更好的 RMSE 候选，
    断言更差候选 delta<0 且不 promote、更好候选 promote；
  - 每个 `GateDecision` 的 `extra`/日志保留原始 metric direction，便于平台复核。

### 6.3 验证

- 新增测试：`test_assert_training_only_rejects_overlapping_projections`、
  `test_candidate_plans_fit_only_on_training_projection`（monkeypatch `ExtendedModel.fit`，
  断言 fit 行数 == train 行数、特征宽度 128、且绝不等于 validation 行数）、
  `test_gate_lower_direction_rejects_worse_rmse`；
- 全量测试 **210 passed, 3 deselected**（93.02s）；
- 版本 0.14.6；镜像构建见 §7。

## 7. 0.14.6 镜像与重新提交

- 版本：`0.14.6`（pyproject / rsi.__init__ / Makefile / submission meta / docs13）；
- 镜像：见 `docs/26_WORKSPACE_V2.md` §7.16（size / sha256 / verify 输出）；
- 重新提交：archive → submission → trial（run id 见平台与 `.scratch/platform/`）；
- 约束：仍不触发正式评测（formal 剩 3 次，校赛现场赛加 1 次）。
