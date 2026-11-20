# 26 · Workspace v2 接入（contest.harness.workspace.v2）

> 适用版本：harness **0.9.0**。平台 `GET /api/v1/rules` 的 `execution_protocol` 已从
> `contest.harness.workspace.v1` 升级为 **`contest.harness.workspace.v2`**；镜像若仍只有
> 自研 CLI（`run/stream/predict`），试运行诊断会报
> “镜像还需要实现 start/candidate/save_state/complete”。本文记录 v2 契约、我们的接入实现与本地验证方法。

## 1. 规范来源

- 平台前端 bundle（`.scratch/platform/index-new.js`）内嵌《选手 Harness SDK 使用说明》，并给出下载地址
  `/downloads/contest-harness-sdk-1.0.4.tar.gz`。
- 已下载并校验：`sha256:c676c8beeef69cece9cbec5a633d05dbb9832eb22b6f761913f2315bd1f85133`
  （与平台元数据一致）。仓库 `vendor/contest_harness-1.0.4/` 保存该 SDK 源码副本用于离线构建，
  升级 SDK 时替换目录并同步本文件。
- 固定入口：镜像内 `python3 /app/harness.py describe|train|predict`，选手模块
  `/app/contest_submission.py`（可用 `CONTEST_SUBMISSION` 改名），导出 `train(rnd)`、`predict(sample, model_dir)`、`TASKS`。

## 2. v2 契约摘要

| 项 | 约定 |
|---|---|
| describe | 一行 JSON：`{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.4","tasks":[...]}` |
| train | stdin 一帧 `start`；进程内通过 JSONL RPC 与宿主交互（history/model/candidate/state/artifact_read） |
| round 生命周期 | `Round.start` → 读 `/data/train.jsonl`、`/data/validation.jsonl` → `rnd.model(...)`（宿主附加官方 prompt）→ `submit_candidate(dir, …)` → `save_state(dir)` → `finish(selected_artifact_id, …, model_call_ids=…)` → 发 `round_complete` 并等 `round_ack` |
| 父产物 | `/previous` 只读；`rnd.parent.path("checkpoint.npz")` 会核对清单摘要，不匹配抛 `PARENT_DIGEST_MISMATCH` |
| 候选产物 | 目录 ≤64 文件、≤64 MiB；`checkpoint_paths`/`code_paths`/`predict_argv` 必须声明；`hypothesis`/`change_summary` ≤4000 B；`configuration` ≤64 KiB |
| 预测 | 平台只挂载 `/model`，env 提供 `CONTEST_ARTIFACT_PATH`（选中 checkpoint 文件）、`CONTEST_ARTIFACT_MANIFEST`、`CONTEST_PREDICTION_CONTEXT`（`contest.predict-context.v1`）、`CONTEST_DATASET_KEY`、`CONTEST_TASK` |
| 预测输出 | 每行请求 `{"task","inputs"}`；分类 `{"outputs":[{"label","score"}]}`（label 唯一、最高分唯一、score∈[0,1]），回归 `{"outputs":{"value":nonneg}}`；每行写完即 flush，缺行/EOF/超时会使测量失败 |
| 日志 | stdout 只允许协议 JSONL；训练日志一律 stderr（SDK `isolate_stdio` 已处理） |

## 3. 我们的实现

- `/app/harness.py`：自写 bootstrap（先把 `/opt/zhixun/site-packages`、`/opt/zhixun/app` 插入 `sys.path`，再调用
  SDK 的 `contest_harness.cli.main`）。镜像 `ENTRYPOINT` 收到 `describe/train/predict` 时直接 `exec python3 /app/harness.py "$@"`。
- `/app/contest_submission.py`：薄壳，`from rsi.platform.submission import TASKS, train, predict`。
- `rsi/platform/submission.py`：真正的 RSI 适配层。
  - `train(rnd)`：加载父 checkpoint → 读公开投影并按 `meta["paths"]` 逐维 flatten → 评估父模型 → 构造候选
    （分类：`retain` / 推理期 recenter；B：`retain` / train 均值 shrink / damage_shrink；C：`retain` / 官方 train 短续训 40 epoch）→
    公开验证集门控（分类 +0.004、B −5 RMSE、C −0.3 RMSE 才采纳）→ 一次 `rnd.model(...)` 让模型在候选中选择并给出假设 →
  `submit_candidate`（`checkpoint.npz` + `submission_meta.json`）→ `save_state` → `finish`（引用真实 `call_id` 与 history id）。
  - `predict(sample, model_dir)`：从 `CONTEST_ARTIFACT_PATH` 加载 checkpoint 与同目录 `submission_meta.json`；
    分类逐行用 `RecenterStream`（因果窗口）或 pair 阈值；回归用 shrink / `DamageShrinkStream`；输出严格符合 SDK 校验。
  - 合规边界：训练只用 `public/train.jsonl`；验证只用于评估与门控；从未把 `answer` 列喂给模型；产物不写 `/app`、不联网。
- 镜像构建：`docker/requirements.txt` 不含 SDK（纯 Python），`vendor/contest_harness-1.0.4/contest_harness` 被拷到
  `/opt/zhixun/app/contest_harness`；`docker/app/*` 被拷到镜像 `/app/`（`scripts/stage_image.sh` 同步支持无 daemon 构建）。

## 4. 本地验证

```bash
# 1) describe（等价于平台诊断调用的入口）
PYTHONPATH="$PWD:$PWD/vendor/contest_harness-1.0.4:$PWD/docker/app" \
  ./.venv/bin/python docker/app/harness.py describe

# 2) 模拟宿主完整一轮（A：train → 候选上传 → state → predict 三条）
./.venv/bin/python scripts/emulate_workspace_v2.py --task A --rounds 1

# 3) 多轮父谱系：round 2 的父产物是 round 1 的 artifact
./.venv/bin/python scripts/emulate_workspace_v2.py --task A --rounds 2

# 4) 回归任务（C 会走 40 epoch 续训候选）
./.venv/bin/python scripts/emulate_workspace_v2.py --task C --rounds 1

# 5) 测试
./.venv/bin/python -m pytest tests/test_platform_v2.py -q
```

镜像级验证（解包 docker-save tar + bwrap）：

```bash
bash scripts/verify_image.sh .scratch/image/harness.tar
# 其中包含：
#   [verify] workspace v2 describe (entrypoint dispatch)
#   [verify] workspace v2 fake-host round (A, train+predict)
```

## 5. 与旧入口的关系

- 旧的 `describe/run/stream/predict` 自研 CLI 全部保留，用于本地 RSI 运维、Track X 测试床与镜像内彩排；
  平台评测路径只通过 `/app/harness.py`（v2）。
- 若平台后续收回 v2 或要求 adapter 改名，只需替换 `vendor/` 中的 SDK 与 `/app/harness.py`；`rsi/platform/submission.py`
  与业务代码不变。

## 6. 已知边界

- 模型网关走宿主 `host_gateway_rpc`：每轮恰好一次 `rnd.model`，prompt 由宿主附加官方提示；本适配层不读取/不写 token。
- `predict` 的 recenter/damage_shrink 是因果流式近似（窗口 256 / 分段检测），与批式 `apply_postprocess` 的差异在
  `docs/12` §12.23 记录；若平台以乱序流调用，damage_shrink 会自动退化为纯 shrink。
- 候选上传体积远低于 64 MiB（A/B/C checkpoint 均 <60 KiB）；`submission_meta.json` 只含后处理参数与统计，不含原始数据。

## 7. 2026-11-17 新版《选手 Harness SDK 使用说明》对照审计

> 来源：用户 2026-11-17 提供的新版说明（`contest-harness-sdk-1.0.2 (1).txt`，11,864 B，文字与
> `vendor/contest_harness-1.0.4/HARNESS_SDK.md` 基本一致，新增了“提交里要有什么 / 目录权限 / 公开数据 /
> 父模型 / 一轮训练 / 预测 / 本地检查”等操作细节）。逐项对照结果如下（PASS=无需改动；
> 已修=本轮发现并修复；注意=边界说明）。

| # | 官方要求 | 我们的实现 | 结论 |
|---|---|---|---|
| 1 | 镜像内安装 SDK，`/app/harness.py` 固定入口，选手模块默认 `/app/contest_submission.py` | `docker/app/harness.py` bootstrap 插入 `/opt/zhixun/{site-packages,app}` 后调 `contest_harness.cli.main`；`docker/Dockerfile` 拷 vendor SDK + 两个 app 文件；`entrypoint.sh` 对 `describe/train/predict` 直接 `exec python3 /app/harness.py "$@"` | PASS |
| 2 | `describe` 向 stdout 打印一行 JSON，至少含 protocol/model_access/tasks；会 import 模块并检查 `train`/`predict`/任务声明 | 本地 `harness.py describe` 输出 `{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.4","tasks":["classification","regression"]}`；`TASKS` 恰好是两类 | PASS（`tests/test_platform_v2.py::test_describe_speaks_workspace_v2`） |
| 3 | `tasks` 只能是镜像实际支持的分类/回归 | 两类都由同一份 `train/predict` 支持（A 分类，B/C 回归） | PASS |
| 4 | `/app` 只读，不在其下建目录/改文件；不假设根目录可写 | 所有写操作在 `/workspace`；镜像 `PYTHONDONTWRITEBYTECODE=1`；bootstrap 只 `sys.path.insert` | PASS |
| 5 | `/data/*` 是“当前阶段公开投影”，行字段 `id/inputs/answer`，训练/验证逐行读 | `train(rnd)` 只用 `rnd.data.training_rows()/validation_rows()`，`_xy_rows` 逐行 flatten；从不读 `answer` 之外的文件名假设 | PASS |
| 6 | `/previous` 是父产物；用 `rnd.parent.path(rel)` 并按清单校验摘要，不搜索 `/app` 找权重、不伪造哈希 | `parent_path = rnd.parent.path("checkpoint.npz")`；`_state_payload` 记 `rnd.parent.checkpoint_sha256`；不写 `baseline-info.json` | PASS |
| 7 | 一轮字段：`round_index` 从 1 起（不是只能训练一次）、`task` 以合同为准、`max_output_tokens`、`limits` | 用 `int(rnd.round_index)`；`rnd.task` 经 `_contract_value` 兼容 `task/task_type/type` 与 `metric/metric_name`、`direction/metric_direction`；`max_tokens=min(256, rnd.max_output_tokens)` | 已修（原实现假设 `rnd.task["task"]` 恰好存在） |
| 8 | 官方 prompt 由宿主附加到第一条 system；不要把 `rnd.prompt` 再塞进 system/user | 只发一条 user 消息（候选表 + 选择指令），不读/不重发 `rnd.prompt` | PASS |
| 9 | 每轮至少一次真实 `rnd.model` 调用；`model_call_ids` 非空且来自真实调用 | 每轮恰好一次 `rnd.model(...)`；`finish(..., model_call_ids=[receipt.call_id])` 显式引用 | 已修（原实现依赖 SDK 默认 `call_ids`，行为等价但显式更稳） |
| 10 | `rnd.history()` 引用实际读过的 id；记录可为空 | 读一次 `rnd.history()`，`finish(history_ids=[...])`；异常时传 `[]` | PASS |
| 11 | 候选：`/workspace` 新目录、≤64 文件/≤64 MiB、至少一个 checkpoint；`hypothesis/change_summary` 非空且 ≤4000 B；`configuration` 对象 ≤64 KiB；`seed` 整数；`predict_argv` 原样可执行 | `tempfile.mkdtemp(dir=/workspace)`；2 个文件（`checkpoint.npz`+`submission_meta.json`），checkpoint 均 <60 KiB；h/c 短文本；config 仅候选摘要；`seed=20261117`；argv `["python3","/app/harness.py","predict"]` | PASS（新增 `_change_summary` 让 retain/续训/后处理三种情况的说明如实区分） |
| 12 | state：先复制 `/state` 再写；`save_state(dir)`；不保存则沿用上轮 | `_new_state_dir` 优先 `rnd.state_workspace(...)`（复制上轮原生状态），失败时回退新建目录；本轮再写 `state.json` 后 `save_state` | 已修（原实现只写新目录，丢上轮 state） |
| 13 | 结束：`finish(selected_artifact_id, hypothesis, change_summary, reason, configuration, seed, history_ids, model_call_ids)`，reason 非空；收到 ack 后退出 | `finish(...)` 参数齐全，reason 含指标与真实 call_id；SDK 负责 ack/退出 | PASS |
| 14 | stdout 只写协议 JSONL，日志走 stderr，不要等 EOF 才输出 | 适配层无 `print`；SDK `isolate_stdio()` 在 import 前把 stdout 重定向到 stderr；所有输出是逐行协议帧 | PASS |
| 15 | predict：`model_dir` 是 `/model` 根；`CONTEST_ARTIFACT_PATH` 是选中 checkpoint 文件；上下文协议 `contest.predict-context.v1`；样本 `task` 为 classification/regression | `_artifact_path` 优先 `CONTEST_ARTIFACT_PATH`，回退 `/model/checkpoint.npz|model.npz|*.npz`；meta 从 checkpoint 同目录读；`sample.task` 与 artifact task 不同则报错 | PASS |
| 16 | 分类输出：label 非空唯一、score∈[0,1] 有限、最高分唯一；回归：`{"value": 非负有限}`；逐行 flush | `_uniform_scores` 归一化/裁剪到 [0,1] 并**强制唯一最高分**（本轮修复并列 top 时可能违反“最高分唯一”的边角）；回归 clamp 到非负数；SDK 逐行 `write+flush` | 已修（并列最高分） |
| 17 | 不得把 PyTRiO Key/外部采样客户端放进镜像、源码、文档、训练记录 | Key 只在 `.scratch/llm.env`（gitignore、不入镜像）；镜像 `COPY` 列表不含 `.scratch`；仓库 `git grep 'sk-bceph'` 无命中；提交时 Key 只填平台密钥栏 | PASS |
| 18 | 旧 `contest_sdk` HTTP 客户端（`/runtime/v1/runs/...`）与 v2 不同 | 我们完全走 v2 `contest_harness` 1.0.4，未用旧客户端 | PASS |

### 7.1 本轮实际执行的验证

```bash
# 官方 SDK 自带离线协议测试（15 项）
cd vendor/contest_harness-1.0.4 && ../../.venv/bin/python -m unittest \
  tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
# => Ran 15 tests OK

# 本地宿主替身（完整 v2 帧/RPC：train → candidate → state → predict）
.venv/bin/python scripts/emulate_workspace_v2.py --task A --rounds 2 --json   # PASS（含第 2 轮父谱系）
.venv/bin/python scripts/emulate_workspace_v2.py --task B --rounds 1 --json   # PASS（回归 shrink/damage）
.venv/bin/python scripts/emulate_workspace_v2.py --task C --rounds 1 --json   # PASS（回归 40 epoch 续训）

# 适配层契约测试
.venv/bin/python -m pytest tests/test_platform_v2.py tests/test_submission.py -q
```

### 7.2 遗留说明与下一步

- **“数据注入”措辞**：新版说明仍未出现“注入”二字；但 `/data` 明确是“当前阶段公开投影”，`round_index`
  从 1 开始、`history()` 返回已完成检查点、`parent_artifact` 形成续训链，加上规则里的“漂移数据流/隐藏测试流”
  与答疑逐字稿（批次下发、每批带标签、先交判断再拿下一批），可确认平台按轮次投放数据。即使某数据集只跑
  1 轮，本适配层也按“父模型 + 当前公开投影”工作，不会假设一定有外部/隐藏数据进来。
- **trial 镜像版本**：平台正在跑的 trial 用的是 0.10.1（本轮修复前）。本轮修复不改变协议骨架，主要是
  contract 字段容错、state 续接、`model_call_ids` 显式化与分类并列 top 打破；若 trial 通过，正式提交前
  建议重建 0.10.2 镜像并重新上传（需用户确认）。
- 分类 score 不强制归一化到和为 1（协议只要求 [0,1] 且最高分唯一）；若平台后续把 score 当概率做阈值，
  再评估是否 softmax 归一化。

### 7.3 v0.10.2 镜像（本轮修复 + 重建）

- 版本冻结点：`pyproject.toml` / `rsi/__init__.py` / `submission_meta.json` 均写到 **0.10.2**。
- 镜像重建（无 daemon 路径）：`bash scripts/stage_image.sh` → `scripts/build_image_tar.py --tag zhixun-rsi-harness:0.10.2`
  → `bash scripts/verify_image.sh`，**verify PASS**（selftest A 数字不变：selection 0.703220 / report 0.686380 /
  calibration 0.763582 / probe 0.456803；describe 输出正确；v2 fake-host 一轮 PASS；离线 A 两轮
  workspace_report metric=0.7345546845546845）。
- 产物：`.scratch/image/harness.tar`，**378,224,640 B**，sha256
  `e1e6133ebb1c6c47d83fa5b6174d782eaaa5e863fe7094309d55ff18dbab30f9`（tags `0.10.2/latest`）。
- 该镜像尚未上传平台；正式提交/重跑前需用户确认（trial 仍在 1/10 额度内，0.10.1 那次 running）。

### 7.4 v0.10.3 韧性修复（平台 trial 失败后的第一轮加固）

平台 trial（0.10.1 镜像）返回 `status=failed / failure_kind=unknown`，`failure_detail` 在 1000 字符处被截断，
只能看到 Preflight 与训练侧已通过；平台不提供更细日志（仅 `/runs/{id}/report` 有效，无 logs/events 端点）。
因此这一轮按“最可能被隐藏数据打穿的地方”做防御性加固，再重建镜像验证：

- **网关调用重试**：`_gateway_model()` 对 `rnd.model(...)` 做 3 次有界重试（0.5s/1.0s 退避）；
  每轮至少一次真实调用，取最后一次成功 receipt 的 `call_id` 上报。避免一次瞬时网关错误直接判整轮失败。
- **单行预测兜底**：`predict()` 外层捕获单行异常（键缺失/类型异常/NaN 等），回退到
  `_fallback_prediction()`——分类输出各类均分（`_uniform_scores` 保证唯一最高分、score∈[0,1]），
  回归输出 `target_mean`（非负有限），并写一行 stderr 说明。坏一行不再拖垮整条隐藏流。
- **本地 30 轮复跑暴露的 emulator 问题（SIM-010）已合并**：`scripts/emulate_workspace_v2.py` 增加
  `install_previous()`，每轮把选中候选复制为下一轮 `/previous`；此前 C 任务（continue_train）第二轮
  因父挂载仍是 baseline 而误报 `PARENT_DIGEST_MISMATCH`，A/B（retain）不受影响。修复后 A/B/C 各 10 轮
  v2 复跑 **PASS**（A 4s / B 6s / C 24s）。
- 测试：新增 `test_predict_fallback_keeps_stream_valid`、`test_gateway_call_retries_transient_errors`；
  `tests/test_platform_v2.py` **11 passed**；全量 `pytest -m "not integration"` 140 passed / 3 deselected。
- 版本 0.10.3；镜像重建与 verify 结果见 §7.5（本轮上传/重跑 trial 仍需用户确认）。

### 7.5 0.10.4：平台完整 review 揭示的 B 任务契约冲突（2026-11-18）

平台把 trial#1 的完整失败详情放在 `/runs?include_resolved=true` 的复审记录里（run `9b7e19ef202d36fecf8fcde7e7a1efb1`，`failure_kind=contestant`，857 字）：A/C 通过、
提交入口原生、每轮模型调用真实；**唯独 B 数据集“坏了”**——冻结契约声明
`task=classification / macro_f1`，而 B 的官方包实际是回归（RUL 分钟、RMSE）。
我们此前严格跟随 `rnd.task`，于是 B 第一轮在 `checkpoint task mismatch: regression != classification`
处直接抛错、**未发生任何模型调用**，违反“每轮至少一次真实调用”的硬契约，导致整套配置无法通过。

修复（0.10.4）：
- `train()`：若宿主契约与 checkpoint task 冲突，**跟随 checkpoint task**（B→regression），
  同步 `metric/direction` 为默认值并在 stderr 打印警告；仅当 checkpoint task 自身非法才抛错。
  这样 B 每轮仍会发出真实 gateway 调用、产出候选、状态与 finish，协议完整。
- `predict()`：`_load_artifact` 纳入 try；失败→`_degenerate_prediction()`；
  行级异常→`_fallback_prediction()`；再经 `_coerce_task_output()` 兜住宿主请求 task 与产物 task
  不一致的形状（分类请求+回归产物 → normal/fault 两标签、0.9/0.1 唯一最高分；回归请求+分类产物 →
  最高分类别索引），保证隐藏流逐行都有合法输出。
- 回归测试：`test_train_follows_checkpoint_task_when_host_contract_disagrees`（fake rnd + 真实 B
  regression checkpoint：完成训练、1 次真实调用、finish call_id、meta task=regression）、
  `test_predict_coerces_task_mismatch_shapes`、`test_predict_on_regression_artifact_with_classification_request`。
- 验证：`tests/test_platform_v2.py` 13 passed；全量 **143 passed / 3 deselected**；
  `scripts/emulate_workspace_v2.py --task A/B/C --rounds 10` 全部 PASS。
- 版本 0.10.4；镜像重建与 verify 结果见 docs/13、docs/27。

### 7.4 v0.11.0 镜像（PI-Agent kernel 默认 + gen2 漂移策略）

- **版本冻结**：`pyproject.toml` / `rsi/__init__.py` / `submission_meta.json` / Makefile 均 **0.11.0**。
- **镜像内 env（新增）**：`RSI_V2_CONTROLLER=auto`、`RSI_V2_MAX_TURNS=6`。平台每轮默认走
  PI-Agent 式 kernel 路由（`rsi/platform/agent_host.py`），内核异常/无有效动作时自动回退 pipeline，
  仍保证每轮至少一次真实 `rnd.model` 调用。
- **gen2 漂移策略**：`rsi/platform/submission.py:_apply_drift_policy` —— 高漂移分类任务优先
  `pair_threshold(delta=-4.0)`（历史先验，不重拟合当前验证集），只要公开验证不劣于父模型 2*margin
  就采用；中漂移只用 recenter；低漂移 retain。规则同时写入 kernel brief 与 `plans.catalog_text()`。
- **重建（无 daemon 路径）**：`bash scripts/stage_image.sh` →
  `python scripts/build_image_tar.py --tag zhixun-rsi-harness:0.11.0` → `bash scripts/verify_image.sh` → **PASS**：
  - selftest A 数字不变：selection 0.703220 / report 0.686380 / calibration 0.763582 / probe 0.456803；
  - `describe` 输出 `contest.harness.workspace.v2` 正确；
  - v2 fake-host 一轮 PASS（日志可见 `train: drift_policy: high drift ...`）；
  - 离线 A 两轮 workspace_report metric = `0.7345546845546845`。
- **产物**：`.scratch/image/harness.tar` **380,149,760 B**，
  sha256 **`25c264746e2c4e95c87cefb646083e53c4311db4b4b2606061783e8ba2eec92d`**
  （config `887d4de605e0e92357e2d8cc33328615ecb3a40a983b7116887e6ff5a5edcfbe.json`，
  tags `0.11.0/latest`）。**尚未上传平台。**
- 全量测试：**150 passed, 3 deselected**。

### 7.5 漂移策略与实验开关（0.11.1）

- `RSI_V2_DRIFT_POLICY=1|0`：默认 1；高漂移 + 分类任务时，若历史校准
  `pair_history` 臂在公开验证上不劣于父模型（`+2·margin`），覆盖 pipeline/kernel 的
  当前选择；关闭时 `_apply_drift_policy` 直接返回原选择（用于消融）。
- `RSI_V2_PAIR_DELTA`：覆盖历史 pair 臂的阈值（默认 -4.0），用于 delta 敏感性实验。
- `RSI_V2_CONTROLLER=auto|kernel|pipeline`：默认取镜像 ENV（0.11.0+ kernel；0.11.2 起 auto：分类走 kernel、回归走 pipeline）；
  pipeline 每轮 1 次真实网关调用，kernel 每轮 6+ 次；两条路径都经过漂移策略。
- 实验证据与假设检验见 `docs/34_EXPERIMENT_REPORT.md`；全真模拟基线见
  `docs/32_FULL_SIMULATION.md`。

### 7.6 0.11.2：`auto` 控制器与平台同构复跑

trial #3 报告（0.10.4，kernel 全任务）显示协议全通过但隐藏批偏弱：A macro-F1 0.4768、B RMSE 735.07、C RMSE 56.12。
平台同构模拟（真实 PyTRiO Qwen3.8-27B，policy on，A/B/C 各 10 轮，kernel）得到 A **0.4802**、B **504.43**、C **42.45**，
与平台结论同量级；但实验矩阵证明回归任务用确定性 pipeline 更稳（B **283.52**、C **38.83**），而 A 的漂移收益来自
PI-agent kernel + 历史校准 pair_threshold。

因此 0.11.2 把镜像默认改为 `RSI_V2_CONTROLLER=auto`：分类任务走 kernel（每轮真实 `rnd.model` 调用 + drift policy），
回归任务走 pipeline（每轮一次真实网关调用，确定性候选评估）。策略开关 `RSI_V2_DRIFT_POLICY` 保持默认开。

镜像：`zhixun-rsi-harness:0.11.2`，tar 380,221,440 B，sha256 `b83b4eae5e7e7b0bf64fe8205257299805e8c30f21427d1b9c51eddcbb55f000`，
`scripts/verify_image.sh` PASS；测试 153 passed。平台侧未上传、未重跑 trial（待用户确认）。

### 7.7 0.12.1：SDK 1.0.4 升级与分块上传

平台于 2026-11-19 将 Harness SDK 从 1.0.2 升级到 **1.0.4**（下载包
`/downloads/contest-harness-sdk-1.0.4.tar.gz`，27,979 B），仓库 vendored 到
`vendor/contest_harness-1.0.4/`。与 1.0.2 相比：

- `describe` 的 `sdk_version` 变为 `1.0.4`（镜像内 SDK 版本必须与平台声明一致）。
- 候选/状态的 RPC 从“内联 files”改为**分块上传 chunked-v1**：
  `upload_begin_request` → 多次 `upload_chunk_request` → `upload_finish_request`，
  然后 candidate/state 请求只带 `upload_id`；宿主必须在 `resource_limits` 里声明
  `artifact_upload_protocol="chunked-v1"`，否则 SDK 直接报 `CHUNKED_UPLOAD_UNSUPPORTED`
  （不会退回旧内联协议）。
- 预算：单次候选/状态 ≤64 文件、合计 512 MiB；每轮候选累计 1 GiB、状态累计 512 MiB；
  默认分块 1 MiB。
- 新增 `_outcome_unknown`：RPC 中断后本轮会话不可再复用（`RPC_OUTCOME_UNKNOWN`），不自动重试。
- 沙箱说明：**2 核 / 4 GiB 内存**（work 4 GiB、output/temp 各 1 GiB；OOM 可能被 kill）。
- **公开验证集只能用于测量/选择，不得并入梯度训练**（与隐藏标签是两条独立边界）；
  `scripts/sim/full_env.py::stage_data` 的 official-seq 分支已按此修正，并加了回归测试
  `tests/test_sim_protocol.py::test_official_seq_stage_data_keeps_validation_out_of_training`。

本地模拟宿主同步升级：`scripts/sim/upload_host.py`（UploadStore）处理 begin/chunk/finish；
`scripts/emulate_workspace_v2.py` 与 `scripts/sim/full_env.py` 的 `resource_limits` 改为 chunked-v1；
SDK 自带测试 `Ran 18 tests OK`，emulator A/B/C 各 2 轮 PASS（含分块上传）。
### 7.8 0.12.1 镜像与验证

- 测试：`pytest -q -m "not integration"` → **163 passed, 3 deselected**（71.97s）。
- 镜像：`zhixun-rsi-harness:0.12.1`，tar **380,334,080 B**，
  sha256 **`1e8697b2888f8ab5a4f0c226cf9eb93704156b7b2f45c3064648a96569bf793d`**，
  config `4ca4b333ec6024da760d3941bcb9f432272c67056999b4949ac139dc6ea2a3aa.json`。
- `scripts/verify_image.sh` **PASS**：describe 报 `sdk_version:"1.0.4"`；fake-host A 一轮 PASS；
  离线 A 两轮 workspace_report metric `0.7345546845546845`；selftest A 数字与 0.11.x 一致。
- **尚未上传平台**。

## 7.9 0.13.0 镜像（P2 结果落地）

- 版本 `0.13.0`：新增 `RSI_V2_CAPACITY_ARM=ls10` 默认（仅分类、官方 v1 checkpoint），
  `--thinking auto` 改为 off（E6 实测同分 9–10× 慢）。
- 镜像：`.scratch/image/harness.tar` **380,354,560 B**，sha256
  `8efa4c0c0a7e7746542774f0bd6d081de9e86fbde239a3dba171757778f14059`，
  config `c64043272e2497d6e4a1f7ecf7db134c123bd2fc8633c5d5a07edc6a1c50d745.json`。
- `scripts/verify_image.sh` PASS：describe 输出 `sdk_version:"1.0.4"`；fake-host A 一轮 PASS；
  离线 A 两轮 workspace_report metric `0.7345546845546845`；selftest 数字与 0.10.2/0.11.x 一致。
- 全量测试 `164 passed, 3 deselected`。

## 7.10 0.14.0 镜像（P3 冲刺上线版）

- 版本 `0.14.0`：可信选择 gate（分类走配对 block bootstrap，`RSI_V2_GATE=on`）、
  累积标签续训（`RSI_V2_ADAPT_ARM=auto`）、官方 v1 容量臂 `ls10`、schema 位置回退、
  行去重 + 数据探针；默认 ENV 见 `docs/39_SPRINT_0.14.0.md` §6。
- 镜像：`.scratch/image/harness.tar` **380,456,960 B**，sha256
  **`3f28d64a99e90608b6a5060579797c48600310d452cba2d39a414f2d0e391eae`**，
  config `b48ee4f1a43cf92d4f3a4bc86c282381176b147053ff82a71f4d8d2d26056b51.json`，
  tags `0.14.0/latest`。
- `scripts/verify_image.sh` **PASS**：describe `sdk_version:"1.0.4"`、fake-host A 一轮 PASS、
  离线 A 两轮 workspace_report metric `0.7345546845546845`。
- 全量测试 **200 passed, 3 deselected**；P2+P3 累计 96 个已完成 campaign run，
  E7–E12 矩阵 22 run / 152 次真实网关调用 / 0 failure（`docs/39_SPRINT_0.14.0.md`）。
- **尚未上传平台、未创建新 submission、未跑 trial**（0.13.0 的新的 trial 仍 queued）。

## 7.11 0.14.1 镜像（平台 trial 0.13.0 诊断修复）

- 背景：0.13.0 trial（run `a7bdfc755dadff8eea4e7fd8cc516268`）跑完但 A kernel
  每轮只执行 `get_run_state` 后无 tool action（calls 3–4、experiments=0、decision=null），
  候选全部落到 pair_history；A macro-F1 0.4635、B RMSE 735.07、C RMSE 56.45。
- 修复：kernel 无决策时降级 pipeline（保留真实 kernel call_ids，controller
  `kernel->pipeline`，configuration.kernel 增加 effective/experiments）；新增回归测试。
- 镜像：`.scratch/image/harness.tar` **380,631,040 B**，sha256
  **`05fdfd0b1e72ad309cfd9ae488ef5131d80bc486f640f22cee2ef0918816e7a3`**，
  config `97c8cc33ffe438be5b079c2854135157d88a9986d9cf8514427bdbc1c7eb27f6.json`，
  tags `0.14.1/latest`。
- `scripts/verify_image.sh` **PASS**：describe sdk 1.0.4、fake-host A 一轮、
  离线 A 两轮 workspace_report metric 0.7345546845546845；全量测试
  **201 passed, 3 deselected**（72.62s）。

## 7.12 0.14.2 镜像（native tool calling + thinking 协商）

- 根因：平台 `max_output_tokens=256` + PyTRiO `Qwen/Qwen3.8-27B` 默认 thinking on，
  reasoning 吃满输出预算导致 content 为空，文本 JSON 动作协议解析失败（详见
  `docs/39_SPRINT_0.14.0.md` §12）。
- 修复：`GatewayRuntime` 默认启用 native tool calling（`tools` + `tool_choice="auto"`），
  优先解析 `tool_calls`；默认 `enable_thinking=False`，宿主拒绝时逐级降级；
  0.14.1 的 `kernel->pipeline` 降级保留为兜底。
- 回归：`tests/test_agent_host.py` 12 passed；全量测试 **203 passed, 3 deselected**（72.61s）。
- 镜像：`.scratch/image/harness.tar` **380,641,280 B**，sha256
  **`1b85e0692f8ebd55fc1aa686e0f29976016fcf1fd903d1a9b595039bf78be8bb`**，
  config `1b8224512c40dba64d1fea147e18c06faa64ca0f0026c806aa02b0f9171372fa.json`，
  tags `0.14.2/latest`；`scripts/verify_image.sh` **PASS**（describe sdk 1.0.4、
  fake-host A 一轮、离线 A 两轮 workspace_report metric 0.7345546845546845）。
- 镜像 ENV 增加 `RSI_V2_NATIVE_TOOLS=1`、`RSI_V2_KERNEL_THINKING=off`、
  `RSI_V2_MAX_OUTPUT=256`；`RSI_V2_CAPACITY_ARM=ls10`、`RSI_V2_CONTROLLER=auto` 不变。
