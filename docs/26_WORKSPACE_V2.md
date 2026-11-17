# 26 · Workspace v2 接入（contest.harness.workspace.v2）

> 适用版本：harness **0.9.0**。平台 `GET /api/v1/rules` 的 `execution_protocol` 已从
> `contest.harness.workspace.v1` 升级为 **`contest.harness.workspace.v2`**；镜像若仍只有
> 自研 CLI（`run/stream/predict`），试运行诊断会报
> “镜像还需要实现 start/candidate/save_state/complete”。本文记录 v2 契约、我们的接入实现与本地验证方法。

## 1. 规范来源

- 平台前端 bundle（`.scratch/platform/index-new.js`）内嵌《选手 Harness SDK 使用说明》，并给出下载地址
  `/downloads/contest-harness-sdk-1.0.2.tar.gz`。
- 已下载并校验：`sha256:c676c8beeef69cece9cbec5a633d05dbb9832eb22b6f761913f2315bd1f85133`
  （与平台元数据一致）。仓库 `vendor/contest_harness-1.0.2/` 保存该 SDK 源码副本用于离线构建，
  升级 SDK 时替换目录并同步本文件。
- 固定入口：镜像内 `python3 /app/harness.py describe|train|predict`，选手模块
  `/app/contest_submission.py`（可用 `CONTEST_SUBMISSION` 改名），导出 `train(rnd)`、`predict(sample, model_dir)`、`TASKS`。

## 2. v2 契约摘要

| 项 | 约定 |
|---|---|
| describe | 一行 JSON：`{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.2","tasks":[...]}` |
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
- 镜像构建：`docker/requirements.txt` 不含 SDK（纯 Python），`vendor/contest_harness-1.0.2/contest_harness` 被拷到
  `/opt/zhixun/app/contest_harness`；`docker/app/*` 被拷到镜像 `/app/`（`scripts/stage_image.sh` 同步支持无 daemon 构建）。

## 4. 本地验证

```bash
# 1) describe（等价于平台诊断调用的入口）
PYTHONPATH="$PWD:$PWD/vendor/contest_harness-1.0.2:$PWD/docker/app" \
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
> `vendor/contest_harness-1.0.2/HARNESS_SDK.md` 基本一致，新增了“提交里要有什么 / 目录权限 / 公开数据 /
> 父模型 / 一轮训练 / 预测 / 本地检查”等操作细节）。逐项对照结果如下（PASS=无需改动；
> 已修=本轮发现并修复；注意=边界说明）。

| # | 官方要求 | 我们的实现 | 结论 |
|---|---|---|---|
| 1 | 镜像内安装 SDK，`/app/harness.py` 固定入口，选手模块默认 `/app/contest_submission.py` | `docker/app/harness.py` bootstrap 插入 `/opt/zhixun/{site-packages,app}` 后调 `contest_harness.cli.main`；`docker/Dockerfile` 拷 vendor SDK + 两个 app 文件；`entrypoint.sh` 对 `describe/train/predict` 直接 `exec python3 /app/harness.py "$@"` | PASS |
| 2 | `describe` 向 stdout 打印一行 JSON，至少含 protocol/model_access/tasks；会 import 模块并检查 `train`/`predict`/任务声明 | 本地 `harness.py describe` 输出 `{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.2","tasks":["classification","regression"]}`；`TASKS` 恰好是两类 | PASS（`tests/test_platform_v2.py::test_describe_speaks_workspace_v2`） |
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
| 18 | 旧 `contest_sdk` HTTP 客户端（`/runtime/v1/runs/...`）与 v2 不同 | 我们完全走 v2 `contest_harness` 1.0.2，未用旧客户端 | PASS |

### 7.1 本轮实际执行的验证

```bash
# 官方 SDK 自带离线协议测试（15 项）
cd vendor/contest_harness-1.0.2 && ../../.venv/bin/python -m unittest \
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
