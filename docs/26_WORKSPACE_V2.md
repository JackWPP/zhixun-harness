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
