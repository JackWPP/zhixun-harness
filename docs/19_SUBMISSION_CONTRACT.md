# 19 提交契约（Submission Contract）

> 状态：**草稿，依据 2026-11-15 23:18 (CST) 平台快照与官方 baseline 包实测**。
> 作者：teammate `contractlab`（task-4 交付物）。
> 一句话结论：**平台只公开到 submission / image-archive / run 这一层；镜像内部的输入目录、入口命令、输出文件路径没有任何官方文档或 SDK 定义**。本文把契约分成「已证实 / 推断 / 未知」三级，任何人写镜像内代码前都应先读 19.3 的多假设与 19.7 的 NO-GO 条件。

**证据目录（本地、只读、无凭据）**

| 位置 | 内容 |
| --- | --- |
| `.scratch/contract/api/body/*.json` | 2026-11-15T23:18 的 11 个只读端点原始响应 |
| `.scratch/contract/api/fetch_meta.json` | 抓取时间、HTTP 状态、字节数、sha256、顶层字段 |
| `.scratch/contract/api/changes_vs_20261115T1416.json` | 与 2026-11-15T14:16 旧快照的字段差分 |
| `.scratch/contract/baselines/verification.json` | 三个 baseline 包的 zip/checkpoint/code 三方校验（机器可读） |
| `.scratch/contract/baselines/SHA256_COMPARISON.md` | **三方 sha256 对比表**（仓库 zip / 平台下载包 / 包内声明值 + label_status 对照） |
| `.scratch/contract/baselines/unpacked/<id>/` | 三个包解压现场（含 TASK.md / TRAINING_GUIDE.md / predict.py） |
| `.scratch/contract/api/CHANGES.md` | API 快照说明与 14:16 差分（含新字段 execution_protocol） |
| `.scratch/contract/emulator/` | A/B 各 10 批批次流 fixture + demo 产物（批次流契约的本地证据） |
| `.scratch/platform/index.pretty.js` | 前端 bundle 反混淆（submission、run、workspace_report 的渲染字段来源） |
| `.scratch/rules.txt` | 组委会《智训引擎》规则 PDF 的文本提取 |

---

## 19.0 TL;DR

1. **已可对接平台 API 的完整提交链路**：`POST /image-archives`（`docker save` 未压缩 tar）→ `POST /documents` → `POST /submissions` → `POST /submissions/{id}/validate` → `POST /submissions/{id}/trial` →（正式开放后）`POST /submissions/{id}/select`。字段与状态机见 19.2。
2. **官方 baseline 包与仓库 `archives/` 完全一致**：三个 zip 与 `archives/SHA256SUMS` 逐字节相同；`checkpoint.npz` 的 sha256 与包内 `baseline-info.json` 声明值全部吻合；但 `code_sha256` **无法从包内容复现**（算法未公开，见 19.2.6）。
3. **预测协议是唯一被官方明示的镜像内契约**：`baselines/predict.py` 从 stdin 逐行读 `{"task":..., "inputs":...}`，向 stdout 逐行写 `{"outputs":...}`；分类 `outputs:[{label,score}]`，回归 `outputs:{value}`。公告要求镜像提供「独立预测入口」，这是目前唯一可验证的 IO 契约。
4. **未知项仍是硬缺口**（19.4）：入口命令、数据挂载路径、`workspace_report.json` 的落盘路径、decision 文件的强制格式、隐藏测试流的注入方式、LLM 端点如何注入、退出码语义，平台侧均无公开定义。
5. **上线前必须先跑**：`./.venv/bin/python scripts/contract_check.py`（退出码 0/1/2/3）与 `./.venv/bin/python scripts/emulate_stream_platform.py demo`（10 批 emulator 端到端）。任一非 0 都视为 NO-GO。

---

## 19.1 平台现状快照（2026-11-15T23:18 CST）

`GET /api/v1/rules` 当前值（`Authorization: Bearer <token>`，token 只在 `.scratch/platform/token`，0600，**不入库**）：

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `formal_ready` | `false` | 正式评测未开放 |
| `blockers` | `["正式评测暂未开放，请等待赛事通知。"]` | 正式提交被阻断 |
| `preparation_pending` | 同上 | 准备项未完成 |
| `trial_enabled` | `true` | 试运行已开放 |
| `trial_blockers` | `[]` | 试运行无平台侧阻断 |
| `trial_limit` / `trial_unlimited` | `10` / `true` | 试运行次数上限 10（当前 usage 0/10） |
| `trial_usage` | `{limit:10, remaining:10, used:0}` | 试运行用量 |
| `formal_usage` | `{limit:2, remaining:2, used:0, best_score:null}` | 正式提交 2 次机会 |
| `rules.wall_seconds` | `21600` | 单次运行墙钟上限 6 小时 |
| `rules.token_budget` | `8000000` | 大模型 token 预算 8M |
| `rules.training_budget` | `0` | 训练额度 0（纯 CPU/本地训练） |
| `rules.label_budget` | `9628` | 标注预算 |
| `rules.metric` / `metric_direction` | `macro_f1` / `higher` | 综合指标 |
| `rules.task` / `target_unit` / `dataset_family` | `classification` / `class` / `uci-gas-drift` | 任务属性 |
| `rules.evaluation_mode` | `batch` | 批次流评测 |
| `rules.evaluation_config_id` | `76e44dafec4225a250977eed7a48b811` | 绑定的评测配置 |
| `rules.formula` | `contest-objective-20261110-rul-v3` | 计分公式（见 `/scoring-rulesets`） |
| `rules.allowed_models` | `["Qwen/Qwen3.5-4B","Qwen/Qwen3.8-27B"]` | 允许驱动 Harness 的模型 |
| `rules.tie_policy` | `block` | 平局阻断 |
| **`rules.execution_protocol`** | **`contest.harness.workspace.v1`** | **本次新发现字段（旧快照没有）；平台侧执行协议版本号，具体语义未公开** |
| `rules.workspace_suite` | `{rounds:10, datasets:[A:draft,B:approved,C:draft]}` | 工作区十轮、三数据集 |

`/scoring-rulesets`（`contest-objective-20261110-rul-v3`，provisional）：objective 35（mean_gain 15 / worst_stage 5 / recovery 5 / retention 5 / resource_efficiency 5）+ subjective 65 + bonus 5+5；A 用 macro_f1/accuracy，B 用 rmse(minutes)/mae，C 用 rmse(cycles)/nasa_score，三数据集各 1/3。

`/announcements` 两条（自 2026-11-15 14:16 快照以来**内容未变**）：
- 《镜像上传与失败排查》：必须 `docker save` 单个 Linux/amd64 镜像为**未压缩 tar** 上传；不要源码 ZIP / Dockerfile / `docker export` / gzip；状态机 `uploading → uploaded → importing → ready|failed|expired`。
- 《续训评测提醒》：必须基于官方 baseline 与平台父检查点**真实续训**（retain 即不训练保留父模型是唯一例外）；镜像需含实际运行代码与配置，并提供**独立预测入口**；模型产物须符合 `checkpoint.npz` 交付约定，**产物总量 ≤64 MiB**；不得改后缀、附带未用父检查点、补造元数据。

`/image-archives` 当前一条：`id=5b80ee1c89b6cb2a1e149fef93456047, status=ready, size_bytes=374968320, cleanup_status=cleaned`（≈357.6 MiB，满足 ≤2GB 公告约束）。

`/dashboard`：`ranked_teams=0, runs=0, submissions=0, teams=1`；`/leaderboard` 空；`/runs`、`/submissions` 空 —— 平台尚无任何正式评测数据，**所有 run 侧契约只能靠前端字段与推理**。

### 19.1.1 与旧快照的差分（`.scratch/contract/api/changes_vs_20261115T1416.json`）

| 端点 | 变化 |
| --- | --- |
| `/rules` | 新增 `rules.execution_protocol="contest.harness.workspace.v1"`；其余字段值不变 |
| `/image-archives` | `items` 从空变为 1 条 `ready` 归档（即上面的 357.6 MiB 镜像） |
| `/announcements` `/baselines` `/dashboard` `/scoring-rulesets` `/module-visibility` | 值不变 |
| `/submissions` `/runs` `/leaderboard` | 仍为空 |

**风险提示**：`execution_protocol` 出现说明平台执行协议仍在演化；正式开放前必须再抓一次快照比对。

---

## 19.2 已证实（有直接证据）

### 19.2.1 镜像上传

- 端点：`POST /image-archives`；`Content-Type: application/x-tar`；**raw body = `docker save` 产出的未压缩 tar**（不是 gzip、不是源码 tar、不是 Dockerfile）。
- 约束：单个 Linux/amd64 镜像；公告未写硬上限，会议纪要为 ≤2GB；我们的归档 357.6 MiB。
- 状态机：`uploading`(上传中) → `uploaded`(等待导入) → `importing`(发布与核验中) → `ready`(镜像可用) / `failed`(发布未完成) / `expired`(已过保留期)。
- `cleanup_status`: `pending` / `failed` / `cleaned`（归档清理；`cleaned` 表示临时文件已清）。
- 证据：`.scratch/platform/upload.log` 实测返回 `HTTP 201 {"id":"5b80ee...","size_bytes":374968320,"status":"uploaded"}`；前端 bundle 镜像状态映射字符串原文。

### 19.2.2 文档上传

`POST /documents`，body `{name, content_base64}`，≤2 MiB，accept `.pdf`/`.md`/`.txt`；返回 `document_id` 供 submission 引用。

### 19.2.3 提交创建字段（前端表单反编译，字段名精确）

```json
POST /submissions
{
  "model": "trio://<hostname>/<path> 或 白名单基础模型名",
  "model_source": "sampler" | "base",
  "base_model_confirmed": true,          // model_source=="base" 时必须 true
  "api_key": "PyTRiO API Key",
  "training_url": "https://...",         // HTTPS 必填
  "source_url": "https://...",           // HTTPS 必填
  "image_archive_id": "5b80ee...",
  "document_id": "..."
}
```

- `model_source="sampler"` 时 `model` 必须是 `trio:` 协议路径：有 hostname、无 username/password/query/fragment/空白。
- `model_source="base"` 时必须勾选「二次确认使用白名单基础模型」→ `base_model_confirmed=true`。
- 旧提交可能有 `model_source="legacy"`（保留成绩但未按新模型类型核验）。

### 19.2.4 提交状态与操作

| 操作 | 方法/路径 | 说明 |
| --- | --- | --- |
| 校验作品 | `POST /submissions/{id}/validate` | 返回 `validation_error[]`；通过后 `status` 可变为 `valid` |
| 试运行 | `POST /submissions/{id}/trial` | body `{evaluation_config_id}`（取 `/rules` 的 `rules.evaluation_config_id`） |
| 选正式版本 | `POST /submissions/{id}/select` | 要求 `status=="valid"` |
| 材料/记录审核 | `POST /submissions/{id}/review`（admin） | body `{records_complete, materials_complete?, notes}` |
| 合规判定 | `POST /submissions/{id}/compliance`（admin） | body `{decision:"noncompliant"|"clear", evidence}` |

试运行按钮可用条件：`image_archive_id` 存在、`status ∈ {valid, frozen}`、`compliance_status != noncompliant`、`policy.trial_enabled && trial_blockers 为空 && trial_usage.remaining > 0`。
**试运行需要 PyTRiO API Key；当前团队没有，因此 trial 无法本机演练**（这是已知缺口，不是平台阻断）。

### 19.2.5 run / workspace_report 结构（平台前端读取的字段）

`GET /runs/{id}/report` 返回：

```
{
  run, workspace_report, rsi{prompt_version, candidates[{artifact_id,parameters,validation{metrics,metric,metric_direction,total}}], selected_candidate},
  submission_id, image_digest, model_identity,
  artifact_sha256, evaluation_config_id, evaluation_config_digest, dataset_digest, evidence_digest,
  workspace_suite{rounds, datasets[{key,baseline_id}], suite_digest},
  objective_result{preview, objective_score, ...}
}
```

`workspace_report`（由 harness 侧产出、平台侧展示）结构：

```
{ datasets: [{
    key: "A"|"B"|"C",
    rounds: [{
      id, round_index,
      decision: "promote"|"retain"|"rollback",
      validation_metrics: {...},
      history_ids: [...],
      hypothesis, change_summary,
      parent_checkpoint_sha256, candidate_checkpoint_sha256, candidate_code_sha256,
      checkpoint_sha256, code_sha256
    }],
    record_complete: bool,
    prediction_count: int,
    cpu_accounting_status: "measured"|"unavailable",
    cpu_seconds: number,
    wall_seconds: number,
    metrics: {macro_f1, accuracy, rmse, mae, nasa_score, severe_miss_rate},
    baseline_checkpoint_sha256, baseline_code_sha256,
    checkpoint_sha256, code_sha256, artifact_sha256,
    evaluation_config_id, evaluation_config_digest
}]}
```

关键展示逻辑（前端代码原文）：`record_complete=true` 显示「完整」；`cpu_accounting_status!=="unavailable"` 且 `cpu_seconds` 为有限非负数才显示 CPU 时长；轮次取最新 `decision==="promote"` 作为当前模型；B 数据集若 `metrics` 无 `rmse` 字段则展示为「轴承健康识别」而非 RUL 回归。
`rsi.candidates[]` 记录候选 artifact、参数与验证指标；`selected_candidate` 为最终选择；trial 报告若 `workspace_suite.rounds<10` 会被标注为「历史小样本试运行」。

### 19.2.6 官方 baseline 包（实测校验，三方对比）

| id | zip sha256（仓库=下载=SHA256SUMS） | 包大小 | checkpoint sha256 声明=实测 | `code_sha256` 声明 | 包内 `label_status` | API `label_status` |
| --- | --- | --- | --- | --- | --- | --- |
| `cpu-a-guide-20261110-v1` | `7a992b92…ec8851` | 1,133,600 | `fd1e08b2…818c757` ✅ | `3b4d2ff7…2980313` | **draft** | **approved** ⚠️ |
| `cpu-b-rul-20261110-v1` | `68db6a72…c69d68` | 3,164,954 | `c7d819fc…314b4` ✅ | `25f3fcef…b5e83` | approved | approved ✅ |
| `cpu-c-guide-20261110-v1` | `6cb8784e…211f` | 7,714,436 | `98ec6e30…00bc4e` ✅ | `3b4d2ff7…2980313` | **draft** | **approved** ⚠️ |

- 三个 zip 与 `archives/SHA256SUMS` **逐字节一致**（`verification.json` 中 `repo_equals_download=true`、`matches_sha256sums=true`；完整表格见 `.scratch/contract/baselines/SHA256_COMPARISON.md`）。
- `checkpoint_sha256` 复现算法 = `"sha256:" + sha256(checkpoint.npz 原始字节)`，三个包全部吻合。
- 2026-11-15 23:33 复核（`contract_check.py --online`）：三方 zip/checkpoint 仍全部一致，**无新差异**；仅剩两类已知差异 —— `code_sha256` 不可复现、A/C 的 `label_status`（API `approved` vs 包内 `draft`），二者均已在检查器中登记为 INFO，不会误报硬失败。
- **`code_sha256` 不可复现**：试过「包根/包内 baselines 目录」「仅 .py / .py+.json+.txt / 全部文件」「path\0bytes\0、path:sha、sha  path、JSON manifest」等数十种组合，均不匹配；A 与 C 声明值相同（两者 `baselines/` 代码逐字节相同），B 不同。结论：该字段是构建方对其源码树的声明式摘要，**算法未公开**，只能当作不透明声明记录，不能作为本地可验证项。
- **label_status 不一致**：`/baselines` 与 `rules.workspace_suite` 把 A/C 标为 `approved`，包内 `baseline-info.json` 写 `draft`。B 一致为 `approved`。这直接影响「哪些数据可作正式训练标签」的判断，需在正式开放前向组委会确认（见 19.4 U6）。
- 包内布局（三包同构）：`START.md` `TASK.md` `TRAINING_GUIDE.md` `baseline-info.json` `baselines/{configs,data.py,model.py,predict.py,train.py,tests,...}` `checkpoint.npz` `history.jsonl` `public/{train,validation}.jsonl` `training.json`（B 额外含 `baselines/docs/`）。
- A 数据集事实：`inputs.features` 长度 128；`answer` ∈ {Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene}；train 445 行 / validation 1244 行；基线 macro-F1=0.714030；validation 全在 `meta.source_batch=2`（隐藏测试对应批次 3+，不外发）。

### 19.2.7 预测协议（唯一明示的镜像内 IO 契约）

官方 `baselines/predict.py` 行为（原文注释：「JSONL prediction adapter. Reads inputs, never labels; no model execution hooks.」）：

```
stdin  (每行一个请求):   {"task": "classification"|"regression", "inputs": {...}}
stdout (每行一个响应):   classification → {"outputs":[{"label":"Ethanol","score":0.83}, ...]}
                        regression     → {"outputs":{"value":123.4}}
```

- 请求对象只允许 `{task, inputs}`，出现其他键即报错；`task` 必须与 checkpoint 的 `meta.task` 一致。
- 分类标签串必须与 `TASK.md` 的英文类名一致；回归输出须还原为任务原单位、截断为非负。
- 公告要求镜像提供「独立预测入口」——即镜像里必须有等价于 `python -m baselines.predict --checkpoint ...` 的 stdin/stdout 适配器（我们的 `docker/predict.sh` → `python -m rsi.predict` 已实现该协议）。
- 平台「独立运行预测」意味着平台会用**自己的**输入调用该入口并评分；自报指标不算。

### 19.2.8 checkpoint.npz 交付约定

- 每个新输出目录含 `checkpoint.npz`（权重 + Adam 一/二阶矩 + 累计步数 + epoch + 种子 + 类别 + 特征 schema + 标准化/目标缩放参数）、`training.json`、`history.jsonl`。
- `--from-baseline`（首轮）或 `--resume`（后续轮）必须恢复全部优化器/标准化状态；父检查点摘要要对得上，累计步数应增加。
- 产物总量 ≤64 MiB；不得改后缀、不得附带未用父检查点、不得补造元数据。

### 19.2.9 资源与合规

- 运行环境：统一、离线、资源受限、无人工干预；开发机 Python 3.10+；基础任务普通 CPU 即可。
- 会议纪要（**非平台 API**，用于推断）：4 核 8G、无 GPU、镜像 ≤2GB、禁外网、禁大模型权重；批次流每批带标签、累积重发、先决策后下一批；只允许 Qwen3.8-27B / 3.5-4B。
- 计分：`/scoring-rulesets` 见 19.1；objective 单位增益消耗算力/标注配额/大模型 Token。

### 19.2.10 审计与运行阶段（平台后台动作名）

`run.queue`（安排正式评测）→ `run.claim`（开始准备评测）→ `run.materialize`（准备运行文件）→ `run.dataset_materialize`（准备评测数据）→ `run.running`（开始运行）→ `run.artifact_check`（检查运行产物）→ `run.reviewing`（开始评测）→ `run.complete` / `run.failed`；另有 `run.platform_fault_confirmed`、`run.platform_replacement`（确认平台故障、安排补跑）、`retry.request` / `retry.decide`（申请/审批重跑）、`trial.queue` / `trial.complete`。
规则 blocker 代码映射（原文）：「尚未绑定真实数据集评测配置」「工业任务未定义」「性能指标和静态基线未确定」「任务、性能指标和优化方向不一致」「静态基线超出指标范围」「数据版本摘要未冻结」「可信计量尚未验收」「正式执行环境尚未验收启用」「可信提交校验服务未配置」「正式确定性评分服务未配置」「执行器认证未配置」。
**含义**：平台执行链是 materialize（工作区文件 + 评测数据）→ run（容器执行）→ artifact_check（检查产物）→ reviewing（服务端评分）。`run.materialize` / `run.dataset_materialize` 的具体目录约定未公开。

---

## 19.3 推断（多假设 + 置信度）

> 置信度：高 = 有强间接证据；中 = 合理但无直接证据；低 = 纯猜测。写镜像代码时必须对同一契约做**多路径兼容**（例如同时写 `/work` 与 `$RSI_STATE_DIR`，两者默认相同）。

### H1 镜像入口（高）
平台对镜像的调用等价于 `docker run <image>`（无参数）或 `docker run <image> run`，且在 `wall_seconds=21600` 内必须无人值守完成全部 10 轮。我们的 `ENTRYPOINT /opt/zhixun/bin/entrypoint.sh`、默认 `COMMAND=run` 已覆盖两种。
**反证/风险**：平台可能传 `--round N`、`--batch-file` 等参数；`entrypoint.sh` 对未知子命令会 `exec`，参数会透传给 `rsi.cli`，可能报错。缓解：入口对常见参数做兼容解析（本轮不实现）。

### H2 数据注入路径（中）
候选：
- (a) 平台把隐藏测试流/批次文件挂载到 `/data` 或 `/data/batches`，镜像只读；
- (b) 平台只改写 `/work`（可写状态目录），数据通过环境变量或 stdin 逐批喂入；
- (c) 平台把三数据集直接放到镜像内既有路径 `/opt/zhixun/app/datasets`（与 baseline 包 `public/` 同构），评测配置定义 train/val/test 版本；
- (d) 平台以 `task` 级 JSONL 通过预测入口（stdin）喂隐藏批次，不落盘。
**当前实现**：只读 `/opt/zhixun/app/datasets`（镜像自带），写 `/work/rsi-runs`；**不扫描 `/data`/`/input`**。若平台走 (a)/(b)，需要新增挂载点探测与批次目录读取。
**置信度**：路径本身低；「平台会 materialize 数据到镜像可见位置」高（审计动作 `run.dataset_materialize` 证实）。

### H3 输出落盘（中）
- 我们的实现：`$RSI_STATE_DIR`（默认 `/work/rsi-runs`）下每次 run 目录含 `workspace_report.json` / `summary.json`，entrypoint 另把二者复制到 `/work/workspace_report.json`、`/work/summary.json`。
- 平台取回方式很可能按约定路径读取（`run.artifact_check` 检查运行产物），但**路径/文件名未知**。
- 兼容建议：同时写 `/work/workspace_report.json`、`/work/rsi-runs/latest/workspace_report.json`、`$RSI_STATE_DIR/workspace_report.json`，并保留 run 目录内原始文件。
**置信度**：字段结构高（前端渲染证实）；路径低。

### H4 decision 文件（中）
- 会议纪要「先决策后下一批」；前端 workspace_report.rounds[].decision ∈ {promote, retain, rollback}；我们的 `rsi/stream` 每批写 `decision.json`（当前只在 stream 子命令，主 `run` 路径没有）。
- **平台是否强制要求独立 `decision.json` 文件、文件名/位置/schema，完全未知**。
- 兼容建议：主流程也每批产出 `decision.json`，字段包含 `decision`、`round_index`、`parent_checkpoint_sha256`、`candidate_checkpoint_sha256`、`reason`、`metrics`。

### H5 预测文件（低-中）
- 会议纪要「批次流每批带标签、累积重发」；官方 predict 协议是 stdin/stdout。
- 平台可能：(a) 每批用 stdin 调独立预测入口并收集 stdout；(b) 要求镜像写 `predictions.jsonl`（每行 `{"id","outputs"}`）。
- 我们的 `rsi/stream` 写 `predictions.jsonl`，`rsi.predict` 支持 stdin；两条都覆盖最好。
**置信度**：schema（`id`+`outputs`）中（与官方 predict 输出同源）；落盘路径低。

### H6 批次=累积重发（中）
会议纪要明确「累积重发」，即第 i 轮拿到的是「历史全部 + 当前批」。我们的 `rsi/stream` 按此建模（`materialise_package` 写入累计 train + 当前 validation），默认 10 轮。
**风险**：若平台只发增量，训练集将持续增长策略需要从累计文件恢复，否则语义不同。缓解：读入时以「文件内全部行」为准，不假设增量。

### H7 退出码（中）
最可能：容器退出码 0 = 成功，非 0 = 运行失败（`run.failed`，可申请重跑一次取较高成绩）；部分失败可能由平台 `artifact_check` 判定。我们的 entrypoint 用 `set -eu`，任一任务异常会非 0 退出；**生产上需要考虑：是否应容忍单任务失败而整体仍返回 0**——未知，保守做法是失败即非 0（宁可申请重跑）。

### H8 LLM 端点注入（低）
允许模型只有 Qwen3.5-4B / Qwen3.8-27B，但运行环境禁外网；平台必然通过某种内网端点注入（环境变量/挂载配置文件）。我们的 entrypoint 支持 `RSI_LLM_ENV` 指向 KEY=VALUE 文件，`RSI_LLM=auto` 无配置时退化为确定性控制器。
**风险**：平台环境变量名未知 → 需要兼容探测常见命名（OPENAI_BASE_URL / OPENAI_API_KEY / RSI_LLM_ENV 等）；未注入时不能硬失败，必须降级。

### H9 多数据集 A/B/C（中）
`workspace_suite.datasets=[A,B,C]`、`rounds=10`；最可能是一次 run 内三数据集并行/串行各 10 轮，或按 `evaluation_config` 的 train/val/test 组合评测。我们的 `rsi.cli run --tasks A,B,C --rounds 10` 已覆盖串行；若平台按单数据集调用（`--task A`），入口也支持。

### H10 产物上限与 checkpoint 交付（高）
公告明示 ≤64 MiB；平台可能扫描镜像/工作区内的 `checkpoint.npz` 并按约定结构校验。我们应确保：工作区最终只保留被选中的检查点链 + 证据文件，总量受控。

---

## 19.4 未知项（Top，按风险排序）

| # | 未知 | 为什么重要 | 当前应对 |
| --- | --- | --- | --- |
| U1 | **镜像入口命令与参数** | 入口不匹配 = 直接 run.failed | 默认无参 `run`；兼容 `run`/`predict`/`selftest` 子命令 |
| U2 | **批次/隐藏数据注入路径** | 读不到数据 = 零预测 | 只读自带 datasets；不依赖 `/data` 存在 |
| U3 | **workspace_report / decision / predictions 的落盘路径与文件名** | 平台取不到产物 = artifact_check 失败 | 多路径写 + 保留 run 目录 |
| U4 | **平台对预测入口的调用方式**（stdin 批 vs 文件 vs 目录） | 不会调用 = 没分数 | `rsi.predict` 支持 stdin 行协议；stream 另外落盘 predictions.jsonl |
| U5 | **退出码语义与部分失败容忍度** | 误判失败/误报成功 | 保守：失败非 0 |
| U6 | **A/C 的 `label_status`：API=approved vs 包=draft** | 用未批准标签训练可能被判不合规 | 训练只用官方 train split；文档声明；正式前向组委会确认 |
| U7 | **试运行需要 PyTRiO API Key，我们暂无** | 无法本机验证 trial 链路 | 用本地 emulator 替代；API Key 到位后再 trial |
| U8 | **LLM 端点注入方式与环境变量名** | 无法调用大模型 = Harness 降级 | 支持 `RSI_LLM_ENV`；无配置自动降级；待平台文档 |
| U9 | **`execution_protocol=contest.harness.workspace.v1` 的具体语义** | 可能是必须遵守的目录/命名协议 | 记录并持续观察；正式开放前重抓 |
| U10 | **平台资源计量口径（CPU 秒/token 如何采集）** | 影响 objective 分与合规 | 自报 `cpu_seconds` + `cpu_accounting_status`；如实记录，不伪报 |
| U11 | **镜像保留期/归档过期时间** | 过期需重新上传 | 保持归档 `ready`；上传后立即核验 |
| U12 | **code_sha256 算法** | 平台若用它校验代码 lineage，我们无法自证 | 按 checkpoint/sha 与文件清单自证；平台侧算法未知 |

---

## 19.5 本仓库实现现状与平台证据对照

| 契约点 | 平台证据 | 仓库实现 | 一致性 |
| --- | --- | --- | --- |
| 镜像 tar 交付 | 公告 | `scripts/build_image_tar.py`（无 daemon 构建）+ `docker/Dockerfile` | ✅ |
| 独立预测入口 stdin/stdout | 官方 `baselines/predict.py` | `docker/predict.sh` → `rsi/predict.py`（同协议） | ✅ |
| 分类/回归输出 schema | 官方 predict.py | `rsi/inference.format_outputs` | ✅ |
| workspace_report 字段 | 前端渲染 | `rsi/runner.py::_workspace_report` + `write_summary` | 部分（字段名对齐，`suite_digest`/`evaluation_config_*` 等平台注入字段未自行产出） |
| 十轮工作区 | `rules.workspace_suite.rounds=10` | `--rounds 10` 默认 | ✅ |
| 三数据集 | `workspace_suite.datasets` | `--tasks A,B,C` | ✅ |
| decision.json | 会议纪要 + 前端 rounds.decision | 仅 `rsi/stream` 每批写；主流程无 | ⚠️ 需补；真实 stream 落盘为**嵌套记录**（`decision` 是对象、无顶层 `round_index`/`parent_checkpoint_sha256`），与本节扁平假设不一致 → 建议双写（见 `.scratch/contract/emulator-run/README.md §3`） |
| 产物 ≤64MiB | 公告 | 检查点小（A 132KB），但工作区可能累积多轮/多数据集 | ⚠️ 需清理策略 |
| 状态目录可写 | 推断 | `/work` VOLUME + `RSI_STATE_DIR` | 推断 |
| 无外网 | 规则 | 默认不联网；`swanlab=False` | ✅ |
| LLM 注入 | 推断 | `RSI_LLM_ENV` + `RSI_LLM=auto` 降级 | 推断 |

---

## 19.6 上线前验证步骤（按序执行，全部留痕）

```sh
# S0 契约快照：刷新 API、校验 baseline、检查漂移（只读）
./.venv/bin/python scripts/contract_check.py --online
#   exit 0=全部通过；1=硬失败；2=仅告警（漂移/已知不一致）；3=网络/鉴权失败

# S1 baseline 三方校验（离线，可单独跑）
./.venv/bin/python scripts/contract_check.py --offline --strict-baselines

# S2 批次流 emulator：生成 10 批 + 自检（2026-11-15 实测 A/B 各 64 项 PASS、退出码 0）
./.venv/bin/python scripts/emulate_stream_platform.py make --dataset A --batches 10
./.venv/bin/python scripts/emulate_stream_platform.py demo --dataset A,B --force
./.venv/bin/python scripts/emulate_stream_platform.py validate --dataset A,B --require-outputs
#   demo 不训练，decision 全为 retain（「retain = 不训练保留父模型」是公告允许的例外）；
#   --dataset C --batches 10 约需 0.4 GB 磁盘（fixture 硬链接去重），默认不建。

# S3 镜像冒烟（有 docker 时）
docker load -i .scratch/image/harness.tar
docker run --rm zhixun-rsi-harness:0.3.0 selftest A
printf '%s\n' '{"task":"classification","inputs":{"features":[...]}}' \
  | docker run --rm -i zhixun-rsi-harness:0.3.0 predict --checkpoint /opt/zhixun/app/datasets/cpu-a-guide-20261110-v1/checkpoint.npz

# S4 真实 harness 过 emulator（不进平台的一次彩排；2026-11-15 实测 A/B 各 2 批 PASS=18、退出码 0）
./.venv/bin/python scripts/emulate_stream_platform.py run --dataset A,B --batches 2 \
  --out .scratch/contract/emulator-run --force \
  --harness-cmd "$PWD/.venv/bin/python $PWD/.scratch/contract/real_harness_adapter.py"
#   真实 CLI 路径（完整 Runner，约 2 分钟）：
./.venv/bin/python -m rsi.cli stream --source official --tasks A --batches 2 \
  --state-dir .scratch/contract/emulator-run/cli-stream --llm off --no-memory
#   对照结论与 decision.json schema 差异见 .scratch/contract/emulator-run/README.md

# S5 试运行（需 PyTRiO API Key，API Key 不可写入仓库/日志）
#   validate → trial → 读 /runs/{id}/report 核对 workspace_report 与 suite rounds
```

每步都要记录：命令、退出码、产物路径、时间戳（写入 `.scratch/contract/logs/`，不得含 token/api_key）。

---

## 19.7 NO-GO 条件（任一命中即不提交正式评测）

1. `formal_ready != true` 或 `blockers` 非空。
2. `trial_blockers` 非空、`trial_usage.remaining == 0`、或无有效 `image_archive_id`（`status != ready`）。
3. `scripts/contract_check.py --online` 退出码 ≠ 0；或 baseline 三方校验失败。
4. 镜像内独立预测入口不符合官方 stdin/stdout 协议（用 emulator `validate` 验证失败）。
5. 工作区产物总量 >64 MiB，或缺少 `checkpoint.npz` 交付约定、父检查点摘要无法对上。
6. 镜像内存在外网访问、凭据/API Key、大模型权重，或训练从头另起一套模型（违反续训合规）。
7. 未跑通 emulator 的 10 批彩排（`run` 子命令非 0 或 `validate` 失败）。
8. 平台规则/公告在正式开放后发生未复核的变化（`execution_protocol`、`evaluation_config_id`、`trial_*`、`workspace_suite`）。
9. A/C `label_status` 冲突未澄清且训练使用了非官方 train split 的标签。
10. 任何 token/密钥进入仓库、镜像、日志或提交文档。

---

## 19.8 附录：抓取与复现

```sh
# 只读 API（token 从文件读取，绝不回显）
TOKEN=$(cat .scratch/platform/token)
curl -s -H "Authorization: Bearer $TOKEN" https://rsi-index.115.zone/api/v1/rules

# baseline 校验
./.venv/bin/python scripts/contract_check.py --offline --strict-baselines -v

# emulator
./.venv/bin/python scripts/emulate_stream_platform.py --help
```

`/rules` 的 `execution_protocol` 是 2026-11-15 新出现的字段，说明平台执行协议版本化；**正式开放后第一件事是重跑 `contract_check.py --online` 并比对 `.scratch/contract/api/changes_*.json`**。
