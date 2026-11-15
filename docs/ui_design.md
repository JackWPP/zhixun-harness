# 智训引擎 · RSI 看板 / Dashboard 设计说明

本目录记录本地实时看板 `rsi/dashboard` 的状态数据契约、页面结构、视觉规范、接入方式与演示流程。
看板服务于三类场景：**开发调试**（观察 harness 每轮决策）、**演示视频**（信息密度高、节奏感强）、**评审展示**（证据链可下钻）。

看板是只读的：不调用 harness、不修改 `state.json`，只按 2s 周期重新读取 harness 原子写入的文件与事件流。

---

## 1. 快速启动

```bash
# 1) 一条命令启动：未指定 --state 时自动生成 schema v3 演示数据包
#    （含软目标 / 批次时间线 / 漂移雷达 / 知识 / 提案来源）
./.venv/bin/python -m rsi.dashboard --port 7861
#    → .scratch/demo_state.json / records.jsonl / rounds.jsonl
#      + memory/*.md + rsi_state.db

# 2) 打开页面（任意浏览器，无需联网）
#    http://127.0.0.1:7861/
#    需要重建演示数据时加 --refresh-demo。
```

接入真实 harness 时，把 `--state` 指向其原子写出的 `state.json` 即可；同目录的 `records.jsonl`、`rounds.jsonl`、`memory/` 与 `rsi_state.db` 会被自动发现。`scripts/mock_state.py` 仍保留，用于生成向后兼容的 schema v1 样例：

```bash
./.venv/bin/python scripts/mock_state.py --out .scratch/demo_state.json \
  --records-out .scratch/demo_records.jsonl
./.venv/bin/python -m rsi.dashboard --state .scratch/demo_state.json --port 7861
```

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--state` | 自动生成 v3 demo 包 | harness 原子写出的 `state.json` 路径 |
| `--records` | 与 state 同目录的 `records.jsonl` | 事件/轮次追加日志路径；不存在时自动忽略 |
| `--host` | `127.0.0.1` | 监听地址；容器内演示可设 `0.0.0.0` |
| `--port` | `7861` | 监听端口；`0` 表示随机端口（测试用） |
| `--demo-out` | `.scratch/demo_state.json` | 未给 `--state` 时 v3 demo 包输出位置（sidecar 文件写到同目录） |
| `--refresh-demo` | 关闭 | 未给 `--state` 时强制重新生成 demo |
| `--quiet` | 关闭 | 关闭访问日志（pytest 中使用） |
| `--version` | — | 打印版本后退出 |

服务启动后 stdout 会打印数据源与监听地址，便于脚本/测试解析：

```text
[dashboard] 数据源: /path/to/state.json
[dashboard] 事件源: /path/to/state.json + /path/to/records.jsonl
[dashboard] listening on http://127.0.0.1:7861
```

停止：前台 `Ctrl-C`，或后台进程 `kill <pid>`。

---

## 2. 页面结构

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 顶栏：品牌 / 运行状态灯 / run_id / 已运行时间 / LLM 后端 / 调用量 / 实时指示 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 运行预算条：Token · CPU · 墙钟 · 实验次数（进度条，>=80% 琥珀 / >=95% 红） │
├─────────────────────────────────────────────────────────────────────────────┤
│ 任务矩阵（>=1440px 三列；901-1439px 两列；<=900px 单列）                    │
│ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                       │
│ │ Task A 卡片   │ │ Task B 卡片   │ │ Task C 卡片   │                       │
│ │ 指标大数字    │ │               │ │               │                       │
│ │ Δ 基线 / 最优 │ │               │ │               │                       │
│ │ 进度环        │ │               │ │               │                       │
│ │ SVG sparkline │ │               │ │               │                       │
│ │ 漂移四段仪表  │ │               │ │               │                       │
│ │ 当前轮次      │ │               │ │               │                       │
│ │ 最近 3 条事件 │ │               │ │               │                       │
│ │ [轮次时间线]  │ │               │ │               │                       │
│ │ [Panel 讨论]  │ │               │ │               │                       │
│ └───────────────┘ └───────────────┘ └───────────────┘                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 软目标与批次（>=1210px 三列软目标卡）：目标 / 当前 / 缺口 + 覆盖条 + 批次线 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 漂移雷达与知识（>=1210px 三列）：4 轴雷达 + 知识命中/技能 + 提案来源堆叠条  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 记忆与自主内核（>=1210px 三列）：MEMORY 用量 / lessons / 策略胜率 + 技能    │
│  + controller 徽章 / kernel 计数 + records·rounds 检索框与结果              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 下方面板（两列，<=1210px 单列）                                             │
│ ┌───────────────────────────────────┐ ┌───────────────────────────────────┐ │
│ │ 全局事件流（级别 / 任务过滤）     │ │ 决策分布 + 任务进度 + 数据链路    │ │
│ └───────────────────────────────────┘ └───────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│ 页脚：版本 / 数据源路径 / 生成时间 / schema 版本 / 数据延迟                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    右侧抽屉：单轮证据（假设/变更/指标/软目标/提案/panel/sha）
```

### 交互

| 操作 | 行为 |
| --- | --- |
| 点击任务卡「轮次时间线」 | 展开/折叠该任务的轮次列表；每轮显示 decision 徽章、假设、变更摘要、指标与 Δ、checkpoint sha 前 12 位 |
| 点击任务卡「Panel 讨论」 | 展开/折叠 panel 讨论，显示角色、立场、证据、置信度条与最终 verdict |
| 点击任意轮次行 | 打开右侧证据抽屉：完整假设/变更、全部指标、任务目标（GOAL）、内核轨迹（usage / latency / tool_calls / trace）、本轮 panel 结论、父/候选 checkpoint 完整 sha、时间 |
| 抽屉：`Esc` / 点击遮罩 / 右上角 × | 关闭抽屉并归还焦点 |
| 事件流级别 / 任务胶囊 | 前端过滤 `info / warn / error` 与 `A / B / C` |
| 悬停指标轨迹圆点 | 显示第几轮与指标值（原生 SVG `<title>`） |
| 悬停漂移仪表 | 显示漂移原因与证据文本 |
| 「软目标与批次」面板 | 每任务显示软目标、当前值、缺口（>0 为未达标）、degraded 徽章与证据，面板固定标注「软目标，不阻塞完成」；批次时间线逐批展示 `before → after` 指标、Δ 与 decision |
| 「漂移雷达与知识」面板 | 每任务 4 轴 SVG 雷达（严重度 / 特征位移 / KS max / 等级）、漂移类型与归因 chips；知识命中/文档/技能条；提案来源堆叠条（seed / agent / knowledge / panel） |
| 抽屉内滚动到 METRICS 之后 | 额外显示本任务软目标（目标/当前/缺口/verdict）与本轮提案来源占比 |
| 搜索框输入后按 Enter / 点「检索」 | 调用 `/api/records/search`，列出最多 20 条命中 records.jsonl / rounds.jsonl 的记录；`Esc` 或「清除结果」复位 |
| 查看任务卡头部 KERNEL / PIPELINE 徽章 | controller 指示：青色=自主内核，琥珀=流水线，取值来自 `/api/memory` 的 `run_agent.controller` |
| 断网 / 服务停止 | 顶部红色横幅提示，保留上次数据，每 2s 自动重试；恢复后横幅消失 |

### 空态与降级

- `state.json` 不存在：`GET /api/state` 返回 `503` JSON（含 `kind`、`state_path`、`hint`），页面显示连接失败横幅。
- `state.json` 短暂不可读 / 非法 JSON：服务端返回最近一次成功快照，`_meta.error` 非空，页面显示琥珀色降级横幅。
- 文件超过 8s 未更新：页脚「数据延迟」变琥珀；超过 30s 变红。
- 指标/轮次缺失：对应区域显示 `—` 与空态占位，不抛错、不白屏。

---

## 3. 状态数据契约（schema v1 / v2 / v3，向后兼容）

`state.json` 由 harness 用 **原子替换**（临时文件 + `os.replace`）写出，看板每次请求按 `(mtime_ns, size)` 判断是否重解析。
所有时间字段为 ISO 8601 字符串（建议 UTC，如 `2026-11-15T06:46:15+00:00`）；缺失字段按可选处理，前端做空值降级。

### 3.1 根对象

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | int | 是 | 当前为 `1`；看板显示在页脚，未知版本仍按 v1 容错渲染 |
| `run` | object | 是 | 运行级信息，见 3.2 |
| `tasks` | array | 是 | 三个任务卡，见 3.3；按数组顺序渲染（A/B/C） |
| `events` | array | 否 | 全局事件流，见 3.5；服务端会与 `records.jsonl` 合并并按 `ts/level/task/message` 去重 |

### 3.2 `run`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `run_id` | str | 运行标识，顶栏展示 |
| `status` | `running` / `finished` / `failed` / `stopped` | 控制状态灯颜色与呼吸动画 |
| `started_at` | ISO 8601 | 开始时间（页脚生成时间另用 `_meta.generated_at`） |
| `elapsed_seconds` | float | 已运行秒数；运行中前端每秒本地续算，不要求 harness 高频写文件 |
| `llm.backend` | `pi` / `local` / `openai-agents` | LLM 后端 |
| `llm.model` | str | 模型名 |
| `llm.calls` | int | 调用次数 |
| `llm.tokens_in` / `llm.tokens_out` | int | token 消耗 |
| `budget.tokens_used` / `tokens_cap` | int | 预算条 1 |
| `budget.cpu_seconds` / `cpu_cap` | float | 预算条 2 |
| `budget.wall_seconds` / `wall_cap` | float | 预算条 3 |
| `budget.experiments_used` / `experiments_cap` | int | 预算条 4 |

预算条百分比 = `used / cap`；`cap` 缺失或为 0 时显示 `—` 且不画进度。

### 3.3 `tasks[]`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `key` | `A` / `B` / `C` | 任务标识，决定卡片强调色（A 青 / B 琥珀 / C 蓝） |
| `name` / `name_en` | str | 中文名 / 英文名 |
| `task` | `classification` / `regression` | 任务类型 |
| `primary_metric` | `macro_f1` / `rmse` | 卡片主指标名；保留英文展示 |
| `direction` | `higher` / `lower` | 指标方向；决定 Δ 颜色与箭头（higher 越大越好） |
| `baseline` | float | 基线指标；sparkline 虚线、Δ 参照 |
| `best` | float \| null | 已接受最优；`null` 表示尚无提升 |
| `current` | float \| null | 当前生效指标（通常等于最优）；`null` 显示 `—` |
| `status` | `waiting` / `running` / `done` / `failed` | 任务状态芯片 |
| `round_index` | int | 当前/最近轮次序号（从 1 开始） |
| `rounds_total` | int | 计划总轮数（默认 10），进度环与 `done/轮数` 用 |
| `drift.severity` | `stable` / `mild` / `moderate` / `severe` | 漂移四段仪表当前档位与配色 |
| `drift.score` | float | 0~1 综合漂移分 |
| `drift.feature_shift` | float | 特征偏移量 |
| `drift.ks_max` | float | 最大 KS 统计量 |
| `drift.causes` | str[] | 漂移原因 chips（展示前 2 个，其余折叠为 `+N`） |
| `drift.evidence` | str | 证据文本（悬停仪表显示） |
| `budget.tokens_used` / `cpu_seconds` / `wall_seconds` / `experiments` | number | 卡片底部统计 |
| `rounds[]` | array | 轮次时间线，见 3.4 |
| `panel[]` | array | panel 讨论，见 3.6 |

### 3.4 `tasks[].rounds[]`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `index` | int | 轮次序号，展示为 `R{n}` |
| `id` | str | 稳定 id（抽屉定位用；可等于 `A-r1`） |
| `status` | `done` / `running` | 进行中显示「进行中」徽章且无指标 |
| `decision` | `promote` / `retain` / `rollback` \| null | 门控决策：提升/保留/回滚；`running` 时为 `null` |
| `hypothesis` | str | 本轮假设 |
| `change_summary` | str | 变更摘要 |
| `strategy` | str | 策略名（真实策略：`retain` / `continue_train` / `ridge_last_layer` / `swa`） |
| `postprocess` | str | 后处理（`none` / `recenter:batch_std` / `recenter:batch_mean` / `shrink(α)` 等） |
| `metrics` | object | 指标表；至少包含 `primary_metric`，可含 `accuracy`、`mae`、`nasa_score` 等；抽屉完整展示 |
| `parent_checkpoint_sha256` | str | 父检查点 sha256（完整 64 hex） |
| `candidate_checkpoint_sha256` | str | 候选检查点 sha256；时间线仅显示前 12 位，抽屉显示完整值 |
| `created_at` | ISO 8601 | 轮次完成时间 |

时间线中每轮的 `Δ` 为「本轮指标 − 当前已接受检查点指标」：`promote` 会推进已接受值，`retain` / `rollback` 不推进。

### 3.5 `events[]`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `ts` | ISO 8601 | 事件时间；服务端按时间倒序返回 |
| `level` | `info` / `warn` / `error` | 事件级别与配色 |
| `task` | `A` / `B` / `C` / `""` | 所属任务；空串表示全局 |
| `message` | str | 事件文本（中文） |

看板响应中的事件额外带 `source`（`state.json` 或 `records.jsonl`），仅用于展示 `jsonl` 小标签，不是磁盘 schema 的必填字段。

### 3.6 `tasks[].panel[]`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `trigger` | str | 触发原因（如「候选显著劣化，触发 rollback 复核」） |
| `round_index` | int | 关联轮次 |
| `verdict` | str | 最终结论（与 decision 枚举一致） |
| `messages[]` | array | 讨论消息 |
| `messages[].role` | `diagnostician` / `experimenter` / `critic` / `arbiter` | 角色：诊断者 / 实验者 / 评审者 / 仲裁者 |
| `messages[].standpoint` | str | 立场 |
| `messages[].evidence` | str | 证据 |
| `messages[].confidence` | float | 0~1 置信度，渲染为百分比与置信度条 |

### 3.7 看板附加字段（`_meta`，不写回 state.json）

`GET /api/state` 的响应 = 原始 state + `ok: true` + `_meta`：

| 字段 | 说明 |
| --- | --- |
| `dashboard_version` | 看板版本（页脚） |
| `generated_at` | 本次响应生成时间（页脚） |
| `state_path` / `records_path` | 绝对路径（页脚/数据链路） |
| `records_available` / `records_events` / `records_skipped_lines` | records.jsonl 是否可用、合并条数、坏行数 |
| `event_count` | 合并去重后的事件总数 |
| `schema_version` | 透传 state 的版本 |
| `stale_seconds` | state.json 的 mtime 距今秒数（页脚延迟） |
| `cache` | `hit` / `miss` / `degraded` |
| `error` | 降级时的错误说明（存在即页面显示琥珀横幅） |

**兼容性约定**：看板只读取以上字段，不依赖额外字段；harness 可以安全地追加新字段。`decision` 为 `null` 表示进行中；`best/current` 允许 `null`；`metrics` 允许任意键；缺失 `drift` / `panel` / `rounds` 时对应区域显示空态。

### 3.8 `records.jsonl`（可选事件源）

每行一个 JSON object；服务端兼容以下类型：

```jsonc
{"type":"event","ts":"2026-11-15T06:47:10+00:00","level":"info","task":"A","message":"任务 A 第 1 轮进入诊断阶段"}
{"type":"round","ts":"2026-11-15T06:50:03+00:00","task":"A","round_index":1,"decision":"promote","strategy":"ridge_last_layer","message":"A 第 1 轮完成：决策 promote"}
{"type":"run_start","ts":"2026-11-15T06:46:15+00:00","run_id":"rsi-20261115-1412"}
```

`round` / `run_start` 缺少 `message` 时服务端会合成可读文本；坏行跳过并计数（`_meta.records_skipped_lines`）。与 `state.events` 完全相同的记录（`ts/level/task/message` 四元组）只保留一条。

### 3.9 schema v2 增量字段（全部可选）

`schema_version=2` 在 v1 基础上**只追加字段**，v1 的 state.json 仍可直接展示（`/api/state` 原样返回，`/api/memory` 会降级为空记忆/无 sqlite）。

| 路径 | 字段 | 说明 |
| --- | --- | --- |
| `run` | `agent` | 内核统计：`controller`（`kernel` / `pipeline`）、`kernel_rounds`、`kernel_failures`、`pipeline_rounds`、`delegates`、`panels`、`memory_writes`、`skill_writes`、`kernel_tokens` |
| `run` | `memory` | 运行级记忆：`enabled`、`dir`、`db`、`entries`、`usage`（人类可读，如 `"23% — 931/4000 chars"`） |
| `tasks[]` | `controller` | 本任务使用的控制器；缺省回退 `run.agent.controller` |
| `tasks[]` | `goal` | 自主目标：`goal`、`status`、`rounds_done`、`rounds_target`、`subgoals[]`、`last_verdict`、`turns_used/max_turns`、`judge{done,reason}` 等 |
| `tasks[]` | `agent` | 任务级内核统计（结构同 `run.agent`，可为 `{}`） |
| `tasks[]` | `memory` | 任务记忆：`enabled`、`entries`、`snapshot_chars`、`review.lessons[]`（`text`/`wins`/`evidence`）、`curated[]`、`entries_after` |
| `tasks[]` | `strategy_stats[]` | 策略战绩：`strategy`、`postprocess_type`、`trials`、`promotions`、`mean_delta`、`best_delta`、`win_rate` |
| `rounds[]` | `agent` | 内核轨迹：`ok`、`fallback`、`error`、`decision`、`finished`、`usage{input,output,reasoning}`、`latency`、`tool_calls`、`trace[{tool,ok,detail,at}]`、`env{}` |

看板对每个字段做类型防御：字段缺失、`null` 或结构不符时该区块显示空态，不影响其他面板。

### 3.9.1 schema v3 增量字段（全部可选）

`schema_version=3` 继续**只追加字段**；v1/v2 的 state.json 仍可直接展示：`/api/goal` 在无 `goal_soft` 时返回 `tasks` 空数组与降级提示，`/api/knowledge` 返回 `knowledge.available=false`，页面显示空态而非报错。

| 路径 | 字段 | 说明 |
| --- | --- | --- |
| `tasks[]` | `goal_soft` | 软目标（仅展示，不阻塞完成）：`targets{metric:value}`、`current{metric:value}`、`gap`（>0 表示未达标）、`degraded`（bool）、`verdict`（`met` / `missed` / `unknown`）、`evidence`、`metric`、`direction`、`note` |
| `tasks[]` | `drift` / `drift_extra` | 漂移雷达数据：`type`（如 `covariate_shift` / `load_regime_shift` / `none`）、`severity`、`score`、`feature_shift`、`ks_max`、`attribution[]`、`causes[]`、`evidence` |
| `state` | `knowledge` | 知识层统计：`hits`（int / list / dict 均可）、`docs[{title,path,hits}]`、`skills_used[]`、`hits_detail` |
| `state` | `batches[]` / `batch_index` | 批次时间线：`index`、`task`、`before_metrics{}`、`after_metrics{}`、`decision`、`note`；`batch_index` 缺省时回退批次数量 |
| `rounds[]` | `agent.proposal_sources` | 提案来源计数：`{"seed":n,"agent":n,"knowledge":n,"panel":n}`；缺失时前端显示降级占位 |

`gap` 的符号约定在看板侧统一：目标为「越大越好」时 `gap = target - current`，为「越小越好」时 `gap = current - target`；`gap > 0` 一律表示未达标。`degraded` 同时接受布尔与字符串。

### 3.10 运行目录 sidecar 文件

| 文件 / 目录 | 内容 | 看板行为 |
| --- | --- | --- |
| `records.jsonl` | 每行一条 JSON：`type` ∈ `event / round / decision / panel / llm_call` 等 | 与 `state.events` 合并去重，`/api/events` + `/api/records/search` |
| `rounds.jsonl` | 每轮一行 `type=round` 记录 | `/api/records/search` 检索；坏行跳过 |
| `memory/MEMORY.md`、`memory/TASK_*.md` | 人类可读记忆快照（bullet 行） | `/api/memory.memory_files` 展示字符数与条目数 |
| `rsi_state.db` | SQLite：`lessons` / `strategy_stats` / `skill_usage` 等表 | 只读打开（`mode=ro`）；读失败时降级并从 state 的 `tasks[].memory.review.lessons` 回退 |

---

## 4. HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/`、`/index.html` | 看板页面（中文 UI） |
| GET | `/app.js`、`/style.css`、`/favicon.svg` | 静态资源白名单，无目录浏览 |
| GET | `/api/state` | 完整 state + `ok` + `_meta`；数据源不可用时 `503` |
| GET | `/api/memory` | 记忆与自主内核聚合：`run_agent`/`controller`、`memory`、三任务 `memory`+`goal`、`strategy_stats`、`lessons`（sqlite 优先，state 回退）、`skill_usage`、`memory_files`、`db{available}` |
| GET | `/api/agent?task=A` | 内核数据：`run_agent`、`task_agent`、`goal`、每轮 `agent`（usage/latency/tool_calls/trace/env/proposal_sources）；不带 `task` 返回三任务汇总；未知任务 `400` |
| GET | `/api/goal` | 软目标与批次：`soft_goal_mode`、`note`、`summary{tasks,goal_soft,degraded,batches}`、`batch_index`、`tasks[{key,...,goal_soft,drift,goal}]`、`batches[]`；缺失 v3 字段时返回降级结构（200） |
| GET | `/api/knowledge` | 知识/技能：`knowledge{available,hits,docs,skills_used,hits_detail}`、`skills_used`、`lessons`、`strategy_stats`、`memory_files`、`db{available}`；无数据时 200 + 空结构 |
| GET | `/api/records/search?q=&limit=` | 在 records.jsonl + rounds.jsonl 做大小写不敏感子串检索；`limit` 默认 20、最大 100；空 `q` 返回最近记录；返回 `items/total/files` |
| GET | `/api/events?limit=&level=&task=&q=` | 事件流；`limit` 默认 50、范围 1~500；返回 `events/total/returned/limit/filters/data_source` |
| GET | `/api/health` | 轻量健康检查（状态、事件数、运行状态） |
| GET | `/favicon.ico` | `204 No Content` |
| 其他 | — | `404` JSON：`{"ok":false,"error":"路径不存在","path":...}` |

所有响应带 `Cache-Control: no-store`、`X-Content-Type-Options: nosniff`、`Referrer-Policy: no-referrer`。
服务端使用 `ThreadingHTTPServer`，静态文件路径做白名单 + 根目录校验，防止目录穿越。

---

## 5. 视觉规范

### 5.1 设计令牌

| 令牌 | 值 | 用途 |
| --- | --- | --- |
| `--bg` | `#0b0f14` | 近黑底色（工业控制塔） |
| `--panel` / `--panel-raise` | `#101821` / `#131d27` | 卡片、面板 |
| `--text` / `--text-dim` / `--text-faint` | `#e6f1f5` / `#93a8b2` / `#5f7480` | 三级文字 |
| `--cyan` | `#38e1d0` | 主强调色（A 任务、运行中、主按钮、描线） |
| `--amber` | `#f5b13d` | 次强调色（B 任务、警告、预算 80%+） |
| `--green` / `--red` / `--gray` | `#3ddc97` / `#ff5d6c` / `#93a6b0` | promote / rollback / retain |
| `--blue` | `#5aa9ff` | C 任务、experimenter 角色 |
| `--purple` | `#b48cff` | arbiter 角色 |
| `--mono` | 系统等宽栈 | 所有数字（`font-variant-numeric: tabular-nums`） |
| `--sans` | `system-ui` 栈（含 PingFang / 雅黑 / Noto CJK） | 中文正文 |

### 5.2 语义色映射

- decision：`promote` 绿 / `retain` 灰 / `rollback` 红；时间线左侧 2px 色条同步。
- drift：`stable` 绿 → `mild` 黄 `#ffd166` → `moderate` 橙 `#ff9f5a` → `severe` 红；四段仪表点亮到当前档位，当前段发光。
- 事件：`info` 青 / `warn` 琥珀 / `error` 红。
- 预算：<80% 青、80~95% 琥珀、>=95% 红，进度条有慢速流光。
- 指标 Δ：按 `direction` 判定好坏，好绿、坏红、持平灰，同时用 ▲ / ▼ / ◆ 三重编码（不只靠颜色）。
- 提案来源：`seed` 青 / `agent` 琥珀 / `knowledge` 蓝 / `panel` 绿，堆叠条与图例同色；软目标 `missed` 红、`met` 绿、`unknown` 灰。

### 5.3 版式与响应式

- 内容最大宽度 1720px，1920 宽屏居中；`>=1440px` 任务三列，`901–1439px` 两列，`<=900px` 单列。
- 下方面板 `1.7fr / 1fr`，`<=1210px` 堆叠为单列；预算条 `<=1210px` 两列、`<=900px` 单列。
- 顶部品牌左侧、运行信息右侧并允许换行；`body { overflow-x: clip }` 防止横向滚动。
- 事件流最大高度 442px 内部滚动（自定义细滚动条）；抽屉宽度 `min(468px, 94vw)`。

### 5.4 动效与无障碍

- 状态呼吸灯（2~2.4s）、预算流光（4.2s）、抽屉滑入（0.24s）、时间线/panel 展开淡入（0.24s）、进度条宽度过渡（0.5~0.8s）。
- `@media (prefers-reduced-motion: reduce)` 下关闭全部动画/过渡/流光，并取消卡片与轮次的位移。
- 语义化标签：`header/main/section/article/footer/aside[role=dialog]`、`aria-live` 运行信息与事件列表、按钮 `aria-expanded`、抽屉 `aria-modal`、数字 `title` 提示。
- 动态文本全部经 `esc()` 转义后插入，无 `innerHTML` 注入外部 HTML 的路径。
- 页面无外部字体、图片、CDN；`http(s)://` 只出现在 SVG 的 `xmlns` 命名空间（`http://www.w3.org/2000/svg`，非网络请求）。

### 5.5 页面信息密度原则

1. **先结论后证据**：卡片首屏给指标大数字、Δ 基线、最优、当前轮次；细节收进时间线/抽屉。
2. **三重编码**：状态同时用颜色 + 文本 + 形状（圆点/箭头/徽章），色弱可读。
3. **每 2s 只重绘变化部分**：前端对 state 做签名比对，未变化时仅走 1s ticker 更新计时与数据延迟。
4. **失败不白屏**：任何缺失字段、503、非法 JSON 都有占位或横幅，保留上次快照。

---

## 6. 接入真实 harness

### 6.1 最小接入步骤

1. harness 用 `--state` 指向的路径**原子写** state.json（同目录临时文件 + `os.replace`），避免看板读到半个文件。
2. 事件逐条 append 到同目录 `records.jsonl`（也可只写 `state.events`，二选一或并用）。
3. 启动看板：
   ```bash
   ./.venv/bin/python -m rsi.dashboard --state /workspace/run/state.json --port 7861
   ```
4. 页面每 2s 轮询 `/api/state`；看板按 `(mtime_ns, size)` 缓存解析结果，文件未变时零解析开销。

### 6.2 写 state 的参考片段

```python
import json, os, tempfile
from pathlib import Path

def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # 原子替换：看板不会读到半截 JSON
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
```

### 6.3 容器内 / 离线部署

- 运行期只依赖 Python 标准库与静态文件，无 CDN、无 npm、无需联网。
- 容器内监听 `--host 0.0.0.0` 时，把 7861 映射到宿主机即可；默认 `127.0.0.1` 更安全。
- 不支持子路径前缀部署（静态资源与 API 均为根路径绝对引用）；如需要请加反向代理并改写，或复制静态目录到子路径。
- 页面无鉴权：只应绑定内网/本机，不暴露公网。

### 6.4 与 schema 的偏差 / 扩展

- 磁盘 `state.json` **未新增必填字段**；看板只在 API 响应中追加 `_meta` 和事件 `source`，不写回文件。
- 真实 harness 的 `strategy` / `postprocess` 为自由字符串，看板按原样展示，不做枚举校验。
- `rounds[].decision` 允许 `null`（进行中轮次），时间线显示「进行中」徽章；`metrics` 允许 `{}`。
- 事件 `task` 允许空串表示全局；前端按 `String(task).toUpperCase()` 匹配 A/B/C。
- `records.jsonl` 中 `type=round` / `type=run_start` 的记录由服务端归一为事件；真实 harness 如果直接写 `message` 字段则原样保留。
- schema v2 的 `agent` / `memory` / `goal` / `strategy_stats` 字段全部可选；看板接受 `schema_version` 1 或 2，缺失的内核区块显示空态占位。
- 三个 v2 只读 API 用独立缓存（按文件 mtime/size），`rsi_state.db` 以 SQLite 只读 URI（`mode=ro`，timeout 0.35s）打开；读取失败只影响 `/api/memory` 的 `db.available`，不会让整个看板报错。
- `run.memory.dir` / `run.memory.db` 允许写相对路径，服务端依次相对 state 目录、其父目录与当前工作目录解析。

---

## 7. 演示脚本建议（约 3 分钟）

| 时间 | 操作 | 讲解点 |
| --- | --- | --- |
| 0:00–0:15 | 打开 `http://127.0.0.1:7861/`，全屏 1440+ | 「这是 harness 的实时看板，三张卡片对应比赛三个工业任务，数据直接读 harness 原子写的 state.json，2 秒刷新一次」 |
| 0:15–0:40 | 指顶栏 + 预算条 | 运行状态灯、run_id、后端模型、调用量；四项预算消耗与 caps，实验次数 11/30 |
| 0:40–1:15 | 任务 A 卡片 | 大数字 0.7308 与 +2.35% 相对基线的绿色 Δ；sparkline 从 0.7140 爬升、第 4 轮回滚下探；漂移严重度 severe（batch1→batch2 漂移）与 causes chips |
| 1:15–1:45 | 展开 A 的「轮次时间线」 | 6 轮：4 promote / 1 rollback / 1 retain；每轮假设、策略、Δ 与 sha 前 12 位；点 rollback 轮次打开抽屉看完整证据与父检查点 |
| 1:45–2:05 | 展开 A 的「Panel 讨论」 | 诊断者 / 实验者 / 评审者 / 仲裁者的立场、证据、置信度，以及最终 verdict；体现「多角色审议 + 门控」 |
| 2:05–2:25 | 看 B/C 卡片 + 全局事件流 | B 轴承 RUL 仍在跑（第 4 轮运行中）；C 首轮提升后进入第 2 轮；事件流按 warn/error 过滤展示回滚与预算告警 |
| 2:25–2:50 | 记忆与自主内核面板 | MEMORY 用量条（931/4000 chars）、KERNEL 控制器徽章、内核轮次/派发/记忆写入计数；6 条 lessons（A 漂移、B 单轴承方差等）、策略胜率条形图、技能使用；在搜索框输入 `promote` 回车，展示 records / rounds 命中 |
| 2:50–3:10 | 软目标与批次 / 漂移雷达与知识 | 软目标卡：A 目标 0.76 / 当前 0.7308 / 缺口 0.0292 / degraded、「软目标，不阻塞完成」；批次 B1–B10 的 before→after 与决策；4 轴雷达展示 A severe、B mild、C stable；提案来源堆叠条说明 seed/agent/knowledge/panel 占比 |
| 3:10–3:25 | 打开 A 某一轮的抽屉 | 除假设/指标/sha 外，展示软目标、提案来源、任务目标与 KERNEL TRACE（8 步工具调用，含一次 `experiment.run` 失败回滚），体现「内核可审计」 |
| 3:25–3:35 | 页脚与「数据链路」 | 数据源路径、生成时间、schema 版本、数据延迟；强调只读文件、离线可运行 |

**可选实时感增强**：演示前先跑一个 5 行脚本，每 3 秒 `build_state()` + 原子写一次 `.scratch/demo_state.json`（把 `elapsed_seconds` 递增、给运行中轮次追加一条事件），页面会在 2s 内自动更新；停止脚本后页脚延迟会变琥珀/红，正好演示降级提示。

---

## 8. 截图占位

开发环境无浏览器，未提交像素级截图；演示/文档需要时按以下位置补图（1440×900 与 390×844 各一张）：

| 文件 | 场景 |
| --- | --- |
| `docs/assets/dashboard-1440-overview.png` | 1440 宽整页：顶栏、预算、三任务卡 |
| `docs/assets/dashboard-1440-timeline.png` | 展开 A 任务轮次时间线 + 漂移仪表 |
| `docs/assets/dashboard-1440-panel.png` | 展开 Panel 讨论（四角色 + verdict） |
| `docs/assets/dashboard-1440-drawer.png` | 单轮证据抽屉（sha 完整值、指标表、panel 结论） |
| `docs/assets/dashboard-1440-kernel.png` | 记忆与自主内核面板（MEMORY 条、lessons、策略胜率、检索结果） |
| `docs/assets/dashboard-390-mobile.png` | 390 宽单列响应式 |

> 截图占位：`docs/assets/` 目录当前不存在，以上为建议命名；补图后在此处改为 Markdown 图片引用。
> 截图时建议浏览器缩放 100%、禁用夜间模式扩展、窗口 1440×900，并保持运行中状态以获得呼吸灯/进度流光效果。

---

## 9. 测试与验收

```bash
# 单元/端到端（真实起服务 + 临时端口，测试结束自动清理进程）
./.venv/bin/python -m pytest tests/test_dashboard.py -q

# 手动验收
./.venv/bin/python -m rsi.dashboard --state .scratch/demo_state.json --port 7861 &
curl -s localhost:7861/api/state | head -c 200
curl -s localhost:7861/api/memory | head -c 200
curl -s "localhost:7861/api/agent?task=A" | head -c 200
curl -s "localhost:7861/api/records/search?q=promote&limit=3" | head -c 200
curl -s localhost:7861/api/goal | head -c 200
curl -s localhost:7861/api/knowledge | head -c 200
curl -s -o /dev/null -w "%{http_code}\n" localhost:7861/
kill %1

# 离线校验：三个静态资源不应出现 http(s)://（favicon 的 xmlns 除外）
grep -n "http" rsi/dashboard/static/index.html rsi/dashboard/static/app.js rsi/dashboard/static/style.css
```

`tests/test_dashboard.py` 覆盖：

1. `GET /` 200、含关键中文（含「记忆与自主内核」「MEMORY」）、引用本地 CSS/JS、无外部 URL；
2. `/app.js`、`/style.css`、`/favicon.svg` 200 且离线；`/favicon.ico` 204；
3. `/api/state` 200：schema v1/v2/v3、A/B/C 三任务、A 六轮、panel 结构、64 位 sha、`_meta`、records 合并；v3 批次字段；
4. 原子替换 state.json 后下一次请求读到新 run_id 与 elapsed（恢复原文件）；
5. `/api/events?limit=5`：数量、倒序、字段；`level/task` 过滤；
6. 404 JSON；`/api/health` ready；
7. `/api/memory`：schema v2、controller、MEMORY 用量百分比、三任务记忆与 goal、lessons/strategy_stats/skill_usage/memory_files、`db.available=true`；
8. `/api/memory` 无 sqlite 降级：`db.available=false`，lessons 从 `state.json` 回退（`source="state"`）；
9. `/api/agent?task=A`：6 轮、`trace[{tool,ok,detail,at}]`、usage/latency、goal；汇总接口三任务；未知任务 400；
10. `/api/records/search`：`q=promote&limit=5` 数量/倒序/字段、空 q 最近记录、大小写不敏感、无匹配 total=0；
11. v2/v3 测试数据由 `rsi.dashboard.demo.write_demo_bundle()` 在临时目录生成（含/不含 sqlite 两种）；
12. `/api/goal`：schema v3、`summary`、三任务 `goal_soft`（targets/current/gap/degraded/verdict）、批次 before/after/decision、漂移 type/attribution/hits；
13. `/api/goal` + `/api/knowledge` 在只有 v2 字段的 legacy state 下返回 200 降级结构（不报错、不白屏）；
14. `/api/knowledge`：`available=true`、`hits/docs/skills_used`、lessons 与 strategy_stats、`_meta.db_available`；
15. `/api/agent`：每轮 `agent.proposal_sources` 键域与数值合法性；
16. 测试进程与临时文件自动清理（当前共 17 个用例）。

浏览器侧人工检查（无自动化浏览器）：

- 1440×900 下无横向滚动（`document.documentElement.scrollWidth <= window.innerWidth`）；
- 三列任务卡不换行、卡片内无文本溢出（`text-overflow: ellipsis` 生效）；
- 展开时间线/panel、打开/关闭抽屉、过滤事件、断开服务后的横幅均正常；
- `prefers-reduced-motion: reduce` 下无动画。

---

## 10. 已知限制与后续

- 无 WebSocket/SSE：2s 轮询足够演示与调试；若状态文件很大可提高轮询间隔（前端 `POLL_MS`）。
- 单文件状态：同一时刻只展示一个 run；多 run 对比留待 v2。
- 时间线展开状态只保存在内存（刷新回到默认 A 展开）；如需持久化可用 `localStorage`。
- 抽屉中 panel 信息按 `round_index` 关联；若 harness 的 panel 没有 `round_index` 则只展示 verdict 区块（当前 mock 有）。
- 页面未做 i18n 切换：中文为主、指标名保留英文。
- 记忆 / 内核数据是只读展示：看板不写 `MEMORY.md`、不写 `rsi_state.db`；sqlite 只读连接失败时 `/api/memory` 降级而不是报错。
- 内核面板的 lessons / 策略 / 技能依赖 sidecar 文件；若 harness 只写 `state.json`，面板会回退到 `tasks[].memory.review.lessons` 与 `tasks[].strategy_stats`，其余区块显示空态。
- 软目标面板是**只读展示**：看板不参与 promote/retain/rollback 判定，`goal_soft.degraded` 与 `verdict` 由 harness 写入，缺失时一律显示「未提供」而不推断。
- 漂移雷达把 `score / feature_shift / ks_max` 归一化到 0~1，第四轴「等级」取 `severity` 档位（stable=0 → severe=3）；未提供这些字段时雷达为空心占位。
- `/api/knowledge` 的 `hits` 支持 int / list / dict 三种写法；若 harness 只写 `state.knowledge.hits` 而无 docs，页面用 `memory/` 文件列表兜底。
- 搜索为文件尾部 2 万行内的逐行 JSON 子串匹配（缓存按文件 mtime/size），适合演示与调试，不适合当作全量日志检索。
- E2E 无浏览器断言；若后续加入 CI 可补 `playwright`（需联网安装，当前离线环境不引入）。
