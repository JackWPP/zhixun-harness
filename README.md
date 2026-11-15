# 智训引擎 RSI Harness —— 工业模型受控递归自我改进

> 北京市大学生"人工智能+"创新大赛 · 企业赛道（软通计算机 / 华为技术）
> 赛题：在漂移数据流上无人干预地维持并提升工业模型，提交 **模型配置 + Harness 系统镜像 + 云端实验记录**。

本仓库是参赛系统的完整实现：一个由 LLM Agent 驱动、在 Docker 内离线自主完成
**漂移监测 → 诊断归因 → 实验设计 → 沙箱执行 → 门控上线 → 记录归档** 闭环的
受控 RSI（Recursive Self-Improvement）harness。工业模型始终是官方基线的**续训
checkpoint**，不改变网络结构、不改变归一化协议，保证与平台加载器兼容。

---

## 1. 交付物

| 交付物 | 位置 | 说明 |
|---|---|---|
| Harness 镜像 | `docker/` + `scripts/build_image_tar.py` | 单个 `linux/amd64`、`docker save` 兼容的未压缩 tar（`harness.tar`），`docker load` 已验证 |
| Agent 核心 | `rsi/agent/` | 基于 pi-agent-core-lc 的 PI 运行时适配、6 角色编排、工具契约、Hook 预算闸门 |
| 自主闭环 | `rsi/runner.py` | 3 任务 × 10 轮，诊断/实验/评审/裁决/门控/记录/回滚 |
| 实验沙箱 | `rsi/executor.py` + `rsi/worker.py` | 每个候选在独立子进程执行，资源/超时/参数范围强约束 |
| 策略库 | `rsi/strategies.py` | retain / continue_train / ridge_last_layer / SWA × 推理期 recenter/shrink 校准 |
| 推理入口 | `rsi/predict.py` | 官方兼容 stdin/stdout 逐行预测，另提供 `rsi-predict` 容器入口 |
| 实时看板 | `rsi/dashboard/` | 零依赖 WebUI，轮次时间线、漂移仪表、Panel 讨论、sha256 证据抽屉 |
| 实验记录 | `<run>/records.jsonl`、`rounds.jsonl`、`summary.json`、`workspace_report.json` | 与平台 `workspace_report` schema 对齐 |
| 文档 | `docs/10_*.md` … `docs/16_*.md` | 架构、合规映射、实验报告、运行手册、Agent 设计、数据与决策记录 |

---

## 2. 三分钟跑起来（本地，无需 GPU / Docker）

```bash
# 一次性环境（uv 安装到仓库内，不污染系统）
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$PWD/.scratch/uv/bin sh
export UV_CACHE_DIR=$PWD/.scratch/uv/cache UV_PYTHON_INSTALL_DIR=$PWD/.scratch/uv/python
.scratch/uv/bin/uv python install 3.12
.scratch/uv/bin/uv venv .venv --python .scratch/uv/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12
VIRTUAL_ENV=$PWD/.venv .scratch/uv/bin/uv pip install -r docker/requirements.txt pytest

# 自检：加载官方 checkpoint、校验 sha、在分层切分上复算指标
./.venv/bin/python -m rsi.cli selftest --task A

# 完全确定性离线闭环（无 LLM，2 轮，A 任务）
./.venv/bin/python -m rsi.cli run --tasks A --rounds 2 --llm off

# 接真实 LLM（OpenAI 兼容端点；key 只从环境/本地 env 文件读取）
cp .scratch/llm.env.example .scratch/llm.env   # 填 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
./.venv/bin/python -m rsi.cli run --tasks A --rounds 1 --llm on --llm-env .scratch/llm.env

# 看板（读取运行中的 state.json）
./.venv/bin/python -m rsi.dashboard --state <run_dir>/state.json --port 7861
```

## 3. 构建并验证比赛镜像

```bash
# 1) 导出基础 rootfs（开发机无 rootful Docker，用 crane 匿名拉取）
crane export python:3.12-slim | tar -x --no-same-owner -C .scratch/image/rootfs
# 2) 组装 overlay（代码 + 官方数据集 + 依赖 + 入口）
bash scripts/stage_image.sh
# 3) 生成 docker save 兼容 tar
./.venv/bin/python scripts/build_image_tar.py \
    --rootfs .scratch/image/rootfs --overlay .scratch/image/overlay \
    --output .scratch/image/harness.tar --tag zhixun-rsi-harness:0.3.0
# 4) 验证（任意有 Docker 的机器）
docker load -i harness.tar && docker images | grep zhixun
```

镜像默认行为：`docker run -v /host/work:/work zhixun-rsi-harness:0.3.0` 会以
`RSI_TASKS=A,B,C RSI_ROUNDS=10 RSI_LLM=auto` 无人干预跑完 3 任务 × 10 轮，
把 `state.json / summary.json / workspace_report.json / records.jsonl` 写到
`/work/rsi-runs/<run-id>/`，并在 `/work/` 留一份固定路径的副本。所有行为可用
环境变量覆盖（见 `docker/entrypoint.sh` 与 `docs/13_RUNBOOK.md`）。

---

## 4. 系统做了什么（一句话版本）

每个任务每一轮：

1. **诊断**（Diagnostician）：读取漂移报告（特征均值/方差偏移、KS、相关性、
   标签分布、窗口性能跌幅），产出结构化归因与置信度；
2. **实验**（Experimenter）：只能在白名单策略库与参数区间内提 1–3 个候选，
   每个候选带假设、预期收益、风险；
3. **评审**（Critic）：否决越界/与假设矛盾/无信息量的实验，批准子集；
4. **执行**（Executor → Worker 子进程）：在隔离进程中训练/校准，产出新
   checkpoint（或只改推理期校准），计算 selection/report/probe 指标；
5. **门控**（Selector + 可选 Arbiter）：相对父模型提升超过噪声阈值才 promote，
   劣化超过 2× 阈值则 rollback，否则 retain；
6. **记录**（Recorder）：轮次假设、变更、父/候选 checkpoint 与代码 sha256、
   CPU/墙钟/token 用量写入 JSONL 与 `state.json`；
7. **卡住时**（连续 2 轮无提升或预算 80%）：触发 4 角色 Panel 讨论并切到
   exploration 策略；仍无改善则保留最优父模型安全退出。

LLM 不可用（离线/端点故障/预算耗尽/tripwire）时，整条链路自动降级为确定性
planner，仍能完成全部 10 轮并产出合规记录——这是镜像在封闭评测环境中不会
"卡死"的兜底。

---

## 5. 实验证据（本地 6h 预算内的真实运行）

| 任务 | 指标 | 官方基线 | 官方验证集 | 本轮闭环最终（selection 切分） | 外部漂移探针（UCI batch3–10） |
|---|---|---|---|---|---|
| A 气体分类 | macro-F1 ↑ | 0.714030 | 0.714030 | **0.843359**（10 轮，4 次 promote） | 0.4054 → **0.5239**（基线→最终，含 recenter） |
| B 轴承 RUL | RMSE ↓ | 681.160 | 681.160 | **462.454**（shrink 校准，1 轮提升后收敛） | —（轴承级留出，见 docs/12） |
| C 发动机 RUL | RMSE ↓ | 52.869 | 52.869 | **41.816**（3 轮 continue_train 后收敛；report 切分 57.46→56.28） | —（单元级留出） |

> 表中的 selection/report 是官方验证集按任务语义切出的**互不重叠**评测子集
> （A 按类别分层、B 轴承内行切分、C 整发动机单元切分），用于门控/报告；
> 外部探针只用于开发期证据，绝不参与训练或选型。详见 `docs/12_EXPERIMENTS.md`。

---

## 6. 赛题合规要点

- **模型层**：所有候选都从官方 baseline checkpoint 续训（`--from-baseline` 语义，
  `contest.cpu-mlp.v1` 格式/归一化/结构不变），提交时保留父 checkpoint sha256 链；
- **策略层**：LLM 只能在白名单策略 + 参数区间内选择，执行器二次校验；预算
  （6h wall / 每任务 40 实验 / 2M token / CPU 秒）硬上限；无隐藏标签访问；
- **实验记录**：每轮记录假设、修改、指标、父子 sha、调用用量，字段与平台
  `workspace_report` 对齐，可一键导出 SwanLab 离线记录；
- **离线**：镜像不依赖外网；LLM 可选，缺失时确定性降级；不写系统目录，不拉
  外部代码；
- **可复现**：镜像内固定依赖版本、固定数据集 sha、固定 seed；`selftest` 一条
  命令复算全部基线指标。

---

## 7. 仓库结构

```
rsi/                    新 harness（本仓库主角）
  agent/                PI 运行时、提示词、工具、协调器（6 角色）
  runner.py             主闭环：3 任务 × N 轮
  executor.py/worker.py 隔离实验执行（子进程 + 资源限制）
  strategies.py         白名单策略库（模型层 + 推理期校准）
  inference.py          recenter / shrink / 流式因果校准
  selection.py          门控（promote / retain / rollback）
  records.py            JSONL + state.json 记录器
  dashboard/            零依赖实时看板
  predict.py            官方兼容预测入口
  cli.py                run / predict / report / serve / selftest
docker/                 Dockerfile、requirements、容器入口
scripts/                镜像构建（无 Docker daemon 亦可产出 docker save tar）
datasets/               官方三任务数据包（A/B/C，含 baseline checkpoint）
extra/uci-gas/          扩展验证数据（UCI Gas Sensor Drift，CC BY 4.0）
harness/                （历史）旧原型：确定性规则引擎，仅作审计留档
audit/                  （历史）对旧原型的 10 项独立审计
docs/01–09              （历史）旧原型的移交文档与风险清单
docs/10–16              新系统的架构/合规/实验/运行/Agent/数据/决策文档
demo/harness-runs/      （历史）旧原型三次演示留痕
```

---

## 8. 旧原型审计（为什么重写）

移交仓库（2 个 commit）里的 `harness/` 是**确定性策略路由器 + 受控续训器**，不是
Agent/RSI：6 个硬编码配方、lr 恒为父模型 0.3×、卡住直接 `break`、策略小模型从未被
调用、梯度统计恒 0、验证集既选又报。审计证据保留在 `docs/04_harness_audit.md` 与
`audit/`。新系统保留其合理内核（训练角色隔离、归一化冻结校验、父 checkpoint 不
变性、预算记账、交付自检、确定性），替换掉全部决策与学习逻辑。

---

## 9. 已知限制与下一步

- 官方正式评测尚未开放（平台 `blockers: 正式评测暂未开放，请等待赛事通知`），
  当前以 workspace 十轮实验记录 + 平台 schema 对齐为验收依据；
- A 的验证集是单批次（batch2）且 Toluene 仅 5 条，selection 切分的方差较大；
  外部 UCI batch3–10 探针显示最终模型仍优于基线，但绝对分数受类别极端不平衡影响；
- B 只有 3 个验证轴承，行级切分无法完全消除轴承内泄漏，已在 `docs/12` 中如实
  披露并用 LOBO 实验给出保守结论；
- SwanLab 默认离线，`SWANLAB_API_KEY` 设置后可 `--swanlab` 同步；
- 镜像当前 366 MB（`docker load` 验证通过），可进一步裁剪 PI/langchain 依赖。

---

## 10. 许可与引用

- 代码：竞赛团队内部交付，未授权不得外传；
- 官方数据包：来自赛事平台，使用遵守赛事协议；
- UCI Gas Sensor Array Drift（#224）：CC BY 4.0，引用 Rodriguez et al.；
- XJTU-SY、NASA C-MAPSS：公开学术数据集，按各自许可引用；
- 第三方库：NumPy（BSD）、pi-agent-core-lc（MIT）、langchain-openai（MIT）、
  SwanLab（Apache-2.0），完整清单见 `docker/requirements.txt`。
