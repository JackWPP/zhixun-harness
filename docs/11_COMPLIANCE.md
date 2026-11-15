# 11 · 赛题合规映射

本文件逐条对照赛题规则、平台评分公式与主观 rubric，说明系统在何处满足、用哪条
证据自证。所有路径均相对仓库根；运行产物在 `run_dir/`（例：
`.scratch/rsi-full/rsi-20261115-154637/`）。

## 11.1 三条基础门槛

| 门槛 | 规则要求 | 本系统如何满足 | 证据 |
|---|---|---|---|
| G1 无人干预完成全部阶段 | 指定设备、统一离线环境、隐藏测试集上自动运行并输出 | `docker run` 单命令启动；默认 kernel 控制器由 LLM 自主决策，LLM 缺失/超时自动降级 pipeline；启动先跑资产自检；无交互式输入 | `docker/entrypoint.sh`、`rsi/runner.py`、`rsi/agent/kernel.py`、`scripts/verify_assets.py`、`docs/13_RUNBOOK.md` |
| G2 实验记录完整 | 全部实验按《实验记录规范》在 SwanLab 云端留存 | 每轮写 `rounds.jsonl`（假设/变更/指标/父子 sha/agent trace/记忆用量）；同时落 SQLite（rounds/experiments/llm_calls/lessons/strategy_stats），JSONL 为审计真源、DB 可重建；`summary.json` 与 `workspace_report.json` 对齐平台 schema；`--swanlab` 可离线记录并 `swanlab sync` | `rsi/records.py`、`rsi/memory/db.py`、`rsi/runner.py::_record_round`、`docs/12_EXPERIMENTS.md`、`docs/17_MEMORY.md` |
| G3 综合性能不低于静态基线 | 封闭评测不低于 baseline | 严格官方数据口径（§12.19）：A selection 0.7032→0.7346（report 0.6864→0.6834），B 678.02→409.74（report 664.0→424.1），C 44.75→40.52（report 57.46→56.02）；A 外部探针 0.4054→0.5239；门控保证最终 checkout 是"历史最优"而非最后一轮 | `summary.json`、`docs/12_EXPERIMENTS.md` §12.19 |

## 11.2 客观分（35 分）

平台公式 `contest-objective-20261110-rul-v3`，权重与系统的对应机制：

| 维度 | 权重 | 系统机制 |
|---|---|---|
| mean_gain（平均提升） | 15 | 每轮从白名单策略中选候选；A 的 recenter_blend、B 的 shrink、C 的受限续训都是本地实测有效方向；门控只接受超过噪声阈值的提升 |
| worst_stage（最差阶段） | 5 | 漂移诊断 severity + 窗口性能跌幅；预算不够时优先保底（retain），绝不让指标自由坠落 |
| recovery（恢复） | 5 | 劣化触发 rollback 到父模型；连续无提升触发 Panel + exploration；C/A 的续训可把性能拉回 |
| retention（保持） | 5 | 每轮 promote 前先计算候选在 report 切分的表现；未提升则 retain 父模型；checkpoint 链可追溯 |
| resource_efficiency（资源效率） | 5 | 预算：6h wall / 每任务 40 实验 / 2M token / CPU 秒记账；A 的收益主要来自推理期校准，**零额外训练成本**；单实验 1800s 超时 |

## 11.3 主观分（65 分）

| rubric 项 | 分值 | 本系统对应实现 |
|---|---|---|
| autonomous_decision_and_improvement | 10 | Agent Kernel 持有目标+18 个受限工具，自己看结果、改计划、跨轮写记忆；6 角色可 delegate/panel；Guard 确定性边界；LLM 缺失自动降级 pipeline。证据：A 任务 3 轮 18/14/14 工具调用、promote→promote→retain（`docs/18` §10.1） |
| drift_detection | 8 | `rsi/drift.py`：均值/方差偏移、KS、相关性、标签分布 JS、窗口性能跌幅、severity、causes、top features |
| route_selection | 6 | 策略库分"模型层"（续训/ridge/SWA）与"推理期"（recenter/shrink）；Diagnostician 归因 + Experimenter 选路线 + Critic 审查 |
| safety_controls | 6 | Agent 无 shell/文件/网络工具；执行器二次校验；预算 tripwire；rollback；隐藏标签不可见；镜像离线 |
| system_quality | 6 | 57 项 pytest（核心/记忆/看板/适配器/提交包）；模块分层；原子状态；checkpoint 重加载校验；docs/10–18 齐全 |
| technical_report | 6 | `docs/12_EXPERIMENTS.md` + `README.md` + 架构文档 |
| controlled_experiments | 5 | 三层互斥切分 + 噪声阈值 margin + 固定 seed + 单变量策略库 + 失败实验留痕 |
| experiment_records | 5 | `records.jsonl` / `rounds.jsonl` / `state.json` / `workspace_report.json` / SwanLab 可选 |
| record_based_analysis | 5 | `summary.json` 的 narrative 由 Reporter 生成，引用具体轮次/指标/sha |
| industrial_constraints | 4 | 仅 CPU、NumPy、6h 预算、无外网、无 GPU、镜像约 370MB、依赖固定；时间感知降级保证慢机器也能出 10 轮 |
| defense | 4 | 文档 + 看板证据抽屉 + 每实验可复现目录 |

## 11.4 加分项

- **extended_data_validation（5）**：A 任务用 UCI Gas Sensor Array Drift 全部 10 批
  构建外部漂移探针（13,910 行，batch3–10 为漂移段），**仅用于验证、绝不训练**；
  微加权 macro-F1 0.4252→0.5281（逐批等权 0.4054→0.5239）。脚本
  `scripts/external_validation.py` 可复现，报告 `docs/evidence/uci_gas_drift.md`；
  另有 `probe`/`synth` 子命令支持任意私有 JSONL 与合成漂移探针。
- **记忆与技能（申报 technical_innovation 的一部分）**：Hermes 风格的有界
  MEMORY/TASK 文件 + SKILL.md 渐进披露 + SQLite 经验库；跨 run 复用有实测证据
  （第二次运行首轮工具调用 18→13，直接复用已验证配置，见 `docs/18` §10.2）；
  记忆有注入扫描、漂移备份、容量上限，且可 `--no-memory` 消融。
- **technical_innovation（5）**：①受控 RSI：LLM 只能在白名单策略与参数区间内做
  选择，执行器二次校验，Critic 有 veto；②推理期 recenter 校准作为"零 checkpoint
  字节"的自适应机制，在漂移批次上带来 +0.10–0.12 而不改变模型文件；③确定性
  降级阶梯保证离线封闭环境仍有完整产出。

## 11.5 禁止事项自检

| 禁止项 | 自查结果 |
|---|---|
| 攻击评测平台 / 获取隐藏测试标签 | 无任何平台 API 写调用；仓库代码无网络请求；探针数据只有公开 UCI |
| 无限恶意迭代 | 每任务最多 10 轮、40 实验（三任务合计 ≤120）、2M token、6h；Panel 最多 2 次 |
| 植入恶意代码 / 未授权外网访问 | Agent 无 shell/网络工具；镜像内无出站请求；`grep -R "requests\|urllib\|curl" rsi/` 只剩注释 |
| 引用开源不遵守许可证 | 见 `docs/15_DATA.md` 与 README 许可节 |
| 改网络结构冒充续训 | 策略库禁止改结构；checkpoint `meta.task/classes/width` 不变；显式权重迁移不在实现范围内 |
| 提交源码 ZIP 冒充镜像 | 只提交 `docker save` 兼容的未压缩 tar；`docker load` 已实测 |

## 11.6 提交物清单

1. **模型配置**：最终 `checkpoint.npz` + `postprocess.json`（推理期校准参数）+ 父
   checkpoint sha256 链（在 `workspace_report.json` 的 `rounds[].parent_checkpoint_sha256`）；
2. **Harness 镜像**：`harness.tar`（`zhixun-rsi-harness:0.5.1`，amd64）；
3. **云端实验记录**：`workspace_report.json` / `rounds.jsonl`，可一键导入 SwanLab；
4. **运行说明**：`docs/13_RUNBOOK.md`（评测方按此启动即可）；
5. **提交包**：`scripts/make_submission.py` 从完成的 run 导出 `dist/submission_<run>/`
   （每任务 checkpoint + postprocess + 独立 predict + manifest，≤64MiB），
   `predict.py` 已在包内独立跑通 A/B/C 样例。

> 平台当前 `blockers: 正式评测暂未开放，请等待赛事通知`；镜像上传接口
> `POST /api/v1/image-archives` 已确认（Content-Type: application/x-tar）。

## 11.9 外部数据与模型兼容披露（v0.5）

**模型兼容**：官方 `TRAINING_GUIDE.md` 明确允许修改代码/超参；改变结构、输入维
度或任务时必须显式实现兼容的权重迁移，且不得把新模型声明为原模型续训。当前
v0.5 的 `extended_pretrain` 只做**同结构续训**（复制父权重后在外部数据上预训练、
再回到官方 train 微调），不改 width/classes/task，参考加载器可直接接受；
结构变更/宽层迁移在 v0.5 中**未启用**；如后续启用，必须记录父/候选参数 sha 并在 predict 入口保留 strict 回退。

**外部数据**：任务 A 的 `extended_pretrain` 使用 UCI Gas Sensor Array Drift 的
batch3–7（公开数据，随镜像 `extra/uci-gas/` 打包）。该策略**默认关闭**
（`RSI_ALLOW_EXTENDED_DATA=0`；设为 1 仅用于消融/扩展数据验证复现），
正式评分路径不启用，仅用于预训练工业模型权重的消融实验；
验证集/隐藏测试/校准 split 的标签在任何阶段都不参与拟合（`rsi/policy.py` 与
Guard 双重强制）。报告与 `workspace_report.json` 会标注外部数据的用途、批次与
隔离说明；关闭开关后流程仍完整可跑（仅性能不同）。

## 11.10 批次流与提交契约（v0.6）

* 官方答疑明确：评测按批次下发数据、**每批都带标签**、先输出“采用哪个模型/不上线”决策才
  下发下一批。实现见 `rsi/stream/`：每批 materialise 小 package → 复用完整 `Runner(rounds=1)`
  → 输出 `decision.json`（扁平平台字段 + `decision_detail`）与 `predictions.jsonl`（官方 predict
  形状）。协议文档：`docs/21_STREAM_PROTOCOL.md`；契约快照与未知项：`docs/19_SUBMISSION_CONTRACT.md`。
* 镜像 entrypoint 支持 `stream` 子命令；本地/平台 emulator：`scripts/emulate_stream_platform.py`
  （A/B 10 批 demo + 真实 CLI 2 批均 PASS，0 errors）。
* 对口合规：Track S（正式）只用官方 train；Track X 外部数据测试床不进入提交谱系；
  批次输出不包含隐藏标签；`scripts/contract_check.py` 复核规则/公告漂移与 baseline 三方一致。

## 11.11 知识库、软目标与记录（v0.6）

* `knowledge/`：12 篇策展离线文档（漂移分类/检测统计/残差监控/归因/校准迁移/域适应/RUL 校准等），
  由 `rsi/knowledge.py` 建 FTS5 索引；Agent 通过 `search_knowledge` 检索，prompt 注入 ≤1.8K 字符，
  内容视为参考数据而非指令；检索次数/hits 写入 state 与看板。
* `goal_soft`：A 0.7346 / B 409.74 / C 40.524 只是**软目标**（advisory），不阻塞收尾；
  硬不变量 = 有效输出 + 时间/实验/token 预算 + 严格数据合规 + 完整记录（用户 m03315 要求，
  测试 `tests/test_goal_nonblocking.py`）。
* SwanLab：镜像默认 offline 记录每轮训练/决策/指标；操作者可用 `swanlab sync` 公开上传
  （token 只在环境变量，绝不落仓库/镜像），见 `docs/22_SWANLAB.md`。
