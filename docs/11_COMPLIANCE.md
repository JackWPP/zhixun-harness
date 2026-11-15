# 11 · 赛题合规映射

本文件逐条对照赛题规则、平台评分公式与主观 rubric，说明系统在何处满足、用哪条
证据自证。所有路径均相对仓库根；运行产物在 `run_dir/`（例：
`.scratch/rsi-full/rsi-20261115-154637/`）。

## 11.1 三条基础门槛

| 门槛 | 规则要求 | 本系统如何满足 | 证据 |
|---|---|---|---|
| G1 无人干预完成全部阶段 | 指定设备、统一离线环境、隐藏测试集上自动运行并输出 | `docker run` 单命令启动；`runner.py` 自动完成 3 任务 × 10 轮；LLM 缺失自动降级；无交互式输入 | `docker/entrypoint.sh`、`rsi/runner.py`、`docs/13_RUNBOOK.md` |
| G2 实验记录完整 | 全部实验按《实验记录规范》在 SwanLab 云端留存 | 每轮写 `rounds.jsonl`（假设/变更/指标/父子 sha/用量）；`summary.json` 与 `workspace_report.json` 对齐平台 schema；`--swanlab` 可离线记录并 `swanlab sync` | `rsi/records.py`、`rsi/runner.py::_record_round`、`docs/12_EXPERIMENTS.md` |
| G3 综合性能不低于静态基线 | 封闭评测不低于 baseline | A：selection 0.7032→0.8434，report 0.6864→0.8126，外部探针 0.4054→0.5239；B：678.02→462.45；C：44.75→41.82（report 57.46→56.28）；且门控保证最终 checkout 是"历史最优"而非最后一轮 | `summary.json`、`docs/12_EXPERIMENTS.md` §12.4 |

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
| autonomous_decision_and_improvement | 10 | 6 角色 LLM 编排：诊断→实验设计→评审→裁决；工具白名单；确定性降级保证不断链 |
| drift_detection | 8 | `rsi/drift.py`：均值/方差偏移、KS、相关性、标签分布 JS、窗口性能跌幅、severity、causes、top features |
| route_selection | 6 | 策略库分"模型层"（续训/ridge/SWA）与"推理期"（recenter/shrink）；Diagnostician 归因 + Experimenter 选路线 + Critic 审查 |
| safety_controls | 6 | Agent 无 shell/文件/网络工具；执行器二次校验；预算 tripwire；rollback；隐藏标签不可见；镜像离线 |
| system_quality | 6 | 19 项 pytest；模块分层；原子状态；重新加载 checker；文档齐全 |
| technical_report | 6 | `docs/12_EXPERIMENTS.md` + `README.md` + 架构文档 |
| controlled_experiments | 5 | 三层互斥切分 + 噪声阈值 margin + 固定 seed + 单变量策略库 + 失败实验留痕 |
| experiment_records | 5 | `records.jsonl` / `rounds.jsonl` / `state.json` / `workspace_report.json` / SwanLab 可选 |
| record_based_analysis | 5 | `summary.json` 的 narrative 由 Reporter 生成，引用具体轮次/指标/sha |
| industrial_constraints | 4 | 仅 CPU、NumPy、6h 预算、无外网、无 GPU、镜像 366MB、依赖固定 |
| defense | 4 | 文档 + 看板证据抽屉 + 每实验可复现目录 |

## 11.4 加分项

- **extended_data_validation（5）**：A 任务用 UCI Gas Sensor Array Drift 全部 10 批
  构建外部漂移探针（13,910 行，batch3–10 为漂移段），**仅用于验证、绝不训练**；
  最终模型 probe macro-F1 0.5239 vs 基线 0.4054。代码路径与训练完全分离，探针
  数据不进入 `train`/`calibration`。
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
2. **Harness 镜像**：`harness.tar`（`zhixun-rsi-harness:0.3.0`，amd64）；
3. **云端实验记录**：`workspace_report.json` / `rounds.jsonl`，可一键导入 SwanLab；
4. **运行说明**：`docs/13_RUNBOOK.md`（评测方按此启动即可）。

> 平台当前 `blockers: 正式评测暂未开放，请等待赛事通知`；镜像上传接口
> `POST /api/v1/image-archives` 已确认（Content-Type: application/x-tar）。
