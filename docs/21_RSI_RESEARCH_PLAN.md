# 21 · RSI 能力与性能提升：调研结论和实验计划

日期：2026-11-16  
范围：比赛规则 PDF、培训逐字稿/时间轴/纪要、当前 checkout、公开研究与官方技术文档。  
状态：研究与计划；本文件不记录本轮新代码改动、LLM 试验或平台试运行。

## 结论先行

下一步重点应从“增加更多策略或多跑几轮”转向**先让每一轮成为真实、可重放、可归因的流式 RSI 决策**，再提升工业模型表现。当前已有漂移指标、候选策略、确定性门控、模型记忆和技能库，但当前本地 runner 仍按公开数据随机抽窗离线诊断；诊断同时能读取目标标签和事后性能。它不能证明在标签尚不可用的比赛时序中完成了监测、诊断、实验、部署和回滚。

培训材料要求先决策再获取下一批、重放要幂等、至少完成三次有历史依据和实验记录的迭代；培训口头说明不替代 SDK 与提交页。检索当前工作树没有 `contest.harness.workspace.v2` 的 `start/candidate/save_state/complete` 适配实现，因此真实协议和标签揭示时点仍是高成本实验前的第一道闸门。

## 证据与当前基线

### 赛题和培训材料

- 培训核对记录指出：作品允许以 Harness/技能策略为主要改进路线，不强制微调驱动 LLM；正式运行有 CPU/内存、镜像和允许模型服务限制；应记录工业模型的训练、调整、评估与决策；重复数据批次必须按提交时序处理。见 [docs/20_TRAINING_REVIEW.md](20_TRAINING_REVIEW.md) §1–3。
- 每轮迭代应包括“读历史 → 提假设 → 做实验 → 根据结果决策”，promote、retain、rollback 都是有效决策。三轮不等于三次都要上线。
- 规则与培训资料中的平台输入输出和标签解封时序必须以官方 SDK/平台当前文件为准。不得根据我们自己的回放假设，把当前批标签拿来先训练再预测。
- 对群里问到的 `contest.harness.workspace.v2`：在当前仓库源码/资料中没有该接口，也没有 `start/candidate/save_state/complete` 适配器；公开网页搜索精确符号亦未找到官方说明。公开 PyTRiO 文档说明的是大模型后训练 API，不能据此推导比赛 workspace 契约。因此如果评测运行时要求这个协议，我们也需要对应 SDK/说明；目前判断是“尚未实现、存在同类风险”，不是断言所有比赛镜像必然都会遇到该报错。获得官方 runner/SDK 文件后先做契约适配，不凭名称猜签名。

### 当前实现

- [rsi/runner.py](../rsi/runner.py) 的 `drift_report()` 从 selection 集随机排列取逐渐扩大的窗口，而非按平台批次顺序消费；它调用 `analyse()` 时目标数据含标签。
- [rsi/drift.py](../rsi/drift.py) 同时计算特征偏移、逐特征 KS、相关性、标签分布变化和性能变化。后两类指标只有标签解封后才能得到。严重度阈值是启发式值，尚未用无漂移/已知漂移流校准误报率与检测延迟。
- [rsi/selection.py](../rsi/selection.py) 提供确定性门控，但目前基于一个冻结 selection 主指标和固定 margin；还没有按批次、设备/发动机单元、随机种子估计不确定性。
- [rsi/strategies.py](../rsi/strategies.py) 已有受控续训、SWA、ridge 和推理期 `recenter` / `shrink` / `damage_shrink`。策略库是可用起点，不代表其收益能跨批次稳定复现。
- 技能/记忆可写入 run 私有副本，且有输入扫描；但是要让“RSI 进化”可归因，候选技能必须版本化隔离、与当前策略并行回放，经过独立门控后才能生效。不可边运行边修改当前轮的行为。
- [rsi/agent/runtime.py](../rsi/agent/runtime.py) 的 token 用量在 agent 整次调用结束后汇总；开始下一次 LLM 调用前检查累计预算，不能保证单次长调用不超过硬预算。修好逐响应硬停止前，不跑高成本付费 Kernel 轮。
- [rsi/runner.py](../rsi/runner.py) 目前 SwanLab 使用 offline mode，且只记录少量轮级指标。完整本地 JSONL/SQLite 有价值，但不能据此声称已有公开云端实验记录。

### 目前可用和不可用的性能证据

- 独立本地复核确认 A/B/C 官方包内 checkpoint 可以加载，且与包内摘要相符；现存确定性 pipeline 结果能够复算。但这些 run 是 `controller=pipeline`，没有 LLM Kernel 轮次、skill 写入或 Agent 自主提出策略，不能算作自主 RSI 成果。
- 本地早期实验文档记录过 A/B/C 的较大 selection 改善，但其中一些 canonical run 目录与镜像已经不在当前 checkout。该文档属于待复现的历史记录，不作为当前线上效果或稳定性证据。A 的历史结果也有 selection/report 差异和跨 seed 不稳定风险；B 的历史诊断显示设备分布与特征/目标关系是主要风险；C 不应为了每轮更新而忽略 retain。
- 2026-11-16 的 Kernel 冒烟在 A 上最后 retain 于 baseline，是有效负结果；当前记录还不构成跨任务、跨批次的自主性能提升证据。详见 [docs/20_TRAINING_REVIEW.md](20_TRAINING_REVIEW.md) §2。

## 外部研究给出的可迁移做法

下列来源只用于设计本地方法；论文数据集成绩不代表本赛题可复制的收益。

1. **在线漂移检测要遵循数据流时序。** Bifet 与 Gavaldà 的 ADWIN 为数据序列维护自适应窗口，通过比较两个子窗口检测分布变化；River 文档还暴露显著性、检查频率和最小窗口等控制项。适合评估它是否能监测一个有界的标量信号（例如已揭示标签后的 prequential loss），不应把它当作高维传感器问题的一键归因器。来源：[原始论文 DOI](https://doi.org/10.1137/1.9781611972771.42)，[River ADWIN 文档](https://riverml.xyz/dev/api/drift/ADWIN/)。
2. **高维逐特征检测须控制多重比较和误报。** Alibi-Detect 文档说明多变量逐特征检验可用 Bonferroni 或 FDR 汇总，也可先对高维表示做降维；Hinder 等人的综述提供无监督漂移检测及其实验比较背景。因此 A 的 `max_feature_shift` 不能单独驱动严重告警；要用异常值稳健的分位数/中位数摘要、联合信号和无漂移 replay 校准误报。来源：[Alibi-Detect 多变量漂移说明](https://docs.seldon.ai/alibi-detect/cd/methods/offline/tabulardrift)，[Hinder 等，2024，Part A](https://doi.org/10.3389/frai.2024.1330257)。
3. **标签延迟评估要先记预测、后记损失。** Grzenda 等人研究验证延迟的流式评估，明确区分“何时预测”和“何时真实标签可用”，并讨论 test-then-train / prequential 等方案。比赛实现应为每条预测保留时序和模型版本；标签到达后再更新损失监控、允许的训练状态和 checkpoint。来源：[Delayed labelling evaluation for data streams](https://doi.org/10.1007/s10618-019-00654-y)。
4. **部署应像 canary 一样对照 champion。** Google SRE 的 canary 指南要求候选与 control 在同一时间窗口比较，定义异常阈值和回滚动作。对于本赛题，这意味着候选先影子评估或小批验证，按批监控主指标、最差阶段、时延和资源；出现越界就恢复已知 champion。来源：[Google SRE Workbook: Canarying Releases](https://sre.google/workbook/canarying-releases/)。
5. **持续学习要显式量化遗忘。** GEM 论文除最终准确率外还提出前向/后向迁移度量。对本项目更实用的简化是记录每次更新前后在历史漂移阶段/保留设备组上的分数变化，报告平均增益、最差退化和恢复所需批次，而非只看最新批。来源：[Gradient Episodic Memory, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)。
6. **无监督协变量对齐可作为轻量候选，需保守验证。** CORAL 通过对齐源域与目标域二阶统计量实现无标签域适应。它为 A 类传感器批次提供了比深模型对抗训练更轻的候选方向，但不解决标签/概念漂移；必须检验传感器异常值、相关变化、类条件变化和当前比赛对未标注目标批自适应的许可。来源：[Sun、Feng、Saenko，AAAI 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10306)。
7. **气体传感器公开基准本身有边界。** UCI Gas Sensor Array Drift 常用于跨时间批次漂移研究；已有研究专门指出其气体类别与漂移基准解释存在局限。因而外部 UCI 十批数据适合做压力测试和方法排错，不应直接替代官方数据协议或充当正式泛化证明。来源：[Drift in a Popular Metal Oxide Sensor Dataset Reveals Limitations for Gas Classification Benchmarks](https://arxiv.org/abs/2108.08793)。

## 建议的 RSI 技术路线

### 1. 先形成可信的流式评测基准

按官方批次顺序做可复现 replay，定义 `predict → commit decision → label reveal → optional train/update → next batch` 状态机。记录 `batch_id`、输入摘要、当前 champion SHA、策略/skill 版本、预测、决策、标签解封时间和恢复状态。对重复批、进程重启、checkpoint 恢复和坏输入测试幂等。`report/probe` 永不进入 Agent 工具/记忆视图。

把监测拆成两个阶段：

- **预测前、标签未知：** 输入稳健统计、输出分布/置信度和已校准的协变量漂移信号；只描述证据和不确定性，不伪装成“已发现性能下降”或“概念漂移”。
- **标签揭示后：** 按原预测版本更新 prequential 分类/回归损失、批次/组别指标和单变量在线性能告警；再决定是否允许从已揭示数据训练。标签揭示前禁止其进入模型、校准、策略、提示和记忆。

用固定的 no-drift、孤立异常点、缓慢漂移、突变、循环漂移 replay 校准每个报警规则。至少报告每千样本误报数、检出率、检测延迟、漂移阶段主指标、最差批性能、CPU/内存/推理时延。

### 2. 分任务提升工业模型表现

| 任务 | 首批候选 | 切分与主风险 | 决策指标 |
|---|---|---|---|
| A · 气体分类 | parent/retain；`recenter` 的 mean/std 与异常值稳健变体；之后再评估轻量二阶统计对齐 | 按真实批次顺序留出批次；异常点、类别先验变化、相关结构和概念变化分开测试 | overall 与逐批 macro-F1、最差批、弱类别召回；不以单一极端特征值触发更新 |
| B · 轴承 RUL | parent/retain、`damage_shrink`、受限 shrink；若训练则比较 train-only early-stop/短程续训 | 按 bearing 整组留出，不把同一设备相邻行当独立样本；当前 checkpoint 缺少设备身份等上下文时，不凭空假设可恢复该信息 | overall/逐 bearing RMSE、最差 bearing、常数控制；比常数更差时选择保守 fallback |
| C · 发动机 RUL | parent/retain 对照受限 staged continuation；监测漂移但不强迫模型更新 | 按 engine unit 留出，使用后续阶段顺序回放；防止 train/test 单元混淆和遗忘 | 平均和最差阶段 RMSE、前序 unit 保持、每批恢复时间；增益未越门槛就 retain |

先做低成本策略和推理校准，之后才跑续训或统计对齐。每个候选只改一个因素，候选数每轮不超过 3。

### 3. 让策略层自身成为可控实验对象

把 skill/policy 定义为带版本号的候选工件：固定父版本、来源实验、适用任务/漂移类型、预期行为、拒绝条件和内容 hash。每次 Agent 改进先写到临时候选目录，与冻结的当前版在同一组流回放中比较；评估器独立复算，技能审核器检查安全和数据边界。只有跨 seed、跨批次的收益通过门控后，才晋升到下一次运行的策略版本；否则 archive/rollback。

至少做三次真实“历史 → 假设 → 实验 → 决策”迭代：包括一次检测/归因假设、一次模型/校准候选比较、一次用先前结果改进 skill/policy 的复验。三次均须保留负结果；不要求每次 promote，也不为满足次数强行上线。

### 4. 使用多维门控而非单分数晋升

候选要同时满足：

- 预先固定的 selection 主指标收益超过基于种子/设备/批次波动校准的最低有效门槛；
- 最差漂移批、关键类别或设备组的退化不超过安全界限；
- 相对前序阶段无不可接受遗忘，且可以从父 checkpoint 确定性恢复；
- token、CPU 秒、wall time、峰值 RAM、checkpoint 大小均在预算内。

最终 report 仅用于实验后独立审计，不参与候选挑选。若只有一次微小提升、置信区间跨零、漂移归因不确定或资源超限，决策为 retain。

## 建议的 8–9 小时执行顺序

| 时段 | 工作 | 退出条件 |
|---|---|---|
| 0:00–0:45 | 找到比赛 `workspace.v2` 正式 schema/SDK 与输入输出样例，确认标签揭示和状态目录；核对 PyTRiO 模型、SwanLab 记录约束 | 契约字段及来源逐项写下；未知时不消耗线上试运行 |
| 0:45–2:00 | 修流式预算 tripwire；建立按批 replay/state machine 和标签隔离测试 | 重放/重复批/断点续跑等价；标签不可提前读取；API 单轮超过预算可中止 |
| 2:00–3:00 | 固化无漂移与已知漂移压力场景，校准检测阈值 | 误报率、检出延迟和资源测量可重复；不以随机 selection 抽窗替代 |
| 3:00–5:15 | A/B/C 低成本策略首轮，固定候选数与 seed；记录每轮假设和 retain/promote/rollback | 三任务逐批结果、最差阶段、配对差值与资源报告齐全 |
| 5:15–6:45 | 做策略/skill 消融：旧版、候选版、无 memory/skill 对照；候选版仍在隔离副本 | 能归因性能来自模型更新、策略提示还是记忆；无提升就不晋升 |
| 6:45–7:45 | 对最多 1–2 个优胜候选做新 seed/新批次复验；核对 token、耗时、峰值 RAM 和回滚 | 优胜效果不靠单个批次；最差组无明显退化 |
| 7:45–8:30 | 写完整实验记录、生成可复现 manifest；确认 SwanLab 公开 URL/权限和无敏感内容 | 形成至少三次完整迭代的记录；平台合规未核实则标注 blocker |
| 8:30–9:00 | Linux/amd64 与 4 核/8GB 预检，启动只绑定 Tailscale 私网的只读进度面板 | 容器协议检查、镜像大小、health check 通过；面板不显示密钥/标签 |

**成本闸门：** 先跑确定性本地 replay/策略实验；只有逐响应预算硬停止和兼容 preflight 通过后，才发起有限的开发期 LLM 调用。Tokenplus 仅作开发兼容性验证，正式运行切换到赛事批准的 PyTRiO 服务。线上试运行必须等协议、镜像和资源 preflight 都通过，并按平台当前额度决定。

## 参考文献与官方技术资料

- A. Bifet, R. Gavaldà, “Learning from time-changing data with adaptive windowing,” SDM 2007. [DOI](https://doi.org/10.1137/1.9781611972771.42)
- River, [`drift.ADWIN` API](https://riverml.xyz/dev/api/drift/ADWIN/)
- A. Grzenda et al., “Delayed labelling evaluation for data streams.” [DOI](https://doi.org/10.1007/s10618-019-00654-y)
- F. Hinder et al., “One or two things we know about concept drift—a survey on monitoring in evolving environments. Part A: detecting concept drift.” [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11220237/)
- Google SRE Workbook, [Canarying Releases](https://sre.google/workbook/canarying-releases/)
- D. Lopez-Paz, M. Ranzato, “Gradient Episodic Memory for Continual Learning,” NeurIPS 2017. [Paper](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)
- B. Sun, J. Feng, K. Saenko, “Return of Frustratingly Easy Domain Adaptation,” AAAI 2016. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10306)
- A. Vergara et al., “Drift in a Popular Metal Oxide Sensor Dataset Reveals Limitations for Gas Classification Benchmarks.” [arXiv](https://arxiv.org/abs/2108.08793)
