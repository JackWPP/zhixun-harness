# 16 · 关键决策记录（ADR）

按时间顺序记录影响系统形态的决策、备选方案与否决理由。答辩时每一页"为什么这样
做"都能在这里找到依据。

## ADR-001 保留官方 baselines，新系统放 `rsi/`

- **背景**：官方包要求续训兼容（结构 / 归一化 / checkpoint 格式），且平台按
  `contest.cpu-mlp.v1` 校验。
- **决策**：`baselines/` 原样保留（仅必要的 bug 修复），复制 `model.py`/`data.py`
  到 `rsi/` 并扩展；`checkpoint.npz` 读入即校验 `task/classes/width/paths`。
- **否决**：直接改官方 `baselines/model.py` 结构（会导致父 checkpoint 不兼容，
  违反续训协议）。
- **后果**：任何候选都能被平台加载器接受；旧原型 `harness/`、`audit/` 仅作审计留档。

## ADR-002 验证集切三层 + 外部探针只验证

- **背景**：旧原型用验证集既选型又报告，指标乐观；B/C 的验证集是"同分布随机切分"，
  隐藏测试是漂移流，直接拿验证集调参会误导门控。
- **决策**：每次 run 先确定性切 `calibration / selection / report` 三层（互斥，
  manifest 记 sha）；A 额外构建 UCI batch3–10 探针，**只进报告不进训练**。
- **否决**：只用整体验证集选型（无法估计漂移下的真实增益）；把探针混进训练
  （会违反"扩展数据只作验证"的自我约束，也失去漂移证据价值）。
- **后果**：本地 selection 指标与官方 val 数值不同（A 0.703 vs 0.714），但决策
  逻辑与隐藏评测一致；报告中明确披露。

## ADR-003 Agent 运行时选 PI（pi-agent-core-lc）

- **背景**：用户担心自研 loop 的边界条件；需要 Hook + Skill + 工具注入 + token
  控制 + 可离线嵌入。
- **候选实测**：pi-agent-core-lc ✅（4.09s 工具闭环、usage/thinking 正常、44MB）；
  openai-agents ✅（5.07s、59MB，留作备选）；pydantic-ai ❌（Qwen thinking 模式
  HTTP 400 `tool_choice` 冲突）；DSH 可行但 928MB + Node，不适合镜像。
- **决策**：PI 作为唯一主后端，封装在 `rsi/agent/runtime.py`；不自研 loop。
- **后果**：`rsi/agent/` 约 1,400 行覆盖角色/工具/降级；依赖固定版本进镜像。

## ADR-004 LLM 只做"选择与审查"，执行器二次校验

- **背景**：开放编码 Agent 会越界、改结构、产生不可复现实验；赛题要求"受控"。
- **决策**：Agent 只有 6 个只读/提交工具；候选必须命中 `strategies.py` 白名单且
  参数落在区间；`validate_proposal` 在 executor 前二次校验；Arbiter 只能更保守。
- **否决**：给 Agent shell/文件/网络工具（无法审计、违反安全边界）。
- **后果**：即便模型输出恶意参数，也只能被拒绝或 clamp；所有实验可枚举。

## ADR-005 推理期校准作为一等策略（recenter / shrink）

- **背景**：A 的漂移主要是协变量偏移（传感器增益），B 的基线输出严重欠拟合
  （比常数还差 27%）；两者都不该靠"改权重"硬扛，且 checkpoint 字节不动最安全。
- **决策**：把 recenter（按目标批均值/方差映射回冻结归一化分布，可 weight 混合）
  与 shrink（向校准中心收缩）纳入策略库；候选可只改 postprocess 不改 checkpoint。
- **证据**：A 官方 val 0.714→0.725（recenter w=0.4）且 UCI batch3–10 probe
  0.4054→0.5239；B 678.02→462.45（shrink α=0.0，即校准中心）；C 44.75→41.82。
- **否决**：改归一化/结构重训（破坏续训协议，且本地数据不足以稳定重训）。
- **后果**：A 的收益是"零 checkpoint 字节"的；B 退化为常数预测，需在报告中如实
  说明"这是基线失效下的安全最优解"。

## ADR-006 确定性降级阶梯

- **背景**：评测环境可能无外网/无 LLM，规则要求"无人干预完成全部阶段"。
- **决策**：`--llm off` 与 LLM 故障都走 planner 的 SEEDED 策略序列；单角色失败
  各自确定性回退；最终仍写满 10 轮与合规报告。
- **否决**：LLM 不可用则中止（直接违反门槛 G1）。
- **后果**：两次全量演练均为 `--llm off` 完成，相当于对降级路径做了端到端验证。

## ADR-007 无 Docker daemon 手工构造 `docker save` tar

- **背景**：开发机 bwrap 沙箱内无 rootful Docker；评测只要求单文件未压缩 tar。
- **决策**：`scripts/stage_image.sh` 组装 overlay，`scripts/build_image_tar.py`
  按经典 docker-save 布局（config/manifest/layers）生成 tar；用 `crane` 拉
  `python:3.12-slim` rootfs；用 userns 内 dockerd `docker load` 实测验证。
- **否决**：只交 Dockerfile（平台明确拒绝源码压缩包）；docker export 容器包
  （不符合 save 布局）。
- **后果**：产物 374.9MB、load 后 366MB；镜像内 `selftest` 与 `run` 已跑通。

## ADR-008 扩展数据只做验证，不参与训练

- **背景**：赛题有 `extended_data_validation` 加分，但混数据会污染"官方数据"口径。
- **决策**：UCI 全量只构 A 的 probe；B/C 外部数据只列调研结论，不入库。
- **后果**：加分证据清晰（同源 batch1/2 = 官方，batch3–10 = 漂移验证）；训练
  数据 100% 来自官方包，规避"使用未授权数据"风险。

## ADR-009 门控阈值 = 2× 种子噪声（A 0.004 / B 5.0 / C 0.3）

- **背景**：小数据集的指标抖动会误触发 promote。
- **决策**：`margin` 取种子间标准差的 2 倍（早期探针：A 0.006、B 9.4、C 0.7；
  保守取 A 0.004、B 5.0、C 0.3），提升 ≥ margin 才 promote，劣化 ≥ 2×margin 才
  rollback。
- **后果**：A 早期连续 retain 属预期；B 的 215 点提升一次即锁定。

## ADR-010 PI 使用进程级持久事件循环

- **问题**：每次 `asyncio.run()` 新建并关闭循环，httpx/langchain 的异步客户端
  绑定旧循环，第二次调用报 `Event loop is closed`。
- **决策**：`PIRuntime._event_loop()` 懒创建全局循环，`run_until_complete` 复用，
  `close()` 在 run 结束调用。
- **后果**：LLM 全链路（诊断/实验/评审/裁决/报告）可连续调用。

## ADR-011 executor 子进程必须继承 PYTHONPATH

- **问题**：镜像内 worker 报 `ModuleNotFoundError: numpy`——executor 把 PYTHONPATH
  覆盖成 repo 根，丢掉了镜像 site-packages。
- **决策**：`python_path = harness_root + os.pathsep + inherited`。
- **后果**：镜像内 2 轮 run 通过；重建镜像后作为 0.3.0 交付。

## ADR-012 SwanLab 默认离线，token 走环境变量

- **背景**：赛题要求云端实验记录，但评测环境可能无外网；凭据不能进镜像。
- **决策**：`--swanlab` 才启用（`swanlab.init(mode="offline")`），产物可 `sync`；
  镜像内不含 token。
- **后果**：本地/离线两不误；正式提交时由运维挂 `SWANLAB_API_KEY` 同步。

## 被否决的备选（速查）

| 备选 | 否决理由 |
|---|---|
| 自研 agent loop | 边界条件多（事件循环、usage、流式、重试），用户明确担心 |
| DSH 作为运行时 | 928MB + Node，评测镜像不经济；仅开发期可用 |
| pydantic-ai 作主选 | Qwen thinking 模式与强制 tool_choice 冲突（HTTP 400） |
| 从零训练新模型 | 平台禁止；且本地无标签预算 |
| 改网络结构 | 破坏续训兼容；官方加载器直接拒绝 |
| KNN / 线性回归替代模型 | B 上实测 653–662 / 471–514，均不如 shrink；且属结构变更 |
| 种子集成 | A 有小幅收益，但实验/时间成本翻倍，优先保交付 |
| nohup 后台跑长任务 | DSH 工具会清理进程组；改用工具自带后台任务 |
| 在镜像内装 Docker | 评测环境不一定有特权；手工 tar 已满足要求 |

## ADR-013 Agent Kernel 取代固定六步循环

- **背景**：v0.3 的 pipeline 是确定性状态机，自主性只停留在"LLM 选策略"，
  Agent 看不到实验结果、不能决定下一步，无法称之为 RSI harness。
- **决策**：新增 kernel 控制器。Agent 持有目标 + 18 个受限工具，在 PI loop 内
  自主观察/假设/实验/评估/决策/写记忆；Guard 在工具边界做确定性校验；
  supervisor 只负责预算、切分、记录、promote 应用与兜底。
- **后果**：A 任务 3 轮实测 18/14/14 次工具调用、决策非模板；LLM 不可用或
  连续失败自动降级 pipeline（同一套记录与门控）。kernel 单轮 2–4 分钟，
  3×10 全 kernel 需数小时，故保留时间感知降级。

## ADR-014 记忆分三层：冻结快照 / 有界文件 / SQLite 经验库

- **背景**：需要跨轮、跨 run 的经验复用，且不能破坏 prefix cache、不能无限膨胀。
- **决策**：参考 Hermes：①session 开始时冻结快照（MEMORY 4000 + TASK 2000 chars）
  注入提示词；②写入立即落盘但当前 prompt 不可见；③结构化经验（rounds/experiments/
  lessons/strategy_stats）落 SQLite(WAL/FTS5)，JSONL 仍是审计真源，可重建；
  ④每任务一次 Review 提炼 lessons，Curator 归档长期未用技能。
- **后果**：跨 run 实测首轮工具调用 18→13、直接复用已验证配置；记忆可 `--no-memory`
  消融；注入扫描 + 漂移 `.bak` 拒写 + 容量上限防止污染/膨胀。

## ADR-015 Guard 放在工具边界，而不是提示词里

- **背景**：仅靠 system prompt 约束 LLM 不可靠，且答辩需要可审计证据。
- **决策**：每个动作在工具执行前由 Guard 校验：策略白名单与参数范围、训练数据
  白名单、每轮/每任务实验上限、decide 必须用 selection+margin 重算门控、
  promote 必须有成功候选、记忆/技能文本注入扫描、所有动作写 trace。
- **后果**：实测拒绝第三方训练数据、第 4 个实验、无依据 promote；Agent 仍保有
  "在合法空间内自由行动"的自主性；Guard 拒绝会作为观察反馈给 Agent。

## ADR-016 数据适配只做验证，绝不做训练

- **背景**：赛题只允许平台数据 + 官方 parent checkpoint 续训；混入第三方数据
  有合规风险，且隐藏测试不可见。
- **决策**：私有 JSONL 只推理；UCI 全量只做外部验证；合成漂移只做回归测试；
  显式 `used_for_training=false` 与 `training_isolation` 报告。
- **后果**：扩展数据验证加分项可复现；训练路径与验证路径物理隔离，答辩可自证。

### ADR-013 外部预训练只做 opt-in，且与验证/隐藏严格隔离

- **背景**：A 赛道官方 train 仅 batch1（445 行），真实漂移批次上泛化极差；
  UCI 数据集其余 batch 公开可得，规则未禁止，且外部数据验证本身是加分项。
- **决策**：只允许通过 `extended_pretrain` 在“外部预训练 + 官方 train 微调”
  中使用；使用 `RSI_ALLOW_EXTENDED_DATA` 显式开启；验证/隐藏/校准永不拟合。
- **后果**：A 选择集 0.7346→0.8645，诚实留出 b8–10 0.352→0.593；代价是依赖
  公开外部数据，必须在报告与 workspace_report 中披露，关闭开关即可回到纯官方数据。

### ADR-014 用 task.json 契约封装“可复用 RSI 运行时”

- **背景**：用户要求把 RSI 流程封装好，能在官方平台/新数据集上复用，而不是
  写死 A/B/C。
- **决策**：任务 = `task.json`（data_root/package/split_strategy/probe/extended/
  metrics/direction/baseline/margin），splits 策略通用化（stratified|group|units），
  planner 对未知任务提供 generic 策略库，`--task-spec`/`RSI_TASK_SPECS` 注册。
- **后果**：`scripts/make_demo_task.py` + `tests/test_portable.py` 证明第 4 个任务
  可无代码改动跑通；平台隐藏流只需换包与 task.json。

## ADR-017 B 赛道用“损伤感知后处理”而不是结构迁移

- **背景**：B 的父基线 val RMSE 681，比常数 536 还差；检查发现行特征与 RUL 的
  关系在留出轴承上不可迁移。spike-b 证明结构迁移（34→136 维因果特征）虽可做到
  零权重恒等（max|Δ|=1.1e-13），但最优也只到 469.8，代价是改 checkpoint 结构、
  predict 特征管线与边界检测。
- **决策**：保持父 checkpoint 不变，新增推理期 `damage_shrink` 后处理：
  `y=(1-w)*(a*raw+(1-a)*center)+w*curve(score)`；`center` 用官方 train 目标均值，
  `curve` 由官方 train 标签拟合（10 个分位箱），`score` 只用当前轴承截至当前行的
  因果统计量；`a=0.5,w=0.6` 在 selection 上选择，report 仅报告。
- **后果**：B selection 491.6→409.7、report 489.6→424.1，父 checkpoint 与
  checkpoint.npz 契约完全不变；后处理内嵌“段边界重置 + 可疑流回退纯 shrink”，
  并在 `postprocess.json` 中自带 detector/curve，predict 端无需额外训练数据。
- **权衡**：依赖“平台按轴承分块、按时间顺序喂入”的假设；已在 round 记录与文档
  中披露，乱序流会触发保守回退。

## ADR-018 C 赛道用两阶段长续训，而不是改结构

- **背景**：C 的宽度试验（h32→h128）无收益；短续训（100 epoch）只到 41.8，
  长训练直接用 lr=1e-3 会梯度爆炸（nonfinite gradient）。
- **决策**：新增 `continue_train_staged`：先长阶段（官方 train、lr=1e-3、
  1000 epoch），再低 lr 抛光（lr=2e-4、400 epoch）；全部沿用官方训练器的
  Adam/梯度裁剪/冻结归一化，结构不变、父谱系连续，属于规则允许的“修改训练参数
  与代码后继续训练”。
- **后果**：C selection 41.82→40.52、report 56.28→56.02；单次实验约 200s，
  仍在每任务预算内；作为候选策略由 agent 在 Guard 内自由选择，不做强制。

## ADR-019：批次流协议在既有 Runner 之上实现（而不是另写一条训练路径）

* 决策：`rsi/stream/` 每批 materialise 一个小 package 并调用未改动的 `Runner(rounds=1)`，
  而不是复制一套轻量训练循环。
* 理由：保证 Guard/记忆/记录/门控与 10 轮静态评测完全同一条代码路径；平台契约未知时仍能
  复用已验收的产物（decision/predictions/summary），避免双实现漂移。
* 代价：每批有少量初始化开销（秒级），可接受。

## ADR-020：离线知识库进入 Agent 决策循环，但只作为“参考数据”

* 决策：`knowledge/` 12 篇策展文档 + FTS5 检索 + `search_knowledge` 工具 + prompt 注入；
  注入文本显式标注“非指令”，Guard 扫描注入内容。
* 理由：符合赛事“skill/知识驱动 RSI”的主观分，同时避免把外部知识变成不可审计的隐式规则。
* 证据：`docs/18_AGENT_KERNEL.md §10.4`（第 1 轮假设引用知识库、检索 1 次/3 hits）。

## ADR-021：软目标 + 硬预算（mission goal 不进交付物）

* 决策：交付运行时只有 advisory `goal_soft` 与硬预算；GoalLoop 的 done 判定只依赖轮次/预算，
  soft target 未达成不影响收尾。Lead 的会话 goal 只用于组织开发工作。
* 理由：用户 m03315 明确“goal 是工作方式，不是交付目标”；激进目标若阻断收尾反而造成 0 分。
* 证据：`tests/test_goal_nonblocking.py` 4 passed；state v3 中 `goal_soft.note` 固定为
  "soft target, non-blocking"。

## ADR-022：decision.json 双写（扁平平台字段 + 嵌套调试记录）

* 决策：每批决策同时写 `decision`（字符串，平台最可能使用的形状）、`round_index`、
  parent/candidate/checkpoint sha256、`validation_metrics`，以及 `decision_detail` 嵌套记录。
* 理由：平台真实 schema 未知（docs/19 U3），双写零成本、两种消费者都能用；contractlab 实测
  两种实现 predictions 完全一致、decision 形状不一致。
