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
