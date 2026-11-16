# 22 · RSI 闭环实现与预检进度

日期：2026-11-16
分支：`codex/rsi-closed-loop`
基线提交：`eba98f2`

## 本轮实现

- PI Runtime 在每次发起模型请求前按序列化上下文预留 token，并按配置限制 completion 长度；返回 usage 后结算预留，缺失 usage 时按保守值计费。并发子 Agent 共用同一原子预算。
- 新增 `rsi.streaming.BatchSession`，以原子状态文件保存批次输入、预测摘要、champion、决定和标签哈希；只有提交决定后才可揭示标签。相同批次重放幂等，模型/输入冲突、顺序错误、坏负载会被拒绝；标签正文放在权限为 0700/0600 的独立目录。它实现时序不变量，但**不是** `contest.harness.workspace.v2` 官方适配器。
- Agent 对技能的 create/patch/delete 操作先写入隔离候选副本。候选带父版、候选内容哈希和状态；晋升必须附带哈希匹配的独立评估报告、至少两个 seed 的逐 seed 指标和最差组退化门槛，晋升后可恢复原技能库。
- runner 漂移摘要按公开 selection 行的既定顺序取窗口，只传特征给 `analyse_covariates()`；报告不再包含目标标签分布或性能损失。标签揭示后的 prequential 性能监测仍需由正式批次协议接入。
- SwanLab 模式可经 `SWANLAB_MODE` 设为 online 并记录决策、假设、候选父子 checkpoint/code SHA、selection/report/probe 指标及 CPU/wall 字段。在线日志改为数值字段走图表、完整结构走文本事件，避免字符串/None/嵌套列表被 SDK 丢弃。

## 验证和快速实验

- `./.venv/bin/python -m pytest -q`：**109 项通过**（最终计数以本轮最后一次全量运行结果为准）。
- `scripts/verify_assets.py`：官方 A/B/C 数据、checkpoint、技能、预测入口和状态库自检通过；检查输出在本地 `.scratch/assets-rsi-20261116.json`。
- 完整确定性回放命令：

  ```bash
  ./.venv/bin/python -m rsi.cli run --tasks A,B,C --rounds 2 --llm off \
    --controller pipeline --seed 42 --experiments-cap 3 \
    --state-dir .scratch/rsi-closed-loop/seed42
  ```

- run `rsi-20261116-144857` 完成三任务 2 轮、9 个候选，CPU 149.54 秒、wall 170.64 秒、0 LLM token。完整原始产物位于 `.scratch/rsi-closed-loop/seed42/rsi-20261116-144857/`，包括 summary、轮次、候选 checkpoint、worker 日志。

| 任务 | baseline selection | 最终 selection | baseline report | 最终 report | 决策摘要 |
|---|---:|---:|---:|---:|---|
| A · macro-F1 | 0.703220 | 0.734555 | 0.686380 | 0.683406 | 两轮 recenter 晋升；report 比公开基线低 0.002973，存在泛化退化 |
| B · RMSE | 678.021915 | 409.736985 | 664.009143 | 424.130024 | damage-shrink 晋升，第二轮 retain；selection 与 report 均改善 |
| C · RMSE | 44.748746 | 40.523699 | 57.460001 | 56.021241 | staged continuation 晋升，下一轮短续训被门控为 retain |

baseline 指标来自现存基线复算记录；report 是独立审计集，不参与晋升。该实验使用 pipeline 确定性规划器，不调用 LLM，不写入技能，也不属于自主 RSI 性能证据。selection 的改善不代表隐藏批次收益。峰值 RAM、批次检测延迟、逐批误报率尚未采集。

为控制进度，本轮 seed 43 在用户要求停止跨 seed 复验后中断。A/B 已完成，C 正在 staged continuation 时中断；该 run 为不完整记录，不纳入效果汇总。A/B 结果可见对应 `.scratch/rsi-closed-loop/seed43/`。

## 在线实验记录

已完成完整 A/B/C 两轮在线 SwanLab 回放（仍是本地公开数据、`controller=pipeline`、0 LLM token，不是比赛封闭流）：

- 项目：[zhixun-rsi](https://swanlab.cn/@wppjkwbuct/zhixun-rsi)
- Run：`large-tiger-3`，ID `1lmjutiy`，[打开 run](https://swanlab.cn/@wppjkwbuct/zhixun-rsi/runs/1lmjutiy)
- 运行目录：`.scratch/rsi-online/full-seed42b/rsi-20261116-152408/`
- 2 轮 × 3 任务、9 个候选；CPU 154.59 秒、wall 185.17 秒；上传完成 6 个文件、17.8 KB。
- 在线记录包含轮级数值图表和 `rsi_event` 文本事件；没有写入 API key、原始数据或标签正文。

## 尚未通过的提交门槛

1. 官方 `contest.harness.workspace.v2` 契约和 SDK 尚未找到；状态机暂未映射 `start/candidate/save_state/complete` 方法。
2. runner 仍从公开包构建离线切分，不是平台隐藏批次流；不能据此声称已实现完整的隐藏流自主闭环。
3. PyTRiO 的真实 endpoint、模型 ID、tool calling、usage 解析和断网行为未验证。Tokenplus 不用于本次运行。
4. SwanLab 在线记录已验证上传和链接可访问；比赛要求的公开权限、最终记录字段和封闭评测出站规则仍待平台确认。
5. Linux/amd64 镜像加载及 4 核/8 GiB 限额彩排已完成；端口 7861 只读面板已绑定 Tailscale 地址并通过健康检查。
6. 平台提交校验尚未运行；不消耗试运行额度前仍需在账户页面复核协议、配额和提交状态。

## Linux/amd64 彩排

- 远端镜像：`zhixun-rsi-harness:rsi-5099370`
- 镜像摘要：`sha256:8a01feb3d8a958cd76c94aa56cd158bf330eb18ae98c951a276d4dd6ebdb48c3`
- 架构与大小：`linux/amd64`，约 420 MB
- 运行约束：`--cpus=4 --memory=8g`；资产自检通过，观测内存约 546 MiB
- 运行：`rsi-20261116-075350`，A/B/C 各 1 轮、6 个候选、CPU 350.6 秒、wall 388.42 秒
- 容器内最终 selection：A `0.720696`、B RMSE `409.736985`、C RMSE `40.523699`
- 看板数据源：容器产物 `/home/yanp/rsi-docker-preflight/rsi-runs/rsi-20261116-075350/state.json`
- 看板健康检查：`http://100.114.169.50:7861/api/health` 返回 `state=ready`、`run_status=done`

下一步仍受官方 workspace 契约、PyTRiO endpoint 和平台提交校验约束。获得官方 schema 后做最小适配；只有协议、资源、云端记录和镜像预检全绿后才进行一次试运行。
