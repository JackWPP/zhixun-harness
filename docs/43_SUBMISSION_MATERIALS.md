# 43 · 提交材料与可复现性（v0.14.9）

本文件是正式评测提交的**材料与可复现性说明**，与镜像、文档、SwanLab
实验记录、GitHub 仓库一起构成完整提交材料。内容只使用可公开核验的事实。

## 1. 提交物清单

| 项目 | 值 |
|---|---|
| 镜像 | `zhixun-rsi-harness:0.14.9`，`docker save` 未压缩单 tar |
| 镜像 tar sha256 | `4d69e1e108d5150fecb245422851255384f8cccec87c8c874858c495c31e3d86` |
| 镜像大小 | 381,081,600 B |
| SDK | `contest-harness-sdk 1.0.4`（workspace v2 协议，chunked-v1 上传） |
| 仓库 | <https://github.com/JackWPP/zhixun-harness> |
| 代码版本 | tag `v0.14.9`（`git describe --tags --always` 应返回 `v0.14.9`） |
| 实验记录 | <https://swanlab.cn/@wppjkwbuct/zhixun-rsi> |
| 训练环境 | CPU-only，2 核 4 GiB 容器（平台沙箱）；本地构建为 6C/12T |

镜像内 `/app/harness.py` 是 v2 原生入口：`describe` / `train` / `predict`
三个子命令直接交给 `contest_harness.cli`，选手模块
`rsi/platform/submission.py` 只通过宿主提供的 `round.data / round.model /
round.submit_candidate / round.save_state / round.finish` 与平台交互。

## 2. 运行方式（镜像内）

```bash
python3 /app/harness.py describe   # 一行 JSON：protocol / model_access / sdk / tasks
python3 /app/harness.py train      # stdin 读 start 帧，逐轮训练并提交候选
python3 /app/harness.py predict    # stdin 逐行 {"task","inputs"}，逐行输出 JSON
```

关键默认环境变量（镜像 ENV，可在评测时覆盖）：

```text
RSI_V2_CONTROLLER=auto        # classification -> kernel, regression -> pipeline
RSI_V2_MAX_TURNS=10
RSI_V2_MAX_OUTPUT=256
RSI_V2_KERNEL_TOOLS=core      # 精简工具集，降低 token 消耗
RSI_V2_KERNEL_THINKING=off    # 平台 256 输出预算下避免 reasoning 挤占
RSI_V2_CAPACITY_ARM=ls10      # 分类任务额外的 label-smoothing 容量臂
RSI_V2_SCRATCH_ARM=on         # 回归任务从零重训候选（B h48/600ep，C h32/300ep）
RSI_V2_WARMUP_ROUNDS=2        # 前两轮保留官方父检查点，避免早期阶段回退
RSI_V2_B_RETAIN=0             # B 静态 shrink 默认关闭（隐藏流尾部会反向）
RSI_V2_B_STATIC=0
```

## 3. 数据边界与合规

- **梯度训练只使用 `/data/train` 投影**（即平台当前阶段公开的
  `public/train.jsonl` 与逐轮揭示的训练批）。代码中显式断言：
  `_assert_training_only(fit_rows, validation_rows)`，训练/验证行 id 有交集
  直接报错；候选目录写出 `data_boundary.json` 记录 `fit_id_sha256`、
  `validation_id_sha256`、`validation_used_for_fit=false`。
- **验证投影只用于测量与选择**：每轮记录 `validation_metrics`；选择器使用
  配对 block bootstrap + 预注册门限（`margin_eff=max(margin,1.5·SE)`），
  没有把验证行并入梯度。
- **推理期不重新拟合标签统计**：`pair_threshold` 只使用历史校准常量
  `-4.0`（A 类别方向的延迟标签经验），不再在提交时对标签做网格搜索。
- **续训兼容性**：所有候选 checkpoint 均为官方 `contest.cpu-mlp.v1`
  单隐层 tanh 格式（`hidden ≤ 128`、参数 < 100 000），由官方 loader 直接加载；
  `scratch_retrain` 也是同格式从零训练，不依赖自定义 loader。
- **预测鲁棒性**：分类输出 label 唯一、score∈[0,1]、最高分唯一；回归输出
  非负有限值；逐行 flush；单行异常回退到合法分布，不中断隐藏流。

## 4. 方法概要

三层结构：

1. **确定性候选生成**（每任务 3–5 个，全部只依赖公开训练数据）：
   - A：`retain` / `recenter(batch_mean/std)` / `pair_history`（固定 delta）/
     `continue_train`（累积标签 20 epoch）/ `continue_train_pair` /
     `capacity_ls10`；
   - B/C：`retain` / `scratch_retrain`（B: h48, 600 epoch；C: h32, 300 epoch）/
     `continue_train`（10–20 epoch）。
2. **可信选择**：分类走 `kernel`（PI-agent 通过宿主网关做诊断、提案、比较、
   决策，每轮真实调用 4–6 次），回归走确定性 pipeline；两条路径都经过
   配对 block bootstrap 门控，只有显著优于父模型才替换，否则保留父产物。
3. **安全网**：前两轮保留父检查点（`RSI_V2_WARMUP_ROUNDS=2`）；分类高漂移
   时允许固定 `pair_history` 覆盖；回归静态 shrink 默认关闭（本地与平台
   trial 均显示其在漂移尾部反向变差）。

## 5. 本地实验记录

协议：官方同构（start 帧、候选上传、状态续接、finish 记录），真实网关
PyTRiO `Qwen/Qwen3.8-27B`（本地核对时也用 tokenplus `qwen/qwen-3.8-27b`），
数据边界与平台一致。SwanLab 项目公开可查，`E*` 系列 run 对应本表。

| 实验 | 任务 | 设置 | 结果 |
|---|---|---|---|
| E1 容量 | A | official-seq 8 轮 × 6 seeds | off .4906±.0899；ls10 .5828±.1095；h64 .4483 |
| E2 自适应 | A | 8–10 轮 × 4 seeds | off .3806；retrain .5214±.0983；pair .3936 |
| K3c 安全网 | A | 10 轮 × 6 seeds | .6278 均值（修复前） |
| recent holdout | A | 10 轮 × 11 seeds | **.7192** 均值（sd .0121，min .7053） |
| BC multi | B/C | 10 轮 × 3 seeds | B 283.52；C 38.72（本地代理） |
| 0.14.8 trial | A/B/C | 平台试运行 | A **.894721**；B RMSE **175.81**/MAE 113.32；C RMSE **55.89**/MAE 39.71 |

正式评测（run `1b0e780323861d1714d1b01893f69b02`）总分 **45.74**，客观分
12.314/35：mean_gain 2.736/15、worst_stage 0.372/5、recovery 4.206/5、
retention 5/5、resource_efficiency 0/5；三个 gate 未过：实验记录完整性、
材料与可复现性（均为管理员核验字段）、不低于静态基线。

## 6. 复现命令

```bash
# 代码
git clone https://github.com/JackWPP/zhixun-harness && cd zhixun-harness
git checkout v0.14.9
uv venv .venv && .venv/bin/pip install -e ".[dev]"

# 测试（215 passed, 3 deselected）
.venv/bin/python -m pytest -q -m "not integration"

# 本地 v2 同构模拟（三个任务各 10 轮）
.venv/bin/python scripts/emulate_workspace_v2.py --task A --rounds 10
.venv/bin/python scripts/emulate_workspace_v2.py --task B --rounds 10
.venv/bin/python scripts/emulate_workspace_v2.py --task C --rounds 10

# 全真批次流（真实网关，读取 .scratch/llm.env）
.venv/bin/python scripts/sim/full_env.py --scenario official-seq --tasks A \
  --rounds 10 --provider real --controller kernel

# 镜像构建与自检
bash scripts/stage_image.sh
.venv/bin/python scripts/build_image_tar.py --tag zhixun-rsi-harness:0.14.9
bash scripts/verify_image.sh
```

## 7. 已知边界（诚实声明）

1. 本地 `official-seq` 的隐藏代理来自公开验证留出段，与平台隐藏流不同；
   B/C 的本地代理与平台 RMSE 不可直接换算（例如 B 本地 283.52 vs 平台
   175.81/720.07），只能用于相对比较。
2. `retention=5/5` 说明没有遗忘公开旧批；`mean_gain=2.74/15` 说明正式隐藏
   流上的提升远小于试运行口径，原因尚未能从团队接口获得 per-stage 明细。
3. `resource_efficiency=0/5` 与 A 任务 kernel 每轮真实调用、C 任务 scratch
   训练耗时相关；0.14.9 已把 kernel 工具集降为 `core`、前两轮跳过重训，
   但尚未在正式评测中验证。
4. 平台的材料/记录 gate 是管理员人工核验字段（提交接口
   `materials_complete/records_complete` 为 null、`compliance_status=unreviewed`），
   本文件与 `docs/26`、`docs/34`–`docs/42`、SwanLab 记录共同作为核验材料。
