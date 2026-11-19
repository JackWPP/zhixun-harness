# 38 · 预算扩容与 SDK 1.0.4 兼容（0.12.1 / 0.13 开发线）

> 状态：P1 预算与 thinking 开关已落地；P2 实验矩阵在跑（见 §4）。本文件随实验完成更新。

## 1. 为什么改

- 0.12.0 之前：本地模拟的 kernel 预算按平台默认（6 turns、256 output tokens）跑，thinking 关闭；
  一次 run 的信息量小，无法验证“开更大预算/开 thinking 是否真的更好”。
- 平台 2026-11-19 将 SDK 升到 **1.0.4**：候选/状态改为分块上传、沙箱 2C4G、
  公开验证集只能测量/选择。镜像必须同步，否则 `describe` 版本不符或上传协议不兼容。
- 用户要求“加大预算 + 做兼容 + 重跑中断实验”：本地扩容**只影响模拟与开发**，
  平台路径仍按宿主上限（8M tokens、单轮 max_output_tokens=256、2C4G、6 turns）执行。

## 2. 预算参数（本地模拟可调，平台默认安全）

| 变量 | 本地（P2 矩阵） | 平台默认 | 作用 |
|---|---|---|---|
| `RSI_V2_MAX_TURNS` | 8（pipeline）/10（kernel） | 6 | kernel 每轮工具调用上限 |
| `RSI_V2_TOKEN_CAP` | 0（不限，单 run 护栏 300K） | 0（宿主 8M 总预算） | kernel token 预算 |
| `RSI_V2_MAX_OUTPUT` | 256（pipeline）/1536（thinking kernel） | ≤256 | 单次网关 max_tokens |
| `RSI_V2_THINKING` | **auto = off**（E6 实测无收益） | off（`enable_thinking=False`） | 是否让 provider 产 reasoning |
| `RSI_V2_MAX_EXPERIMENTS_ROUND` | 6 | 3 | 每轮实验配额（Guard） |
| `RSI_V2_MAX_EXPERIMENTS_TASK` | 40 | 40 | 每任务实验总配额 |
| `RSI_V2_CAPACITY_ARM` | off/h64/ls10/both | off | 容量候选臂（官方 v1 格式） |
| `RSI_V2_ADAPT_ARM` | auto/pair/retrain/retrain_pair | auto | 累积标签适配臂 |
| `RSI_V2_SEED` | 1/2/3 | 20261117 默认 | 行序与候选模型初始化 |

`scripts/sim/full_env.py` 新增 `--thinking {auto,on,off}`、`--max-turns`、`--token-cap`、
`--max-output`、`--cap-train`，并透传到 participant 子进程；`scripts/sim/campaign.py`
的 run 描述也支持这些字段，矩阵见 `scripts/sim/campaigns-p2.json`（33 runs）。

## 3. thinking：为什么按路径开关

| provider 方言 | 关推理字段 | 开推理字段 |
|---|---|---|
| tokenplus / siliconflow | `thinking:{"type":"disabled"}` | `thinking:{"type":"enabled"}` |
| PyTRiO | `enable_thinking:false` | `enable_thinking:true` |

实测（tokenplus `qwen/qwen-3.8-27b`）：开 thinking 后简单动作 reasoning ≈ 145 字符/36 tokens，
content 正常，延迟 1.19s vs 关闭 0.62s；**但平台 `max_output_tokens=256` 时 reasoning 会吃掉
大半预算、导致 content 为空**（历史踩坑）。因此：

- 本地 kernel：thinking **on**、`max_output=1536`，并带空 content 自动放大重试；
- 本地 pipeline / 平台所有路径：thinking **off**、`max_output≤256`；
- `--thinking auto` 默认 **off**；E6 A/B（kernel、2 rounds × 2 seeds）中 on/off 的 hidden 分数逐位相同（.5682），
  但 on 的 wall 为 568s/492s vs off 的 55s/55s（约 9–10×），因此默认关闭；`--thinking on` 仅保留作调试。

## 4. SDK 1.0.4 兼容清单

- `vendor/contest_harness-1.0.4/`（27,979 B）；镜像 `COPY`、`PYTHONPATH`、`describe` 全部指向 1.0.4；
  `describe` 输出 `sdk_version:"1.0.4"`（`scripts/verify_image.sh` 已验证）。
- 分块上传：participant 侧由 SDK 自动做 begin/chunk/finish；宿主侧本地实现了
  `scripts/sim/upload_host.py::UploadStore`，`resource_limits` 声明 `chunked-v1`，
  单次 512MiB / 每轮 1GiB / 1MiB chunk。
- 新错误码 `RPC_OUTCOME_UNKNOWN`：不自动重试，直接失败本 run（本地模拟保留原始 stderr）。
- 沙箱 2C4G：本地矩阵 `--cap-train 24000`，C 的 continue_train epochs 由 candidate 指定；
  OOM 会被 kill，日志里保留 child stderr。
- 公开验证集：`scripts/sim/full_env.py::stage_data` 的 official-seq 分支只把
  `b1 + 已揭示 drops` 放进训练，validation 固定 b2；回归测试
  `tests/test_sim_protocol.py::test_official_seq_stage_data_keeps_validation_out_of_training`。

## 5. P2 矩阵（33 runs）

| 组 | 场景 | 臂 × seeds × rounds | runs |
|---|---|---|---|
| E1 容量 | official-seq A | off/h64/ls10 × 3 × 8 | 9 |
| E2 自适应 | official-seq A | off / retrain / pair / retrain_pair × 2 × 8 | 8 |
| E3 六臂 | official-seq A | baseline/retrain/retrain_pair × 2 × 8 | 6 |
| E4 B/C | official | B、C 各 × 2 × 10 | 4 |
| E5 泛化 | external / schema | 各 2 seeds × 4 / 3 | 4 |
| E6 thinking | official-seq A（kernel） | on/off × 2 × 2 | 4 |

运行：`.venv/bin/python scripts/sim/campaign.py --matrix scripts/sim/campaigns-p2.json`，
manifest `.scratch/sim/campaign-manifest.jsonl`，日志 `.scratch/sim/logs/campaign-p2.log`，
SwanLab online。结果与去留结论见 `docs/37_CAPACITY_ABLATION.md` 与 §6（实验完成后补）。

## 6. 结果与去留（P2 完成：33+10 runs，0 failures）

- **E1 容量臂**（official-seq A，8 轮）：cap-off .4906（6 seeds）/ h64 .4483（3 seeds）/
  **ls10 .5828**（6 seeds）；ls10−off = **+.0923** CI [+.0110,+.1806]（6/6 非负）。
  → 采用 ls10 为 opt-in 默认（`RSI_V2_CAPACITY_ARM=ls10`，仅分类、官方 v1 checkpoint）；
  h64 负收益，移除默认。
- **E2/E3 自适应**（official-seq A，8 轮）：policy off .3806（4 seeds，确定性）；
  adapt-retrain **.5214**（4 seeds）；配对差 **+.1408** CI [+.0557,+.2259]；
  pair-only .3936（2 seeds）远弱 → 默认优先 retrain，pair 可选叠加。
- **E4 B/C**（official，10 轮，pipeline）：B RMSE **283.52**（2/2 seeds）；
  C RMSE **38.71 / 39.32**；与历史 best 持平或更好，回归路径稳定。
- **E5 泛化**：external hidden **.5226 ± .0011**（2 seeds）；schema hidden **.1663**（2/2）——
  schema 变体（字段改名）仍无法从统计量上恢复，记为已知限制，不做过度声称。
- **E6 thinking**：on/off hidden 均为 .5682（2/2 seeds 相同），wall 568s/492s vs 55s/55s
  → **默认 off**，仅保留开关。
- **默认开关**：`RSI_V2_CONTROLLER=auto`（现有）、`RSI_V2_DRIFT_POLICY=1`（现有）、
  `RSI_V2_CAPACITY_ARM=ls10`（新）、`RSI_V2_THINKING` 不设（=off）、平台路径不传 thinking。
- 版本 0.13.0：镜像 sha256 `8efa4c0c…4059`（380,354,560 B），verify PASS；
  平台合规 trial 已启动：submission `4385210dd3c33f436b6a43a4d4930978`、
  run `f83623338e79962dcb808e0e4c6efc63`（image archive `818d63e0…7ce6`）。

## 7. 边界

- 本地 thinking/预算只影响模拟，**平台 8M/256/2C4G 不变**；平台路径不传 thinking 字段。
- hidden proxy 仍非平台隐藏流；结论只声称“同协议下配对比较”。
- 旧代理实验（3 类 644 行、+0.147、2.8×）已撤回，不进入新结论。
