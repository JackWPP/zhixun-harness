# C 与资源效率本地候选验证

> 这是 2026-11-26 的历史候选记录。当前实现、分组保护与验证状态见 [0.15.0.dev1 总体优化说明](52_ENTERPRISE_OPTIMIZATION.md)；下文“尚未实现”和旧配置仅描述当时状态。

日期为 2026-11-26。候选版本 `0.14.13.dev1`，基于上游 `17cc307b10caed360405d739bdfb09f457910bad`。这是独立本地副本中的源代码候选，尚未构建或上传新比赛镜像，尚无新的平台正式分数。

## 已实现的改动

1. 修复 SDK workspace-v2 提交选模中的 C NASA 口径。C 使用逐样本指数惩罚的无截断总和，溢出显式记录为 `None` 并拒绝该候选。其他旧分析入口未统一迁移，默认仍保留截断均值，不能混用两种数字。
2. 为 C 增加联合选择条件。候选相对本轮实际父模型的 RMSE 必须下降超过 0.3，MAE、NASA 均不得上升，三项必须完整且有限。模型网关改选、控制器和漂移策略之后仍执行最终检查。
3. 新增 `RSI_V2_C_EFFICIENT=1`。该模式固定使用 pipeline，只生成一个续训候选，在实际父 checkpoint 上训练 10 epoch，学习率 0.0003，默认 seed 20261117。保留原模型结构、特征、标准化器和 Adam 状态；仅 public/train 参与 fit。取消该路径中 shrink、40 epoch 续训和 300 epoch 从头训练的组合搜索。
4. 将 `RSI_V2_PROBE_RETAIN=C` 提前到候选训练之前。需要保持原 C 模型时直接跳过训练，仍完成验证、模型网关调用、日志、状态、提交和结束协议。`fit_rows=0` 且 `fit_id_sha256=None`。
5. 本地 Docker 配方和无 Docker 镜像打包脚本已默认启用高效 C 候选，清空强制 retain。设置 `RSI_V2_PROBE_RETAIN=C` 可切换到保留父模型的节省计算方案。

正式运行依然需要真实模型网关调用。本次本地协议测试使用 fake gateway，不代表已验证真实 LLM 的行为或 Token 成本。

## 完整公开验证集结果

官方训练集 113,089 行，公开验证集 23,049 行。训练与验证 engine ID 不重合；实验检查数据 SHA 和父 checkpoint SHA。验证集用于比较和选模，未参与 fit。

以下均以官方初始 checkpoint 为父模型，Python 3.12.14、NumPy 1.26.4、Mac arm64、单线程运行。三个指标均为越低越好。

| 指标 | 官方父模型 | 新 10 epoch 候选 | 相对下降 |
|---|---:|---:|---:|
| RMSE | 52.868573 | 51.778500 | 2.0619% |
| MAE | 38.899830 | 38.003251 | 2.3048% |
| NASA 总和 | 5,823,439,325.34 | 3,798,023,257.04 | 34.7804% |

旧 40 epoch 候选在同一公开集上的 RMSE 为 51.358785，NASA 为 3,657,837,202.87，比新候选略好，但单纯模型训练 CPU 用时约为新候选的 3.82 倍。新候选选择了较低成本的折中，并不代表所有已试配置中指标最优。

旧 300 epoch 从头训练候选的 RMSE 为 51.198016，NASA 为 4,086,105,981.61。只选 RMSE 最低的模型会错过 NASA 的取舍。

分组风险仍然存在。FD001 的 NASA 从 1,577,040.57 升至 1,826,204.96，增加约 15.80%；FD002 的 MAE 从 31.6191 升至 31.8222，增加约 0.64%。本轮联合条件约束整个可见验证集，不能保证每个子集或隐藏阶段都改善。

额外的 seed 17、73、91 探索均改善整体三项指标，但这些早期探索使用 NumPy 2.3.5。最终默认候选和正式对照数字已用官方 NumPy 1.26.4 复核。按 106 个 engine 配对重采样 2,000 次，RMSE 差值的 2.5%–97.5% 分位为 [-1.6232, -0.5458]，NASA 相对变化为 [-38.98%, -23.85%]。由于候选已在同一验证集上选择，这只是重采样敏感性分析，不能当作独立检验或隐藏集收益保证。

## 资源测量

| 路径 | 单轮进程内 CPU 秒 | 本地端到端秒 | 最终模型 |
|---|---:|---:|---|
| 上游旧配置，先生成候选再强制 retain | 40.264 | 42.419 | 官方父模型 |
| 新高效候选 | 2.729 | 3.360 | 新续训模型 |
| 新提前 retain | 1.607 | 2.204 | 与父 checkpoint 逐字节相同 |

三种测量均为同机单线程、完整公开 C 数据、fake gateway、各一轮。新高效候选在这次测量中将进程内 CPU 用时降低 93.22%；提前 retain 降低 96.01%。CPU 字段来自本地进程记录，端到端时间由外部脚本测量。它们均不是平台资源效率分数。

单独比较模型 fit，旧 40 epoch 和 300 epoch 两个候选合计消耗 33.054 CPU 秒，新候选为 0.800 秒，下降 97.58%。这一比例不包含读取数据、验证、网关和提交，不能与上表混为同一个时间口径。

每条路径均保留一次网关调用。当前没有真实网关 Token 测量，没有隐藏环境 CPU 计量，也没有公开的资源效率 5 分换算公式。SDK 声明的 configuration 记录用于审计，不能据此推断平台一定按自报 CPU 改分。

## 验证范围

- 完整非 integration 测试集通过 228 项，3 项 integration 测试按标记排除。
- 最后的小幅修改后，针对 NASA、联合 gate、提前 retain、高效 C 入口及 LLM 越过 gate 的定向测试通过 12 项。
- 用 SDK 本地宿主连续运行 10 轮 C。固定公开数据下第一轮采用新候选，后九轮保留父模型；每轮一次 fake gateway 调用。
- 独立重新加载每轮 checkpoint，在完整 23,049 行验证数据上重算指标，三项均未逐轮恶化。协议预测仅检查每轮 3 行输入，完整指标验证由单独脚本完成。
- 最后源码另跑一轮正常入口，确实选择 `continue_train`，得到同一模型指标。
- 未验证真实模型网关、隐藏阶段分布、Linux 容器资源计量或最终比赛镜像。固定公开数据的十轮验证不等于企业赛十个隐藏阶段。

父 checkpoint SHA256 为 `98ec6e30e38aee56ca15b7d624a8f412226d46fe97b170667a567601c700bc4e`。

正常入口生成的新 checkpoint SHA256 为 `0676f96111f2b22522e1bcd1e57638c8bf971d683e72d748cbb2c4ad659f0e43`。

## 复现与产物

在本仓库根目录运行，Python 环境需有本项目依赖、pytest、NumPy 1.26.4，以及 `pi-agent-core-lc==0.9.1`、`pi-agent-harness-lc==0.9.1`。

```sh
# 训练候选并比较旧配置，输出包含完整分组结果
.venv/bin/python experiments/c_candidates.py --cost-comparison --output-dir outputs/c_candidates_numpy1264

# 非 integration 测试
RSI_V2_C_EFFICIENT=1 RSI_V2_PROBE_RETAIN='' .venv/bin/python -m pytest -q -m 'not integration'

# 十轮本地正常入口，再独立复算每轮模型的完整公开验证集指标
RSI_V2_C_EFFICIENT=1 RSI_V2_PROBE_RETAIN='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BASELINE_CPU_THREADS=1 .venv/bin/python scripts/emulate_workspace_v2.py --task C --rounds 10 --json > outputs/c_runtime_10rounds.json
.venv/bin/python experiments/verify_c_runtime.py

# 原始配置与提前 retain 的资源对照，需要未修改的基准仓库
.venv/bin/python experiments/resource_benchmark.py --baseline-repo /path/to/unchanged-17cc307b-repo
```

`outputs/c_candidates_numpy1264/cost_comparison.json` 保存最终完整实验数据。`outputs/candidate_summary.json` 保存精简结果。`outputs/resource_benchmark.json` 保存原版和提前 retain 的资源对照，`outputs/c_efficient_entry_benchmark.json` 保存高效候选入口测量。`outputs/c_runtime_verification.json` 保存十轮完整验证，`outputs/c_runtime_artifact/checkpoint.npz` 为正常入口生成的模型。

emulator 报告会引用系统临时目录。复验十轮模型时应连续执行上面两条命令；临时目录被清理后须重跑 emulator。最后一轮产物已经复制到持久 outputs 目录。

## 下一步判断依据

先用相同可见数据比较父模型与候选的 RMSE、MAE、NASA，再检查最差子集和计算成本。正式平台结果应同时读取隐藏集三项原始指标、阶段恢复/保持情况、宿主 CPU 计量状态与 Token。当前没有足够证据将这些公开指标直接换算为总分增量。

下一轮算法优先级是降低 FD001 的 NASA 退步，并检查 FD002 的 MAE；资源优先级是对训练内容、配置、父 checkpoint 完全一致的重复实验做安全复用。重复实验复用尚未实现，不能仅因 ID 或验证分数相同就跳过训练。

官方参考为 [SDK 1.0.4](https://rsi-index.115.zone/downloads/contest-harness-sdk-1.0.4.md)、[本次基准提交的 C 任务说明](https://github.com/JackWPP/zhixun-harness/blob/17cc307b10caed360405d739bdfb09f457910bad/datasets/cpu-c-guide-20261110-v1/TASK.md) 和 [C 训练指南](https://github.com/JackWPP/zhixun-harness/blob/17cc307b10caed360405d739bdfb09f457910bad/datasets/cpu-c-guide-20261110-v1/TRAINING_GUIDE.md)。
