> **更正（2026-11-19）**：本文中的 A hidden proxy / +0.147 / “复现平台”等结论已被 `docs/36_HONEST_A_EVALUATION.md` 撤回；请以 docs/36 的 official-seq 结果为准。

# 35 · 平台同构复跑与 0.11.2 控制器决策

> 数据口径：隐藏批均为「公开 validation 留出段」代理，不是平台隐藏测试流；所有数字只用于本地决策，不构成线上分数承诺。

## 1. 背景

平台 trial #3（0.10.4 镜像，`bf8dc9d0…`）状态 `completed` 但 `eligible=false`、`score=null`：

| 数据集 | 指标 | trial #3 | 公开验证（最后一轮） |
|---|---|---|---|
| A | macro-F1 | **0.4768** | 0.6069 |
| B | RMSE | **735.07** | 503.67 |
| C | RMSE | **56.12** | 51.39 |

协议（train / predict / workspace_report / finish）全部通过，问题是**用公开验证选后处理 → 隐藏批偏弱**。

## 2. 平台同构模拟（0.11.1，kernel + drift policy）

真实 PyTRiO `Qwen/Qwen3.8-27B`，A/B/C 各 10 轮，每轮 6 次真实调用，策略 `pair_history` 在 A 上每轮触发：

| 数据集 | hidden 代理 | 对比 trial #3 |
|---|---|---|
| A | **0.4802** | 0.4768（同量级） |
| B | **504.43** | 735.07（更好） |
| C | **42.45** | 56.12（更好） |

结论：模拟能复现平台量级；0.11.1 的 drift policy 让 A 从「不进化」的 0.3330 提升到 0.4802。

## 3. 控制器实验（pipeline vs kernel）

同一模拟器、同一数据、policy on：

| 任务 | kernel | pipeline | 采用 |
|---|---|---|---|
| A（分类，高漂移） | **0.4802** | 0.4802（policy 生效） | kernel |
| B（回归） | 504.43 | **283.52** | pipeline |
| C（回归） | 42.45 | **38.83** | pipeline |

## 4. 0.11.2 决策与验证

- 镜像默认 `RSI_V2_CONTROLLER=auto`：分类走 kernel（保留 PI-agent 自主诊断/实验/决策 + 每轮真实调用），回归走 pipeline（确定性选臂，避免公开验证排名过拟合）。
- `RSI_V2_DRIFT_POLICY` 默认开，回归任务不触发（仅分类 high drift 生效）。
- 本地验证：`pytest -m "not integration"` **153 passed, 3 deselected**；emulator auto 路由 A kernel / B/C pipeline 各 2 轮 PASS；镜像 `0.11.2` verify PASS（380,221,440 B，sha256 `b83b4eae…`）。
- 预期（hidden 代理，非平台隐藏流）：A ≈ 0.480、B ≈ 283.5、C ≈ 38.8。平台上传/重跑 trial 待用户确认（剩余 7 次）。

## 5. 复现

```bash
# 模拟器（真实 provider，10 轮）
.venv/bin/python scripts/sim/full_env.py --scenario official --tasks A,B,C --rounds 10 \
  --provider real --controller auto --policy on --swanlab auto --tag platform-like-auto

# 镜像验证
bash scripts/stage_image.sh && .venv/bin/python scripts/build_image_tar.py \
  --rootfs .scratch/image/rootfs --overlay .scratch/image/overlay \
  --output .scratch/image/harness.tar --tag zhixun-rsi-harness:0.11.2
bash scripts/verify_image.sh
```
