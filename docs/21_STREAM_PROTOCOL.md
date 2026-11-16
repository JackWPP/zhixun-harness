# 批次流协议（batch-stream protocol）

赛题评测按批次下发数据：**每批都带标签**，每次会把此前所有数据再给一次；程序必须先输出
“当前批次采用哪个模型 / 不上线”的判断，才能拿到下一批。平台的具体目录/调用方式尚未公开
（见 `docs/19_SUBMISSION_CONTRACT.md`），因此这里定义一个与平台假设解耦、可在本地 emulator
中完整演练的协议，并同时由 `rsi.cli stream` 与 `scripts/emulate_stream_platform.py` 实现/校验。

## 1. 输入：一批数据

本地把一批建模为小 package（与官方包布局一致，可被现有 `Runner` 直接消费）：

```
batch_0003/
  package/
    public/train.jsonl      # 累计可训练数据（官方 train + 之前各批已交付的 train 部分）
    public/validation.jsonl # 当前批的验证行（带 answer，供 harness 内部选择/报告）
    checkpoint.npz          # 当前父模型（上一批部署的模型；第一批是官方 baseline）
  run/                      # Runner 的完整训练循环产物
  decision.json             # 本批决策（见下）
  predictions.jsonl         # 当前批逐行预测（官方输入输出形状）
```

`rsi/stream/sources.py` 提供两种来源：

* `official` —— 按顺序把官方 validation 流切成 N 批；第 i 批训练集 = 官方 train +
  第 0..i-1 批标签（绝不包含第 i 批验证行）。
* `external-a` —— Track X 测试床：UCI gas 的 batch3..10 每批按 70/30 切 train/eval，
  只用于验证“harness 能否在持续到达的新领域数据上继续迭代”，**不进入官方提交谱系**。

## 2. 处理：每批一次完整 RSI 循环

`rsi/stream/runner.py` 对每批调用未修改的 `rsi.runner.Runner(rounds=1)`，因此仍然经过：

漂移监测 → 诊断/归因 → 实验（Guard 校验）→ 评估（selection）→ 决策（promote/retain/rollback）
→ 预测 → 记录 → 记忆/SQLite。

与静态 10 轮的区别只有两点：

1. 每批的 package 会替换（数据随批次增长，父 checkpoint 来自上一批部署的模型）；
2. 每批结束后 *立即* 写 `decision.json` 与 predictions，然后才推进父模型。

## 3. 输出：决策 + 预测

`decision.json`（每批必有；同时写平台最可能使用的扁平字段与本地调试用的 `decision_detail` 嵌套记录）：

```json
{
  "batch": 3,
  "round_index": 3,
  "decision": "promote",
  "delta": 0.03,
  "reason": "...",
  "decision_detail": {"decision": "promote", "delta": 0.03, "reason": "..."},
  "parent_checkpoint_sha256": "sha256:...",
  "candidate_checkpoint_sha256": "sha256:...",
  "checkpoint_sha256": "sha256:...",
  "validation_metrics": {"accuracy": 0.9, "macro_f1": 0.88},
  "parent": {"checkpoint": "...", "primary": 0.72, "postprocess": {...}},
  "candidate": {"checkpoint": "...", "primary": 0.75, "selection": {...}},
  "chosen": {"checkpoint": "...", "primary": 0.75, "postprocess": {...}},
  "predictions": {"count": 311, "path": ".../predictions.jsonl"},
  "run_dir": ".../run",
  "controller": "pipeline|kernel",
  "degraded": false
}
```

`predictions.jsonl`：每行 `{"id": "...", "outputs": ...}`，分类为
`{"outputs":[{"label","score"},...]}`，回归为 `{"outputs":{"value":float}}`，
与官方 `predict.py` 完全一致。

`stream_report.json`（运行级）：每任务的批次记录、retention（每隔 N 批在所有历史批次上
复评，衡量遗忘）、最终部署 checkpoint、错误列表。

## 4. 本地运行

```bash
# Track S：官方数据批次流（严格口径）
./.venv/bin/python -m rsi.cli stream --tasks A --source official --batches 8 \
  --llm off --state-dir .scratch/stream-s

# Track X：外部数据持续迭代测试床
RSI_ALLOW_EXTENDED_DATA=1 ./.venv/bin/python -m rsi.cli stream --tasks A --source external-a \
  --batches 4 --external-root extra/uci-gas/Dataset --holdout 0.5 \
  --llm off --state-dir .scratch/stream-x
```

关键参数：`--controller auto|kernel|pipeline`、`--parallel-tasks N`（任务间并行）、
`--experiments-cap`（每批预算）、`--batch-wall`、`--retention-every`、`--no-memory`。

## 5. 平台对接

`scripts/emulate_stream_platform.py`（contractlab 交付）用官方 validation 构造 10 批目录，
按 `docs/19_SUBMISSION_CONTRACT.md` 的三级清单（已证实/推断/未知）校验输出 schema。
平台真实契约未知时，采用：

* 数据目录通过入口脚本的已知路径约定读取（`entrypoint.sh` 支持 `RSI_STREAM_DIR`）；
* 每批结束写 `decision.json` + `predictions.jsonl` 到工作目录；
* 门禁：输入输出 schema 校验失败 → 写入 error 标记并保留上一版预测，不静默失败；
* 超时/预算不足 → 提前输出当前最优模型的预测并标记 `degraded`。

**合规边界**：Track S 严格只用官方 train；验证/隐藏标签永不参与拟合；Track X 的外部数据
不进入正式 checkpoint 谱系（`data_policy=external-testbed`）。

## 6. 整夜 Track X 实测（v0.6.0）

- 流：UCI gas batch6–9（4 批，每批 50/50 分层 train/eval），`--experiments-cap 2`、离线
  pipeline、`RSI_ALLOW_EXTENDED_DATA=1`；每批完整走“诊断→候选→执行→门控→预测→记录”。
- 结果（详见 docs/12 §12.21）：harness 部署模型 vs frozen：batch6 0.4697 vs 0.3209、
  batch7 0.4871 vs 0.3395、batch8 0.4053 vs 0.2563、batch9 0.5511 vs 0.1868；全部优于
  frozen 与 recenter-only；oracle（用该批标签重训）0.95–0.99 为测试床上界。
- 契约验证：4 批均产出 `decision.json`（扁平 `decision` 字符串 + `decision_detail`）、
  `predictions.jsonl`、`batch/run/summary.json`、`stream_report.json`；`decision` 字段与
  docs/19 的平台假设一致。
- 边界：Track X 只在 `RSI_ALLOW_EXTENDED_DATA=1` 下运行；外部数据仅用于测试床，不进入
  Track S 的 checkpoint 谱系、manifest 或 workspace_report；正式提交仍为严格官方数据口径。

## 7. 历史标签自适应（P0 落地，v0.7 起）

> 说明：§6 的 external-a 源在 v0.6 时把当前批 train 行也纳入了训练（存在当前批标签泄漏，
> 数字偏乐观）。v0.7 起 `external_a_batches` 改为“batch i 的标签在 batch i+1 才释放”
> （meta 新增 `pending_rows` / `labels_released_after_batch`），§6 保留为历史记录。

- `stream --adapt-history`：第 i 批（i>0）先以「官方 train + 前 i-1 批标签」连续训练
  30 epoch（lr 5e-4，无结构改变），再做当前批 `recenter(batch_mean, w=0.5)`；当前批
  标签从不参与当前批预测。entrypoint 默认开启（`RSI_STREAM_ADAPT=1`）。
- 证据（leak-free，8 批）：外部 UCI b3–b10 macro-F1 均值 frozen .3977 / recenter .5046 /
  **history-adapt .7733**；官方 validation 顺序 8 批 frozen .6158 → adapt .6361。
  逐批表与决策见 docs/23 §8、docs/12 §12.23。
- 边界：checkpoint 结构/归一化/格式不变、父谱系连续；仍走完整 Runner/Guard/记忆/记录；
  只影响批次流路径，不改变单次 run 或 submission 的 predict 契约。

## 8. 延迟标签的 pair_threshold 校准（v0.7+，默认开启）

- 数据流：`batch_0..batch_n`，每批预测后返回该批标签；`decision.json` 在预测时生成，`pair_calibration` 只使用**更早批次**的标签。
- 每批开始时：`fit_pair_threshold` 在累计历史 `X/y` 上搜索 AA/Eth 判定阈值 `delta`；门槛：Eth≥30、AA≥10、且历史二类平衡准确率提升≥0.02，否则禁用。
- 叠加顺序：`recenter`（若有）→ `pair_threshold`（按 raw logits 的 `logit[3]-logit[0]` 重判 AA/Eth）。
- 官方 8 批对照：off 0.7269 → on 0.7680（第 5 批 .333→.662，其余不变）。
- 关闭：CLI `--no-pair-calibration`；镜像环境变量 `RSI_STREAM_PAIR=0,3` 可改 pair。
- 注意：该机制只改善“判定方向”，不改变 checkpoint；decision 记录里保留 delta/rows/score 便于审计。

## 9. v0.8 诚实协议、门控与 auto 路由

- **协议**：`Batch(train, dev, prediction, answers, reveal)`；当前批无标签，预测后才评分；hidden 批永不揭示。
  `materialise_package` 写 `public/{train,validation,prediction}.jsonl`，Runner 不再接触当前批标签。
- **arms**：`frozen/recenter/pair/adapt/full`；`adapt` 用已揭示历史续训父模型后做当前批 recenter；
  `full` 另走 Runner 实验选型。`decision.json` 与 `evaluation.json` 分离（决策不含答案依赖字段）。
- **滚动门控**：`adapt_gate_mode="recent"`，用最近一次已揭示批次（≤2000 行）比较 before/after，
  不改善则回滚父 checkpoint；证据：A external full .781→.787，B/C 无法救回 → 门控只作安全网。
- **auto 路由**（默认）：A official `pair` / external `full`；B `adapt`；C `frozen`。
  实现 `rsi.stream.runner.resolve_arm`，单元测试固化；显式 `--arm` 可覆盖。
- **证据**：`docs/evidence/private_sim.md`；实验表 docs/12 §12.24；防泄漏 E0 PASS、未知任务 E7、kernel E8。
