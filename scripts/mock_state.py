#!/usr/bin/env python3
"""生成智训引擎 RSI Harness 的演示用 state.json（schema v1）。

用途：
  1. 没有真实 harness 运行时，给看板提供一份真实感数据；
  2. 演示视频 / 评审截图前快速铺一份"已跑 10 轮"的运行现场；
  3. 看板默认启动（`python -m rsi.dashboard`）时自动调用它。

用法：
    ./.venv/bin/python scripts/mock_state.py --out .scratch/demo_state.json
    ./.venv/bin/python scripts/mock_state.py --out .scratch/demo_state.json --records-out .scratch/records.jsonl

数据完全离线生成（只用标准库），不改动任何真实产物。mock 语义：
  - A：气体传感器漂移六类识别，已跑 6 轮后收敛，macro-F1 0.7140 → 0.7308；
  - B：轴承 RUL 回归，已跑 3 轮，第 4 轮正在诊断，RMSE 681.16 → 631.44；
  - C：涡扇发动机 RUL 回归，已跑 1 轮，第 2 轮运行中，RMSE 52.87 → 51.42。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = 1
RUN_ID = "rsi-20261115-1412"
ELAPSED_SECONDS = 1410.0  # 23 分 30 秒


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _sha(seed: str) -> str:
    """稳定的 64 位十六进制摘要，用来伪造 checkpoint sha256（mock 专用）。"""
    return hashlib.sha256(f"zhixun-mock::{seed}".encode("utf-8")).hexdigest()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _f(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


# --------------------------------------------------------------------------- #
# 三个任务的静态元信息
# --------------------------------------------------------------------------- #
TASK_META = {
    "A": {
        "name": "气体传感器漂移分类",
        "name_en": "Gas Sensor Drift Classification",
        "task": "classification",
        "primary_metric": "macro_f1",
        "direction": "higher",
        "baseline": 0.714030,
        "rounds_total": 10,
    },
    "B": {
        "name": "滚动轴承剩余寿命预测",
        "name_en": "Bearing RUL Prediction",
        "task": "regression",
        "primary_metric": "rmse",
        "direction": "lower",
        "baseline": 681.160,
        "rounds_total": 10,
    },
    "C": {
        "name": "涡扇发动机剩余寿命预测",
        "name_en": "Turbofan RUL Prediction",
        "task": "regression",
        "primary_metric": "rmse",
        "direction": "lower",
        "baseline": 52.869,
        "rounds_total": 10,
    },
}

DRIFT = {
    "A": {
        "severity": "severe",
        "score": 0.82,
        "feature_shift": 0.61,
        "ks_max": 0.342,
        "causes": [
            "batch 1 → batch 2 传感器响应漂移",
            "Acetaldehyde 跨批次完全不可迁移（F1 恒为 0）",
            "Toluene 验证样本仅 5 个，量化步长 0.008",
        ],
        "evidence": (
            "验证集 30/1244 行标准化后 |z| > 10；单特征 KS 最大 0.342；"
            "batch 2 上 Acetaldehyde 召回为 0、Toluene 仅 5 个样本。"
        ),
    },
    "B": {
        "severity": "mild",
        "score": 0.31,
        "feature_shift": 0.18,
        "ks_max": 0.117,
        "causes": [
            "验证集 90.2% 来自 Bearing3_4 单一轴承",
            "不同轴承间转速/载荷工况差异",
            "有效样本量只有 9 个训练轴承 + 3 个验证轴承",
        ],
        "evidence": (
            "分组后 RMS 特征 KS=0.117；常数预测 RMSE≈612，基线 681 反而更差；"
            "单轴承主导使验证 RMSE 置信区间约 ±38 min。"
        ),
    },
    "C": {
        "severity": "stable",
        "score": 0.06,
        "feature_shift": 0.03,
        "ks_max": 0.041,
        "causes": [
            "验证集为同分布随机切分，不存在漂移对象",
            "纯快照协议 window_sizes=[1]，无历史窗口可用",
        ],
        "evidence": (
            "验证集标准化 |z|max=3.0（A 为 299.6）；FD001–FD004 子集训练/验证比例几乎一致，"
            "漂移诊断只输出 snapshot_protocol / no_causal_labels 两个无条件结论。"
        ),
    },
}

TASK_BUDGET = {
    "A": {"tokens_used": 612_480, "cpu_seconds": 1286.4, "wall_seconds": 612.0, "experiments": 6},
    "B": {"tokens_used": 468_220, "cpu_seconds": 1442.8, "wall_seconds": 508.0, "experiments": 3},
    "C": {"tokens_used": 300_160, "cpu_seconds": 683.3, "wall_seconds": 290.0, "experiments": 2},
}


# --------------------------------------------------------------------------- #
# 轮次定义
# --------------------------------------------------------------------------- #
ROUND_SPECS = {
    "A": [
        {
            "status": "done",
            "decision": "promote",
            "hypothesis": (
                "验证集来自 batch 2 且冻结了 batch 1 的归一化统计，输出层分布已偏移；"
                "短程续训（不重拟合 scaler）应能让输出层重新对齐。"
            ),
            "change_summary": "continue_train：Adam lr=1e-3, 20 epochs, data=train（冻结归一化）",
            "strategy": "continue_train",
            "postprocess": "none",
            "metrics": {"macro_f1": 0.721800, "accuracy": 0.881200, "acetaldehyde_f1": 0.0},
            "done_at": 96,
        },
        {
            "status": "done",
            "decision": "promote",
            "hypothesis": "隐层特征仍有判别力，瓶颈在输出头；用闭式 ridge 重拟合输出层应比继续 Adam 更稳。",
            "change_summary": "ridge_last_layer：alpha=10.0, data=train（冻结隐层）",
            "strategy": "ridge_last_layer",
            "postprocess": "none",
            "metrics": {"macro_f1": 0.723600, "accuracy": 0.882500, "acetaldehyde_f1": 0.0},
            "done_at": 208,
        },
        {
            "status": "done",
            "decision": "promote",
            "hypothesis": "短续训的方差大于收益；对多次快照做权重平均（SWA）+ 类别均衡应降低方差。",
            "change_summary": "swa：lr=5e-4, 10 epochs, 3 snapshots, class_balance=true",
            "strategy": "swa",
            "postprocess": "none",
            "metrics": {"macro_f1": 0.727900, "accuracy": 0.886100, "acetaldehyde_f1": 0.0},
            "done_at": 322,
        },
        {
            "status": "done",
            "decision": "rollback",
            "hypothesis": "长程续训 + 特征增广可以补回 batch 2 的分布覆盖。",
            "change_summary": "continue_train：lr=3e-3, 40 epochs, augment=true",
            "strategy": "continue_train",
            "postprocess": "none",
            "metrics": {"macro_f1": 0.719400, "accuracy": 0.879300, "acetaldehyde_f1": 0.0},
            "done_at": 436,
        },
        {
            "status": "done",
            "decision": "promote",
            "hypothesis": "推理期批次级 recenter（按 checkpoint 冻结统计量重新标准化）可纠正 covariate shift。",
            "change_summary": "continue_train lr=5e-4 + postprocess recenter(batch_mean)",
            "strategy": "continue_train",
            "postprocess": "recenter:batch_mean",
            "metrics": {"macro_f1": 0.730800, "accuracy": 0.888600, "acetaldehyde_f1": 0.0},
            "done_at": 548,
        },
        {
            "status": "done",
            "decision": "retain",
            "hypothesis": "对回归头做 shrink 校准的思路可迁移到分类 logits 温度缩放，降低过度自信。",
            "change_summary": "ridge_last_layer alpha=1.0 + postprocess shrink(alpha=0.85)",
            "strategy": "ridge_last_layer",
            "postprocess": "shrink:alpha=0.85",
            "metrics": {"macro_f1": 0.731200, "accuracy": 0.888900, "acetaldehyde_f1": 0.0},
            "done_at": 604,
        },
    ],
    "B": [
        {
            "status": "done",
            "decision": "promote",
            "hypothesis": "基线在验证轴承上系统性高估 RUL；用 calibration 标签做短程续训可修正中心偏差。",
            "change_summary": "continue_train：lr=5e-4, 15 epochs, data=train+calibration",
            "strategy": "continue_train",
            "postprocess": "none",
            "metrics": {"rmse": 647.320000, "mae": 531.400000},
            "done_at": 706,
        },
        {
            "status": "done",
            "decision": "retain",
            "hypothesis": "若瓶颈在输出头，ridge 重拟合应显著降低 RMSE；否则说明表征不足。",
            "change_summary": "ridge_last_layer：alpha=100.0, data=train+calibration",
            "strategy": "ridge_last_layer",
            "postprocess": "none",
            "metrics": {"rmse": 652.810000, "mae": 537.900000},
            "done_at": 830,
        },
        {
            "status": "done",
            "decision": "promote",
            "hypothesis": "Ridge 证明表征是瓶颈；SWA 在低学习率下做方差缩减，应稳定优于单次续训。",
            "change_summary": "swa：lr=3e-4, 8 epochs, 3 snapshots（父检查点 = r1）",
            "strategy": "swa",
            "postprocess": "none",
            "metrics": {"rmse": 631.440000, "mae": 512.700000},
            "done_at": 942,
        },
        {
            "status": "running",
            "decision": None,
            "hypothesis": "按轴承分组的误差显示 Bearing3_4 主导验证集；尝试 group-aware 样本加权降低单轴承偏置。",
            "change_summary": "continue_train：lr=2e-4, 6 epochs, data=train+calibration, class_balance=false",
            "strategy": "continue_train",
            "postprocess": "none",
            "metrics": {},
            "done_at": 1006,
        },
    ],
    "C": [
        {
            "status": "done",
            "decision": "promote",
            "hypothesis": "同分布快照任务只需小幅续训；过长训练会过拟合 FD001–FD004 的工况差异。",
            "change_summary": "continue_train：lr=1e-3, 10 epochs, data=train",
            "strategy": "continue_train",
            "postprocess": "none",
            "metrics": {"rmse": 51.420000, "mae": 37.610000, "nasa_score": 311.0},
            "done_at": 1084,
        },
        {
            "status": "running",
            "decision": None,
            "hypothesis": "验证 RMSE 已低于基线 2.7%，下一轮验证 shrink 校准是否继续压低高 RUL 段的偏差。",
            "change_summary": "continue_train lr=5e-4 + postprocess shrink(alpha=0.92)",
            "strategy": "continue_train",
            "postprocess": "shrink:alpha=0.92",
            "metrics": {},
            "done_at": 1366,
        },
    ],
}


# --------------------------------------------------------------------------- #
# Panel 讨论（multi-agent 辩论记录）
# --------------------------------------------------------------------------- #
PANELS = {
    "A": [
        {
            "trigger": "漂移诊断：severity=severe（batch shift）",
            "round_index": 3,
            "verdict": "promote",
            "messages": [
                {
                    "role": "diagnostician",
                    "standpoint": "漂移是全局的，但可分特征：Acetaldehyde 完全不可迁移，属于标签/物理机制失效而非统计偏移。",
                    "evidence": "per-class F1：Acetaldehyde=0.000，Toluene=0.667（5 个样本）；标准化 |z| max=299.6 由单点造成。",
                    "confidence": 0.82,
                },
                {
                    "role": "experimenter",
                    "standpoint": "SWA + class_balance 不扩参、不重拟合 scaler，是当前预算下最稳的方差缩减方案。",
                    "evidence": "3 个 snapshot 的验证 macro-F1 分别为 0.7271/0.7283/0.7282，方向一致，标准差 0.0007。",
                    "confidence": 0.71,
                },
                {
                    "role": "critic",
                    "standpoint": "提升 +0.0043 小于 Toluene 单样本量化步长 0.008，无法排除是由 5 个样本翻面造成的。",
                    "evidence": "macro-F1 的 1/6 分量只由 5 个验证样本决定；同验证集重复评估的 bootstrap 95% CI 为 ±0.0061。",
                    "confidence": 0.64,
                },
                {
                    "role": "arbiter",
                    "standpoint": "promote。候选方向一致且回滚成本为零（保留父检查点），先吸收再在第 6 轮做显著性复核。",
                    "evidence": "门控规则：Δ>0 且 3/3 快照同向 → promote；父检查点 sha 已记录可回滚。",
                    "confidence": 0.68,
                },
            ],
        },
        {
            "trigger": "终轮复核：Toluene 量化噪声 vs 显著性阈值",
            "round_index": 6,
            "verdict": "retain",
            "messages": [
                {
                    "role": "diagnostician",
                    "standpoint": "recenter 后 batch 对齐基本完成，残差集中在 Acetaldehyde；漂移已从 severe 降为中残差问题。",
                    "evidence": "recenter 后验证集 |z|>10 的行由 30 降到 4；per-class F1 除 Acetaldehyde 外均 >0.68。",
                    "confidence": 0.79,
                },
                {
                    "role": "experimenter",
                    "standpoint": "shrink(alpha=0.85) 只校准输出尺度，预计把高置信错误压低，但不会改变类别排序。",
                    "evidence": "候选 macro-F1=0.7312（父=0.7308）；accuracy 同步 +0.0003，方向一致但量级极小。",
                    "confidence": 0.66,
                },
                {
                    "role": "critic",
                    "standpoint": "Δ=+0.0004 远低于预先登记的 0.002 决策阈值，不构成可复现提升，应拒绝。",
                    "evidence": "阈值 0.002 在 run 开始前固定；候选与父检查点在同一验证集上评估，无独立留出集。",
                    "confidence": 0.86,
                },
                {
                    "role": "arbiter",
                    "standpoint": "retain。保留父检查点，把 6 轮证据链与阈值写进报告；这是「拒绝不显著提升」的正向案例。",
                    "evidence": "决策日志：6 轮 4 promote / 1 rollback / 1 retain；最终 best=0.7308，相对基线 +2.35%。",
                    "confidence": 0.9,
                },
            ],
        },
    ],
    "B": [
        {
            "trigger": "验证集单轴承偏置：RMSE 置信区间过宽",
            "round_index": 2,
            "verdict": "retain",
            "messages": [
                {
                    "role": "diagnostician",
                    "standpoint": "误差与轴承分组强相关，表征对工况不敏感；但验证集 90.2% 来自 Bearing3_4，任何结论都只对该轴承成立。",
                    "evidence": "按轴承分组的 RMSE：Bearing3_4=648.1、Bearing1_4=702.4、Bearing2_4=389.6；有效样本量 n=3。",
                    "confidence": 0.8,
                },
                {
                    "role": "experimenter",
                    "standpoint": "Ridge alpha=100 的 RMSE=652.8 反而更差，说明瓶颈不在输出头；下一轮应改走 SWA 低方差路线。",
                    "evidence": "Ridge 解在训练集残差下降但验证集回升，典型表征不足信号。",
                    "confidence": 0.74,
                },
                {
                    "role": "critic",
                    "standpoint": "验证集由单一轴承主导，+33.8 的「提升」落在置信区间内，不能据此宣称模型变好。",
                    "evidence": "单轴承主导下 bootstrap 95% CI ≈ ±38.4 min；与常数预测 612 相比基线仍然更差。",
                    "confidence": 0.83,
                },
                {
                    "role": "arbiter",
                    "standpoint": "retain。拒绝 Ridge 候选，保留 r1；要求后续实验必须报告分组指标与区间，不只报总 RMSE。",
                    "evidence": "门控规则：验证集分组覆盖不足时，只接受方向一致且幅度 > CI 半宽的候选。",
                    "confidence": 0.85,
                },
            ],
        }
    ],
    "C": [
        {
            "trigger": "同分布快照任务：首轮续训收益确认",
            "round_index": 1,
            "verdict": "promote",
            "messages": [
                {
                    "role": "diagnostician",
                    "standpoint": "C 的验证集是同分布随机切分，漂移诊断没有对象；只输出 snapshot_protocol 与 no_causal_labels 两个无条件结论。",
                    "evidence": "验证集 |z|max=3.0；FD001–FD004 在训练/验证中的比例几乎一致；无时间窗口可用。",
                    "confidence": 0.9,
                },
                {
                    "role": "experimenter",
                    "standpoint": "短程续训 + shrink 只做输出校准，预期小步下降，风险与成本都最低。",
                    "evidence": "10 epochs 续训后 RMSE 52.87→51.42，MAE 38.90→37.61，方向一致。",
                    "confidence": 0.7,
                },
                {
                    "role": "arbiter",
                    "standpoint": "promote。收益虽小但稳定，继续第 2 轮验证 shrink 是否在高 RUL 段带来额外收益。",
                    "evidence": "门控规则：Δ=-2.7% 且同分布任务无漂移干扰 → promote；保留父检查点。",
                    "confidence": 0.78,
                },
            ],
        }
    ],
}


# --------------------------------------------------------------------------- #
# 事件流（按发生时间升序）
# --------------------------------------------------------------------------- #
EVENT_SPECS = [
    (2, "info", "", "harness 启动：run_id={run_id}，3 个任务，每任务最多 10 轮，全局 token 上限 4,000,000"),
    (4, "info", "A", "加载基线 checkpoint（A）sha256:fd1e08b2aad0…，参数 4326，验证集 batch 2 / 1244 行"),
    (7, "info", "B", "加载基线 checkpoint（B）sha256:9c31be04f2d1…，参数 865，验证集 3 个轴承 / 1679 行"),
    (10, "info", "C", "加载基线 checkpoint（C）sha256:4ab7d9031e6c…，参数 833，验证集 62 台发动机 / 23049 行"),
    (14, "info", "A", "LLM 诊断调用 #1：local/Qwen3-32B-Instruct，输入 8.2k tokens，输出 1.1k tokens，耗时 6.4s"),
    (18, "warn", "A", "漂移诊断：severity=severe（score=0.82），batch 1→2 出现协变量偏移，Acetaldehyde 跨批次不可迁移"),
    (24, "info", "A", "实验 exp-A-r1 启动：continue_train lr=1e-3 epochs=20 data=train（父检查点已校验 sha）"),
    (38, "info", "A", "实验 exp-A-r1 完成：macro_f1 0.714030→0.721800（+1.09%），决策 promote"),
    (44, "info", "A", "原子写入 state.json 与 records.jsonl：A/r1 promote，新最优 sha256:b1a3f7…"),
    (58, "info", "A", "实验 exp-A-r2 启动：ridge_last_layer alpha=10，验证「头 vs 表征」瓶颈假设"),
    (72, "info", "A", "实验 exp-A-r2 完成：macro_f1 0.723600，连续两次提升，决策 promote"),
    (96, "info", "A", "Panel 讨论触发：severity=severe 上升为多角色议题（诊断者/实验者/评审者/仲裁者）"),
    (104, "info", "A", "实验 exp-A-r3 启动：swa lr=5e-4 snapshots=3 class_balance=true"),
    (118, "info", "A", "实验 exp-A-r3 完成：macro_f1 0.727900（+0.43%），决策 promote"),
    (126, "warn", "A", "评审者质疑：+0.0043 小于 Toluene 单样本量化步长 0.008，bootstrap 95% CI=±0.0061"),
    (132, "info", "A", "仲裁者裁定 promote：3/3 快照同向且可回滚，转入第 4 轮反证实验"),
    (146, "info", "A", "实验 exp-A-r4 启动：continue_train lr=3e-3 epochs=40 augment=true"),
    (168, "error", "A", "实验 exp-A-r4 候选验证 macro_f1=0.719400（-1.16%），触发门控 rollback；原因：batch 1 过拟合"),
    (175, "warn", "A", "回滚完成：恢复父检查点 sha256:8d02c1…，实验目录保留用于审计"),
    (188, "info", "A", "诊断更新：漂移残差集中于 Acetaldehyde，recenter 推理期校准进入候选"),
    (196, "info", "A", "实验 exp-A-r5 启动：continue_train lr=5e-4 + recenter(batch_mean)"),
    (212, "info", "A", "实验 exp-A-r5 完成：macro_f1 0.730800（+0.40%），new best，决策 promote"),
    (226, "info", "B", "任务 A 达到收敛判据（连续 2 轮 Δ<0.002），标记 done；转入任务 B 深度诊断"),
    (232, "info", "B", "LLM 诊断调用 #2：按轴承分组误差分析，输入 11.6k tokens，输出 1.4k tokens"),
    (240, "warn", "B", "验证集构成告警：Bearing3_4 占 90.2%（1515/1679 行），RMSE 置信区间约 ±38.4 min"),
    (248, "info", "B", "实验 exp-B-r1 启动：continue_train lr=5e-4 epochs=15 data=train+calibration"),
    (266, "info", "B", "实验 exp-B-r1 完成：RMSE 681.160→647.320（-4.97%），MAE 563.11→531.40，决策 promote"),
    (276, "info", "B", "实验 exp-B-r2 启动：ridge_last_layer alpha=100，检验输出头瓶颈"),
    (292, "info", "B", "实验 exp-B-r2 完成：RMSE=652.810（劣于 r1），决策 retain；表征瓶颈假设成立"),
    (300, "info", "B", "Panel 讨论：验证集单轴承主导，评审者要求后续报告分组指标与置信区间"),
    (308, "info", "B", "实验 exp-B-r3 启动：swa lr=3e-4 epochs=8 snapshots=3（父检查点=r1）"),
    (326, "info", "B", "实验 exp-B-r3 完成：RMSE=631.440（-2.45%），MAE 512.70，决策 promote"),
    (338, "info", "C", "任务 C 启动：同分布快照任务，漂移诊断无对象（snapshot_protocol / no_causal_labels）"),
    (344, "info", "C", "实验 exp-C-r1 启动：continue_train lr=1e-3 epochs=10 data=train"),
    (362, "info", "C", "实验 exp-C-r1 完成：RMSE 52.869→51.420（-2.74%），NASA score=311.0，决策 promote"),
    (372, "info", "A", "终轮复核 exp-A-r6：候选 macro_f1=0.7312，Δ=+0.0004 < 0.002 阈值"),
    (378, "info", "A", "实验 exp-A-r6 完成：决策 retain（拒绝不显著提升），最终 best=0.7308"),
    (388, "info", "", "预算快照：tokens 1,380,860 / 4,000,000（34.5%），CPU 3412.5s / 7200s，实验 11/30"),
    (398, "info", "B", "任务 B 进入第 4 轮：group-aware 样本加权实验规划中（LLM 调用 #3）"),
    (404, "info", "C", "任务 C 进入第 2 轮：shrink(alpha=0.92) 输出校准实验启动"),
    (410, "info", "C", "实验 exp-C-r2 运行中：续训 10 epochs → 推理期 shrink 校准，预计 90s"),
    (418, "info", "", "records.jsonl 追加 1 条决策记录；state.json 原子替换完成（mtime 更新）"),
    (424, "info", "B", "等待 B 的 LLM 诊断响应（第 4 轮提案），当前预算充足，无需触发降级策略"),
    (430, "info", "A", "证据归档：A 任务 6 轮轨迹写入 experience_trajectories.json，供跨任务策略检索"),
    (436, "info", "", "healthcheck：dashboard 未运行不影响 harness；看板每 2s 轮询 state.json"),
    (442, "warn", "C", "验证集高 RUL 段（>180 循环）残差偏大，shrink 校准幅度存在过校正风险"),
    (448, "info", "B", "LLM 调用 #3 返回：建议 group-aware 加权 + 更严格的分组早停，置信度 0.71"),
    (452, "error", "", "实验子进程 exp-B-r4 首次启动失败（CUDA 不可用回退 CPU 超时 30s），已自动重试，仅记录不占实验次数"),
    (458, "info", "", "自动重试成功：exp-B-r4 以纯 CPU 模式启动，超时预算 900s"),
    (462, "info", "C", "exp-C-r2 中期指标：验证 RMSE 51.31（-0.21%），继续完成后走门控"),
    (468, "info", "A", "任务 A 最终报告段落生成完成：6 轮 4 promote / 1 rollback / 1 retain，证据链完整"),
]


# --------------------------------------------------------------------------- #
# 构建 state
# --------------------------------------------------------------------------- #
def _make_rounds(task_key: str, started: datetime) -> list[dict]:
    meta = TASK_META[task_key]
    accepted_sha = _sha(f"baseline-{task_key}")
    rounds: list[dict] = []
    for idx, spec in enumerate(ROUND_SPECS[task_key], start=1):
        candidate_sha = _sha(f"{task_key}-r{idx}-{spec['strategy']}-{spec['postprocess']}")
        rounds.append(
            {
                "index": idx,
                "id": f"exp-{task_key}-r{idx}",
                "status": spec["status"],
                "decision": spec["decision"],
                "hypothesis": spec["hypothesis"],
                "change_summary": spec["change_summary"],
                "strategy": spec["strategy"],
                "postprocess": spec["postprocess"],
                "metrics": {k: _f(v, 6) for k, v in spec["metrics"].items()},
                "parent_checkpoint_sha256": accepted_sha,
                "candidate_checkpoint_sha256": candidate_sha,
                "created_at": _iso(started + timedelta(seconds=spec["done_at"])),
            }
        )
        if spec["decision"] == "promote":
            accepted_sha = candidate_sha
    return rounds


def _task_state(task_key: str, started: datetime) -> dict:
    meta = TASK_META[task_key]
    rounds = _make_rounds(task_key, started)
    done = [r for r in rounds if r["status"] == "done"]
    accepted = None
    for r in done:
        if r["decision"] == "promote":
            accepted = r
    best = float(accepted["metrics"][meta["primary_metric"]]) if accepted else None
    current = best
    # A 的最后一次 promote 是 r5；r6 为 retain，所以 best/current 落在 r5。
    status = "done" if task_key == "A" else "running"
    round_index = rounds[-1]["index"]
    rounds_total = meta["rounds_total"]
    return {
        "key": task_key,
        "name": meta["name"],
        "name_en": meta["name_en"],
        "task": meta["task"],
        "primary_metric": meta["primary_metric"],
        "direction": meta["direction"],
        "baseline": _f(meta["baseline"], 6),
        "best": best,
        "current": current,
        "status": status,
        "round_index": round_index,
        "rounds_total": rounds_total,
        "drift": DRIFT[task_key],
        "budget": TASK_BUDGET[task_key],
        "rounds": rounds,
        "panel": PANELS[task_key],
    }


def _events(started: datetime) -> list[dict]:
    events = []
    for offset, level, task, message in EVENT_SPECS:
        if "{run_id}" in message:
            message = message.replace("{run_id}", RUN_ID)
        events.append(
            {
                "ts": _iso(started + timedelta(seconds=offset)),
                "level": level,
                "task": task,
                "message": message,
            }
        )
    return events


def build_state(now: datetime | None = None) -> dict:
    """构造一份 schema v1 的 mock state。`now` 便于测试固定时间。"""
    now = now or datetime.now(timezone.utc)
    started = now - timedelta(seconds=ELAPSED_SECONDS)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": RUN_ID,
            "status": "running",
            "started_at": _iso(started),
            "elapsed_seconds": ELAPSED_SECONDS,
            "llm": {
                "backend": "local",
                "model": "Qwen3-32B-Instruct",
                "calls": 148,
                "tokens_in": 1_284_320,
                "tokens_out": 96_540,
            },
            "budget": {
                "tokens_used": 1_380_860,
                "tokens_cap": 4_000_000,
                "cpu_seconds": 3_412.5,
                "cpu_cap": 7_200.0,
                "wall_seconds": ELAPSED_SECONDS,
                "wall_cap": 4_800.0,
                "experiments_used": 11,
                "experiments_cap": 30,
            },
        },
        "tasks": [_task_state(key, started) for key in ("A", "B", "C")],
        "events": _events(started),
        "_demo": {
            "generator": "scripts/mock_state.py",
            "note": "演示数据，非真实 harness 产物",
        },
    }


def write_state(state: dict, out_path: Path) -> None:
    """原子写入（临时文件 + os.replace），避免看板读到半截 JSON。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".mock_state.", suffix=".tmp", dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp_name, out_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_records(state: dict, records_path: Path) -> None:
    """可选：写一份 records.jsonl 样例，供看板验证事件流合并逻辑。"""
    records_path = Path(records_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    run = state["run"]
    lines: list[dict] = [
        {
            "type": "run_start",
            "ts": run["started_at"],
            "run_id": run["run_id"],
            "message": f"run 开始：{run['run_id']}",
        }
    ]
    for event in state["events"][-6:]:
        lines.append(
            {
                "type": "event",
                "ts": event["ts"],
                "level": event["level"],
                "task": event["task"],
                "message": event["message"],
            }
        )
    for task in state["tasks"]:
        for rnd in task["rounds"]:
            if rnd["status"] != "done":
                continue
            lines.append(
                {
                    "type": "round",
                    "ts": rnd["created_at"],
                    "task": task["key"],
                    "round_index": rnd["index"],
                    "decision": rnd["decision"],
                    "message": (
                        f"{task['key']} 第 {rnd['index']} 轮完成：决策 {rnd['decision']}，"
                        f"策略 {rnd['strategy']}"
                    ),
                }
            )
    with records_path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成智训引擎 RSI Harness 演示用 state.json（schema v1，纯离线）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default=".scratch/demo_state.json", help="输出 state.json 路径")
    parser.add_argument("--records-out", default=None, help="可选：同时写一份 records.jsonl 样例")
    parser.add_argument("--indent", type=int, default=2, help="JSON 缩进（0 为压缩）")
    args = parser.parse_args(argv)

    state = build_state()
    out_path = Path(args.out)
    if args.indent <= 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False)
            fh.write("\n")
    else:
        # 复用 write_state 的原子写；indent 固定 2 即可满足展示需求。
        write_state(state, out_path)
    if args.records_out:
        write_records(state, Path(args.records_out))

    done_rounds = sum(1 for t in state["tasks"] for r in t["rounds"] if r["status"] == "done")
    print(f"[mock] 已生成 {out_path}（schema v{SCHEMA_VERSION}，{len(state['tasks'])} 任务，"
          f"{done_rounds} 个已完成轮次，{len(state['events'])} 条事件）")
    if args.records_out:
        print(f"[mock] 已生成 {args.records_out}（records.jsonl 样例）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
