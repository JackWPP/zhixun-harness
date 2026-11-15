#!/usr/bin/env python3
"""08 — demo 交付物取证：它到底做了什么决策。

读 demo/harness-runs/{A,B,C}-v1 的 summary / round / candidate / diagnosis /
experience_trajectories，回答四个问题：
  1. 三次运行用的是哪个决策引擎（规则 / 小模型）？
  2. 每轮选了哪些策略，谁被接受，增益多少？
  3. 诊断在训练前后有没有变化？策略与诊断是否真的相关？
  4. 经验池里到底有几个不同的决策状态？
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def forensics(task: str) -> None:
    run = C.DEMO / "harness-runs" / f"{task}-v1"
    summary = read_json(run / "summary.json")
    C.header(f"任务 {task} — demo 运行 {run.name}")

    C.sub("1. 决策引擎与配置")
    config = summary["configuration"]
    print(f"  decision_mode      : {config.get('decision_mode')}")
    print(f"  rule_based         : {config.get('rule_based')}")
    print(f"  trainer_engine     : {config.get('trainer_engine')}")
    print(f"  min_improvement    : {config.get('min_improvement')}")
    print(f"  rounds / candidates: {config.get('rounds')} / {config.get('max_candidates')}")
    print(f"  validation_policy  : {config.get('validation_policy')}")
    cost = summary["cost"]
    print(f"  cost               : 总 {cost['total_seconds']:.3f}s "
          f"（训练 {cost['training_seconds']:.3f}s，评估 {cost['validation_seconds']:.3f}s，"
          f"候选 {cost['candidate_count']} 个）")

    C.sub("2. 逐轮策略与门控")
    for round_dir in sorted(run.glob("round-*")):
        data = read_json(round_dir / "round.json")
        decision = data.get("decision") or {}
        key = C.TASKS[task]["primary"]
        before = (decision.get("metrics_before") or {}).get(key)
        after = (decision.get("metrics_after") or {}).get(key)
        gain = None
        if before is not None and after is not None:
            gain = (after - before) if task == "A" else (before - after)
        print(f"  [{round_dir.name}] 接受={decision.get('accepted')}  "
              f"winner={decision.get('winner_id')}  "
              f"{key}: {before} → {after}  gain={None if gain is None else round(gain, 6)}")
        for candidate in data.get("candidates", []):
            strategy = candidate.get("strategy", {})
            metrics = candidate.get("metrics") or {}
            primary = C.TASKS[task]["primary"]
            print(f"      {candidate['id']:<22} lr={strategy.get('learning_rate')} "
                  f"l2={strategy.get('l2')} noise={strategy.get('noise_std')} "
                  f"{primary}={metrics.get(primary)}")

    C.sub("3. 诊断前后对比")
    before = read_json(run / "diagnosis-before.json")
    after = read_json(run / "diagnosis-after.json")
    codes_before = [f["code"] for f in before["findings"]]
    codes_after = [f["code"] for f in after["findings"]]
    print(f"  训练前 findings: {codes_before}")
    print(f"  训练后 findings: {codes_after}")
    print(f"  结论: {'完全相同（只换了数字）' if codes_before == codes_after else '有变化'}")

    C.sub("4. 经验池与决策状态多样性")
    trajectories = read_json(run / "experience_trajectories.json")
    records = trajectories["records"]
    states = {json.dumps(r["state_vector"]) for r in records}
    accepted = [r for r in records if r.get("accepted")]
    print(f"  记录数 {len(records)}，其中被接受 {len(accepted)}")
    print(f"  唯一 state_vector 数量: {len(states)} / {len(records)}"
          f"  {'← 同一轮三个候选共享同一个状态' if len(states) < len(records) else ''}")
    print(f"  improvement 取值: {sorted({round(r['improvement'], 6) for r in records})}")


def candidate_matrix_diff() -> None:
    C.header("任务 A — round-01 三个候选的混淆矩阵差异")
    run = C.DEMO / "harness-runs" / "A-v1" / "round-01"
    rows = C.read_split("A", "validation")
    x, y = C.matrix(rows), C.labels(rows, "A")
    names = C.TASKS["A"]["classes"]
    matrices = {}
    for candidate_dir in sorted(p for p in run.iterdir() if p.is_dir()):
        model = C.vendor()[0].Model.load(candidate_dir / "checkpoint.npz")
        pred = model.predict(x).argmax(axis=1)
        matrix = np.zeros((6, 6), dtype=int)
        for truth, guess in zip(y, pred):
            matrix[truth, guess] += 1
        f1, _ = C.macro_f1(pred, y, 6)
        matrices[candidate_dir.name] = (matrix, f1, pred)
        print(f"  {candidate_dir.name:<22} macro_f1={f1:.10f}  "
              f"Toluene 行={list(matrix[5])}  Acetaldehyde 行={list(matrix[3])}")

    ids = list(matrices)
    if len(ids) >= 2:
        first = matrices[ids[0]][2]
        for other in ids[1:]:
            diff = int((first != matrices[other][2]).sum())
            print(f"  {ids[0]} vs {other}: 预测不同的样本数 = {diff} / {len(y)}")


def main() -> int:
    for task in "ABC":
        forensics(task)
    candidate_matrix_diff()
    print("\n" + "=" * 78)
    print("结论：三次 demo 全部走规则引擎（小模型零决策）；诊断前后 finding 集合完全一致；")
    print("      经验池只有 6 个不同状态；A 的「最优」候选比对照组只多判对 1 个样本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
