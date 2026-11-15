"""Orchestrate diagnosis -> candidate continuation (via RuleDrivenHarnessTrainer) -> validation gate -> audit."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
from typing import Optional

from .data import digest, prepare_bundle, write_json
from .diagnostics import diagnose, improvement, primary_metric
from .strategies import StrategyRecommenderMLP, choose_strategies, diagnosis_to_state_vector
from .training import CostLedger, ExperienceBuffer, RuleDrivenHarnessTrainer, TrajectoryRecord, run_candidate


WARNINGS = [
    "本地开发参考，不是正式比赛成绩；未访问隐藏数据。",
    "只用 train 优化权重/构造采样权重与扰动；validation 只用于诊断、候选评估和检查点选择。",
    "多轮针对公开验证集选策略可能过拟合；提升不保证迁移至隐藏集。",
    "诊断是统计关联与候选假设，不是真实漂移原因或因果结论。",
    "给定检查点已经见过全部公开 train，重新从 train 切分不能得到从未见过的独立验证。",
    "训练预算在 mini-batch 边界检查，可能超出最后一个 batch 的耗时；诊断/评估/IO 不计入训练预算但计入总耗时。",
    "候选早停与接受仅按任务主指标；C 的本地 NASA 常规分数仅作辅助，未确认与平台公式一致。",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def select_candidate(task, incumbent_metrics, candidates, min_improvement=0.0):
    if not math.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and nonnegative")
    winner = None
    best_gain = min_improvement
    for candidate in candidates:
        gate = candidate["gate"]
        if candidate.get("status") not in ("completed", "budget_exhausted"):
            gate.update(eligible=False, reason="候选执行失败或状态无效，禁止进入门控", improvement=None)
            continue
        if not gate["eligible"] or not candidate.get("metrics") or not candidate.get("best_checkpoint"):
            continue
        gain = improvement(task, incumbent_metrics, candidate["metrics"])
        gate["improvement"] = gain
        qualifies = math.isfinite(gain) and gain > min_improvement
        gate["passes_primary_threshold"] = qualifies
        gate["reason"] = "达到主指标改进阈值，参与本轮择优" if qualifies else "主指标未严格超过最小改进阈值，保留当前模型"
        if qualifies and gain > best_gain:
            winner, best_gain = candidate, gain
    if winner:
        winner["gate"]["reason"] = "本轮主指标改进最大，接受；相同增益优先先执行的低复杂度候选"
    return winner


def run(
    task,
    archive,
    output,
    epochs=5,
    max_candidates=3,
    rounds=1,
    min_improvement=0.0,
    max_train_seconds=60.0,
    patience=2,
    decision_mode="rule",
    strategy_model_path: Optional[str] = None,
):
    if task not in ("A", "B", "C"):
        raise ValueError("invalid task")
    if not 1 <= rounds <= 20 or not 1 <= patience <= 1000:
        raise ValueError("rounds must be 1..20; patience must be 1..1000")
    if not 1 <= epochs <= 1000 or not 1 <= max_candidates <= 4:
        raise ValueError("epochs must be 1..1000; max_candidates must be 1..4")
    if not math.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and nonnegative")
    if decision_mode not in ("rule", "small_model", "hybrid"):
        raise ValueError("decision_mode must be rule, small_model or hybrid")

    strategy_model = None
    if strategy_model_path:
        strategy_model = StrategyRecommenderMLP.load(Path(strategy_model_path))
    elif decision_mode in ("small_model", "hybrid"):
        default_pretrained = Path(__file__).parent / "strategy_recommender_pretrained.npz"
        if default_pretrained.is_file():
            strategy_model = StrategyRecommenderMLP.load(default_pretrained)

    ledger = CostLedger(max_train_seconds)
    experience_buffer = ExperienceBuffer()
    trainer = RuleDrivenHarnessTrainer(patience=patience, experience_buffer=experience_buffer)

    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output must be a NEW directory; previous runs are never overwritten")
    if not output.parent.is_dir():
        raise ValueError("output parent must exist")
    output.mkdir()
    started = time.perf_counter()
    events = output / "events.jsonl"

    def event(kind, payload):
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": utc_now(), "event": kind, **payload}, ensure_ascii=False, allow_nan=False) + "\n")

    configuration = {
        "epochs": epochs,
        "max_candidates": max_candidates,
        "rounds": rounds,
        "min_improvement": min_improvement,
        "max_train_seconds": max_train_seconds,
        "patience": patience,
        "primary_metric": primary_metric(task),
        "validation_policy": "evaluation_and_selection_only",
        "rule_based": decision_mode == "rule",
        "decision_mode": decision_mode,
        "trainer_engine": "RuleDrivenHarnessTrainer",
    }
    summary = {
        "schema_version": "harness.run.v1",
        "task": task,
        "run_id": output.name,
        "status": "failed",
        "started_at": utc_now(),
        "configuration": configuration,
        "source": {},
        "baseline": {},
        "rounds": [],
        "selected": {},
        "diagnosis_final": {},
        "warnings": WARNINGS.copy(),
    }
    event("run_started", {"task": task, "configuration": configuration})
    try:
        bundle = prepare_bundle(archive, task, output / "source")
        summary["source"] = bundle.source
        summary["source"]["harness_code_sha256"] = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        parent = bundle.checkpoint
        model = bundle.model_class.load(parent)
        diagnosis = diagnose(model, bundle.train, bundle.validation, task)
        metrics = diagnosis["validation_metrics"]
        summary["baseline"] = {
            "checkpoint": str(parent),
            "checkpoint_sha256": digest(parent),
            "metrics": metrics,
            "diagnosis": diagnosis,
        }
        write_json(output / "diagnosis-before.json", diagnosis)
        event("baseline_evaluated", {"checkpoint_sha256": digest(parent), "metrics": metrics})
        exhausted = False
        for round_index in range(1, rounds + 1):
            round_dir = output / ("round-%02d" % round_index)
            round_dir.mkdir()
            strategies = choose_strategies(
                task,
                model,
                diagnosis,
                epochs,
                max_candidates,
                decision_mode=decision_mode,
                strategy_model=strategy_model,
            )
            state_vec = diagnosis_to_state_vector(task, model, diagnosis)
            record = {
                "round": round_index,
                "parent_checkpoint_sha256": digest(parent),
                "diagnosis": diagnosis,
                "strategies": [s.to_dict() for s in strategies],
                "candidates": [],
            }
            summary["rounds"].append(record)
            write_json(
                round_dir / "plan.json",
                {
                    "parent_checkpoint_sha256": digest(parent),
                    "diagnosis": diagnosis,
                    "strategies": record["strategies"],
                },
            )
            event(
                "round_planned",
                {
                    "round": round_index,
                    "parent_checkpoint_sha256": digest(parent),
                    "strategy_ids": [s.id for s in strategies],
                },
            )
            for index, strategy in enumerate(strategies, 1):
                if ledger.remaining <= 0:
                    exhausted = True
                    event("training_budget_exhausted", {"round": round_index, "unstarted_strategy": strategy.id})
                    break
                candidate_dir = round_dir / ("%02d-%s" % (index, strategy.id))
                event(
                    "candidate_started",
                    {
                        "round": round_index,
                        "strategy": strategy.to_dict(),
                        "parent_checkpoint_sha256": digest(parent),
                    },
                )
                candidate = run_candidate(
                    bundle,
                    parent,
                    strategy,
                    candidate_dir,
                    ledger,
                    event,
                    patience,
                    trainer=trainer,
                )
                candidate["artifact_directory"] = str(candidate_dir)
                record["candidates"].append(candidate)
                if candidate["status"] == "budget_exhausted":
                    exhausted = True
                    break
            winner = select_candidate(task, metrics, record["candidates"], min_improvement)
            before = metrics
            if winner:
                parent = Path(winner["best_checkpoint"])
                metrics = winner["metrics"]
                model = bundle.model_class.load(parent)
            record["decision"] = {
                "accepted": winner is not None,
                "winner_id": winner["id"] if winner else None,
                "reason": "采用本轮主指标最优且超过阈值的候选" if winner else "无候选满足门控；保留本轮父检查点",
                "metrics_before": before,
                "metrics_after": metrics,
                "selected_checkpoint_sha256": digest(parent),
            }
            for candidate in record["candidates"]:
                candidate["gate"]["accepted"] = winner is candidate
                write_json(Path(candidate["artifact_directory"]) / "candidate.json", candidate)
                gain_val = candidate["gate"].get("improvement")
                experience_buffer.add(
                    TrajectoryRecord(
                        task=task,
                        round_index=round_index,
                        candidate_id=candidate["id"],
                        strategy_id=candidate["strategy"]["id"],
                        state_vector=[float(v) for v in state_vec.tolist()],
                        improvement=float(gain_val) if (gain_val is not None and math.isfinite(gain_val)) else 0.0,
                        accepted=bool(candidate["gate"]["accepted"]),
                        eligible=bool(candidate["gate"].get("eligible", False)),
                    )
                )
            write_json(round_dir / "round.json", record)
            event("round_decision", {"round": round_index, **record["decision"]})
            if exhausted or winner is None:
                break
            if round_index < rounds:
                diagnosis = diagnose(model, bundle.train, bundle.validation, task)
        selected_dir = output / "selected"
        selected_dir.mkdir()
        selected = selected_dir / "checkpoint.npz"
        shutil.copyfile(parent, selected)
        # Reload and independently re-evaluate the exact delivered file, never trust reported metrics alone.
        reloaded = bundle.model_class.load(selected)
        actual = ledger.evaluate(reloaded, bundle.validation)
        if abs(improvement(task, metrics, actual)) > 1e-12:
            raise ValueError("saved selected checkpoint metrics do not match gate decision")
        original = bundle.model_class.load(bundle.checkpoint)
        import numpy as np
        if not np.array_equal(reloaded.mean, original.mean) or not np.array_equal(reloaded.scale, original.scale):
            raise ValueError("selected checkpoint changed original normalization")
        if (reloaded.meta["target_mean"], reloaded.meta["target_scale"]) != (original.meta["target_mean"], original.meta["target_scale"]):
            raise ValueError("selected checkpoint changed original target scaler")
        summary["selected"] = {
            "checkpoint": str(selected),
            "checkpoint_sha256": digest(selected),
            "metrics": actual,
            "selected_from": str(parent),
            "improvement": improvement(task, summary["baseline"]["metrics"], actual),
            "steps": reloaded.meta["steps"],
            "epochs": reloaded.meta["epochs"],
        }
        summary["diagnosis_final"] = diagnose(reloaded, bundle.train, bundle.validation, task)
        write_json(output / "diagnosis-after.json", summary["diagnosis_final"])
        experience_buffer.save_json(output / "experience_trajectories.json")
        attempted = [c for r in summary["rounds"] for c in r["candidates"]]
        errors = [c for c in attempted if c["status"] == "error"]
        if errors:
            summary["warnings"].append("有 %d 个候选执行错误，已禁止接受；请检查候选日志。" % len(errors))
        if attempted and len(errors) == len(attempted):
            summary["status"] = "failed"
            summary["error"] = "所有候选执行失败；最终检查点仅保留原起点。"
        else:
            summary["status"] = "budget_exhausted" if exhausted else "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = type(exc).__name__ + ": " + str(exc)
        event("run_failed", {"error": summary["error"]})
    summary["finished_at"] = utc_now()
    summary["cost"] = {
        "total_seconds": time.perf_counter() - started,
        "training_seconds": ledger.training_seconds,
        "validation_seconds": ledger.validation_seconds,
        "candidate_count": sum(len(r["candidates"]) for r in summary["rounds"]),
        "note": "validation_seconds counts candidate/final evaluation; diagnostic evaluation is included only in total_seconds",
    }
    from .reporting import write_report
    write_json(output / "summary.json", summary)
    write_report(summary, output)
    event("run_finished", {"status": summary["status"], "selected": summary["selected"], "cost": summary["cost"]})
    return summary
