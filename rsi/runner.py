"""Autonomous RSI runner.

For every task the loop is:

    read history -> drift diagnosis -> propose experiments -> critic review
    -> isolated execution -> deterministic gate -> promote/retain/rollback
    -> record (records.jsonl + dashboard state + optional SwanLab)

The runner is the only component allowed to advance the parent checkpoint. It
is deliberately deterministic around the executable boundary: an LLM can
propose, but parameters are validated against the catalog, training happens in
a child process, and promotion is decided by the frozen selection split.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rsi.agent.config import TokenBudget, load_llm_config
from rsi.agent.coordinator import Coordinator
from rsi.agent.runtime import PIRuntime
from rsi.data import Dataset, load_jsonl_dataset
from rsi.drift import analyse
from rsi.executor import ExperimentProposal, Executor
from rsi.inference import apply_postprocess, resolve_postprocess
from rsi.metrics import from_predictions, primary
from rsi.model import ExtendedModel, checkpoint_sha256, code_sha256
from rsi.planner import fallback_proposals
from rsi.records import Recorder, atomic_write_json, utcnow_iso
from rsi.selection import best_candidate, gate, relative_change
from rsi.splits import build_splits, sha256_file
from rsi.tasks import TaskConfig, resolve_tasks

HARNESS_ROOT = Path(__file__).resolve().parent.parent


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _compact(value, limit=600):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


@dataclass
class TaskState:
    config: TaskConfig
    splits: dict
    train: Dataset
    calibration: Dataset
    selection: Dataset
    report: Dataset
    probe: Dataset | None
    parent_path: Path
    parent_sha: str
    baseline_sha_actual: str
    postprocess: dict
    selection_metrics: dict
    report_metrics: dict
    probe_metrics: dict | None
    executor: Executor
    rounds: list[dict] = field(default_factory=list)
    panel: list[dict] = field(default_factory=list)
    status: str = "pending"
    stall_rounds: int = 0
    force_explore: bool = False
    best_primary: float = 0.0
    best_round: int = 0
    tokens_used: int = 0
    started_at: float = 0.0
    finished_at: float | None = None
    drift: dict = field(default_factory=dict)


class Runner:
    def __init__(self, keys=("A", "B", "C"), rounds: int = 10, llm_mode: str = "auto",
                 seed: int = 42, state_dir: str = ".scratch/rsi-runs",
                 wall_cap: float = 21600.0, cpu_cap: float = 36000.0,
                 experiments_cap: int = 40, swanlab: bool = False,
                 run_dir: Path | None = None, verbose: bool = True):
        self.configs = resolve_tasks(keys)
        self.rounds = int(rounds)
        self.seed = int(seed)
        self.state_dir = Path(state_dir)
        self.wall_cap = float(wall_cap)
        self.cpu_cap = float(cpu_cap)
        self.experiments_cap = int(experiments_cap)
        # Experiment budget is per task: each dataset gets a fair, bounded share
        # (the platform evaluates tasks independently); the displayed total is
        # the sum across tasks so the dashboard never reads "used > cap".
        self.experiments_cap_total = self.experiments_cap * max(1, len(self.configs))
        self.swanlab = bool(swanlab)
        self.verbose = verbose
        self.started_at = time.time()
        self.run_id = "rsi-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir) if run_dir else self.state_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = Recorder(self.run_dir, self.run_id)
        self.llm_config = load_llm_config(llm_mode)
        self.budget = TokenBudget(cap=self.llm_config.token_cap)
        self.runtime = PIRuntime(self.llm_config, self.budget)
        self.events: list[dict] = []
        self.tasks: dict[str, TaskState] = {}
        self.experiments_used = 0
        self.global_cpu = 0.0
        self.swan_run = None
        self._init_swanlab()

    # -- logging -----------------------------------------------------------
    def log(self, level: str, task: str, message: str, **extra) -> None:
        record = {"ts": utcnow_iso(), "level": level, "task": task, "message": message}
        self.events.append(record)
        if len(self.events) > 240:
            self.events = self.events[-240:]
        self.recorder.event(level, task, message, **extra)
        if self.verbose:
            print(f"[{record['ts']}] {level.upper():7s} {task}: {message}", flush=True)

    def _init_swanlab(self) -> None:
        if not self.swanlab:
            return
        try:
            import swanlab

            self.swan_run = swanlab.init(project=os.environ.get("SWANLAB_PROJECT", "zhixun-rsi"),
                                         mode="offline",
                                         log_dir=str(self.run_dir / "swanlab"))
            self.log("info", "-", "SwanLab offline run initialised")
        except Exception as exc:  # pragma: no cover - optional dependency
            self.log("warn", "-", f"SwanLab disabled: {exc}")

    def _swan_log(self, data: dict) -> None:
        if self.swan_run is None:
            return
        try:
            self.swan_run.log(data)
        except Exception:
            pass

    # -- evaluation --------------------------------------------------------
    def evaluate(self, model: ExtendedModel, dataset: Dataset | None,
                 postprocess: dict, config: TaskConfig) -> dict | None:
        if dataset is None or len(dataset) == 0:
            return None
        predicted = apply_postprocess(postprocess, model, dataset.x)
        values = from_predictions(config.task, predicted, dataset.y, config.classes)
        values["primary"] = primary(config.task, values)
        values["rows"] = int(len(dataset))
        return values

    # -- drift -------------------------------------------------------------
    def drift_report(self, state: TaskState, round_index: int) -> dict:
        model = ExtendedModel.from_checkpoint(state.parent_path)
        order = np.random.default_rng(self.seed + round_index).permutation(len(state.selection))
        window_rows = max(16, int(len(state.selection) * (round_index + 1) / self.rounds))
        index = np.sort(order[:window_rows])
        window = Dataset(f"stream-r{round_index}", [state.selection.ids[i] for i in index],
                         state.selection.x[index], state.selection.y[index],
                         state.selection.paths, state.selection.sha256,
                         [state.selection.groups[i] for i in index])
        report = analyse(model, state.train, window, state.config.classes)
        payload = report.to_dict()
        payload["window_rows"] = int(window_rows)
        payload["round_index"] = round_index + 1
        payload["reference"] = "train"
        return payload

    # -- records / state ---------------------------------------------------
    def _write_state(self, status: str = "running") -> None:
        state = {
            "schema_version": 1,
            "run": {
                "run_id": self.run_id,
                "status": status,
                "started_at": _iso_now() if not self.run_id else datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "elapsed_seconds": round(time.time() - self.started_at, 1),
                "llm": {"backend": self.runtime.backend if self.runtime.available() else "deterministic",
                        "model": self.llm_config.model,
                        "calls": self.budget.calls,
                        "tokens_in": self.budget.used_input,
                        "tokens_out": self.budget.used_output},
                "budget": {"tokens_used": self.budget.used, "tokens_cap": self.budget.cap,
                           "cpu_seconds": round(self.global_cpu, 1), "cpu_cap": self.cpu_cap,
                           "wall_seconds": round(time.time() - self.started_at, 1),
                           "wall_cap": self.wall_cap,
                           "experiments_used": self.experiments_used,
                           "experiments_cap": self.experiments_cap_total},
            },
            "tasks": [self._task_state(t) for t in self.tasks.values()],
            "events": self.events[-120:],
        }
        self.recorder.write_state(state)

    def _task_state(self, state: TaskState) -> dict:
        cfg = state.config
        current = state.selection_metrics.get("primary")
        return {
            "key": cfg.key, "name": cfg.name, "name_en": cfg.name_en, "task": cfg.task,
            "primary_metric": cfg.primary_metric, "direction": cfg.direction,
            "baseline": cfg.baseline_primary,
            "best": state.best_primary or None,
            "current": current,
            "status": state.status,
            "round_index": len(state.rounds),
            "rounds_total": self.rounds,
            "drift": {key: state.drift.get(key) for key in
                      ("severity", "score", "feature_shift", "ks_max", "causes", "evidence")},
            "budget": {"tokens_used": state.tokens_used, "tokens_cap": self.budget.cap,
                       "cpu_seconds": round(state.executor.cpu_seconds, 1),
                       "cpu_cap": self.cpu_cap,
                       "wall_seconds": round((state.finished_at or time.time()) - state.started_at, 1),
                       "wall_cap": self.wall_cap,
                       "experiments_used": state.executor.experiments,
                       "experiments_cap": self.experiments_cap},
            "rounds": state.rounds,
            "panel": state.panel,
        }

    # -- main --------------------------------------------------------------
    def run(self) -> dict:
        self.log("info", "-", f"run {self.run_id} started; tasks={[c.key for c in self.configs]} "
                              f"rounds={self.rounds} llm={'on' if self.runtime.available() else 'off'}")
        self.log("info", "-", f"llm config: {_compact(self.llm_config.masked())}")
        self._write_state("running")
        for config in self.configs:
            try:
                self.run_task(config)
            except Exception as exc:  # one task failure must not abort the run
                self.log("error", config.key, f"task failed: {exc.__class__.__name__}: {exc}")
                if config.key in self.tasks:
                    self.tasks[config.key].status = "failed"
            finally:
                self._write_state("running")
        summary = self.write_summary()
        self._write_state("done")
        self.runtime.close()
        self.log("info", "-", f"run finished: {_compact({'cpu': round(self.global_cpu, 1),
                                                          'tokens': self.budget.used})}")
        return summary

    # -- per task ----------------------------------------------------------
    def run_task(self, config: TaskConfig) -> None:
        key = config.key
        task_dir = self.run_dir / key
        self.log("info", key, "building deterministic splits (calibration/selection/report)")
        splits = build_splits(config, task_dir, seed=self.seed)
        train = splits["train"]
        calibration = splits["calibration"]
        selection = splits["selection"]
        report = splits["report"]
        probe = splits.get("probe")
        parent_path = Path(config.checkpoint_path)
        parent_sha = checkpoint_sha256(parent_path)
        baseline_sha = parent_sha
        model = ExtendedModel.from_checkpoint(parent_path)
        postprocess = {"type": "none"}
        sel_metrics = self.evaluate(model, selection, postprocess, config)
        rep_metrics = self.evaluate(model, report, postprocess, config)
        probe_metrics = self.evaluate(model, probe, postprocess, config)
        executor = Executor(key, config.task, config.classes, splits["paths"],
                            HARNESS_ROOT, task_dir, timeout_seconds=int(os.environ.get("RSI_EXPERIMENT_TIMEOUT", 1800)))
        state = TaskState(config=config, splits=splits, train=train, calibration=calibration,
                          selection=selection, report=report, probe=probe,
                          parent_path=parent_path, parent_sha=parent_sha,
                          baseline_sha_actual=baseline_sha, postprocess=postprocess,
                          selection_metrics=sel_metrics, report_metrics=rep_metrics,
                          probe_metrics=probe_metrics, executor=executor,
                          status="running", best_primary=sel_metrics["primary"],
                          started_at=time.time())
        self.tasks[key] = state
        self.log("info", key, f"baseline {config.primary_metric}={sel_metrics['primary']:.6f} "
                              f"(official target={config.baseline_primary:.6f}); "
                              f"rows train={len(train)} cal={len(calibration)} sel={len(selection)} rep={len(report)}"
                              + (f" probe={len(probe)}" if probe else ""))
        coordinator = Coordinator(config, self.runtime, self.recorder)
        token_start = self.budget.used

        for round_index in range(len(state.rounds), self.rounds):
            if time.time() - self.started_at > self.wall_cap * 0.95:
                state.status = "budget_stop"
                self.log("warn", key, "wall-clock budget nearly exhausted, stopping rounds")
                break
            if state.executor.experiments >= self.experiments_cap:
                state.status = "budget_stop"
                self.log("warn", key, "per-task experiment budget exhausted, stopping rounds")
                break
            self._run_round(state, coordinator, round_index)
        else:
            state.status = "done"
        if state.status == "running":
            state.status = "done"
        state.finished_at = time.time()
        state.tokens_used = self.budget.used - token_start
        self.log("info", key, f"task finished: {config.primary_metric}={state.selection_metrics['primary']:.6f} "
                              f"({len(state.rounds)} rounds, {state.executor.experiments} experiments)")
        # reporter closes the task record (one extra call, useful for the report)
        context = coordinator.build_context(len(state.rounds), self.rounds, state.drift,
                                            state.selection_metrics,
                                            self._history(state), self._budget_payload())
        summary_text = coordinator.report(context, {
            "fallback_summary": self._fallback_summary(state),
            "rounds": [r["hypothesis"] for r in state.rounds],
            "decision_counts": {d: sum(1 for r in state.rounds if r["decision"] == d)
                                for d in ("promote", "retain", "rollback")},
        })
        atomic_write_json(task_dir / "summary.json", {
            "task": key, "primary_metric": config.primary_metric,
            "baseline": config.baseline_primary, "final": state.selection_metrics["primary"],
            "report_metrics": state.report_metrics, "probe_metrics": state.probe_metrics,
            "rounds": state.rounds, "panel": state.panel, "narrative": summary_text,
            "llm": dict(coordinator_usage := {"tokens_used": state.tokens_used,
                                              "calls": self.budget.calls,
                                              "roles": coordinator.used_roles}),
            "checkpoint": {"path": str(state.parent_path), "sha256": state.parent_sha,
                           "baseline_sha256": baseline_sha},
            "postprocess": state.postprocess,
        })
        self._write_state("running")

    # -- one round ---------------------------------------------------------
    def _run_round(self, state: TaskState, coordinator: Coordinator, round_index: int) -> None:
        key = state.config.key
        round_number = round_index + 1
        started = time.time()
        token_before = self.budget.used
        parent_sha_before = state.parent_sha
        self.log("info", key, f"round {round_number}/{self.rounds}: diagnosing drift")
        drift = self.drift_report(state, round_index)
        state.drift = drift
        history = self._history(state)
        context = coordinator.build_context(round_number, self.rounds, drift,
                                            state.selection_metrics, history,
                                            self._budget_payload())
        self._write_state("running")

        diagnosis = coordinator.diagnose(context)
        self.log("info", key, f"round {round_number}: diagnosis={diagnosis['severity']} "
                              f"source={diagnosis['source']} causes={diagnosis['causes'][:2]}")
        context = coordinator.build_context(round_number, self.rounds, drift,
                                            state.selection_metrics, history,
                                            self._budget_payload())
        if state.force_explore:
            proposals = fallback_proposals(key, round_number + 5)[:2]
            state.force_explore = False
            self.log("info", key, "round: forced exploration proposal set after panel")
        else:
            proposals = coordinator.propose(context, diagnosis)
        proposals = proposals[:3]
        review = coordinator.criticize(context, proposals)
        approved = [proposals[i] for i in review["approvals"]][:2]
        if not approved:
            approved = proposals[:1]
        self.log("info", key, f"round {round_number}: {len(proposals)} proposals, "
                              f"{len(approved)} approved by {review['source']}")
        results = []
        for proposal in approved:
            if state.executor.experiments >= self.experiments_cap:
                break
            self.log("info", key, f"round {round_number}: executing {proposal.strategy} "
                                  f"{_compact(proposal.params, 160)} pp={_compact(proposal.postprocess, 120)}")
            result = state.executor.run(state.parent_path, proposal,
                                        calibration=state.splits["paths"]["calibration"],
                                        selection=state.splits["paths"]["selection"],
                                        report=state.splits["paths"]["report"],
                                        train=state.splits["paths"]["train"],
                                        drift={k: drift.get(k) for k in ("severity", "score", "causes")},
                                        drift_probe=state.splits["paths"].get("drift_probe"),
                                        seed=self.seed + round_number)
            self.experiments_used += 1
            self.global_cpu += result.cpu_seconds
            results.append(result.to_dict())
            status = "ok" if result.ok else f"failed: {result.error}"
            self.log("info" if result.ok else "warn", key,
                     f"round {round_number}: {result.experiment_id} {status}"
                     + (f" selection_primary={result.metrics['selection']['primary']:.6f}"
                        if result.ok and result.metrics.get("selection") else ""))
        if not results:
            self.log("warn", key, f"round {round_number}: no experiment executed")
            self._record_round(state, round_number, drift, diagnosis, proposals, review,
                               results, {"decision": "retain", "reason": "no experiment executed",
                                         "delta": 0.0}, parent_sha_before, token_before, started)
            return

        best = best_candidate(results, state.config.direction)
        if best is None:
            self.log("warn", key, f"round {round_number}: all candidates failed")
            self._record_round(state, round_number, drift, diagnosis, proposals, review,
                               results, {"decision": "rollback", "reason": "all candidates failed",
                                         "delta": 0.0}, parent_sha_before, token_before, started)
            return

        candidate_primary = best["metrics"]["selection"]["primary"]
        parent_primary = state.selection_metrics["primary"]
        decision = gate(state.config.task, parent_primary, candidate_primary, state.config.margin)
        # boundary arbitration: only when two candidates are within 2 margins
        ok_results = [r for r in results if r.get("ok") and r.get("metrics", {}).get("selection")]
        if len(ok_results) >= 2:
            primaries = sorted(r["metrics"]["selection"]["primary"] for r in ok_results)
            spread = abs(primaries[-1] - primaries[0])
            if spread <= 2 * state.config.margin:
                verdict = coordinator.arbitrate(context, approved, ok_results)
                if verdict.get("source") == "llm":
                    if verdict.get("decision") == "rollback" and decision.decision == "promote":
                        decision = type(decision)("retain", "arbiter downgraded a marginal promote", decision.delta)
                    elif verdict.get("decision") == "retain" and decision.decision == "promote":
                        decision = type(decision)("retain", "arbiter asked for retain at boundary", decision.delta)
                    self.log("info", key, f"round {round_number}: arbiter decision={verdict.get('decision')} "
                                          f"-> gate={decision.decision}")
        self.log("info", key, f"round {round_number}: gate={decision.decision} delta={decision.delta:+.6f} "
                              f"({decision.reason})")
        if decision.decision == "promote":
            state.parent_path = Path(best["candidate_checkpoint"])
            state.parent_sha = best["checkpoint_sha256"]
            state.postprocess = best["postprocess"]
            model = ExtendedModel.from_checkpoint(state.parent_path)
            state.selection_metrics = self.evaluate(model, state.selection, state.postprocess, state.config)
            state.report_metrics = self.evaluate(model, state.report, state.postprocess, state.config)
            if state.probe is not None:
                state.probe_metrics = self.evaluate(model, state.probe, state.postprocess, state.config)
            state.best_primary = state.selection_metrics["primary"]
            state.best_round = round_number
            state.stall_rounds = 0
            self.log("info", key, f"round {round_number}: promoted -> {state.parent_sha[:22]}... "
                                  f"{state.config.primary_metric}={state.best_primary:.6f}")
        else:
            state.stall_rounds += 1
        record = self._record_round(state, round_number, drift, diagnosis, proposals, review,
                                    results, decision.to_dict(), parent_sha_before, token_before,
                                    started, best=best)
        self._swan_log({"task": key, "round": round_number,
                        "primary": state.selection_metrics["primary"],
                        "baseline": state.config.baseline_primary,
                        "decision": decision.decision})
        if state.stall_rounds >= 2 and len(state.panel) < 2 and self.budget.remaining() > 20000:
            issue = (f"连续 {state.stall_rounds} 轮未提升（当前 {state.config.primary_metric}="
                     f"{state.selection_metrics['primary']:.6f}，基线 {state.config.baseline_primary:.6f}，"
                     f"漂移={drift.get('severity')}）")
            self.log("warn", key, f"round {round_number}: stall detected, convening panel")
            entries = coordinator.panel(context, issue)
            verdict = "retain"
            for entry in entries:
                if entry.get("role") == "arbiter" and entry.get("source") == "llm":
                    verdict = "retain"
            state.panel.append({"trigger": issue, "round_index": round_number,
                                "verdict": verdict,
                                "messages": [{k: e.get(k) for k in
                                              ("role", "standpoint", "evidence", "confidence", "source")}
                                             for e in entries]})
            state.force_explore = True
            state.stall_rounds = 0
            self._write_state("running")

    # -- round record ------------------------------------------------------
    def _record_round(self, state: TaskState, round_number: int, drift: dict, diagnosis: dict,
                      proposals: list, review: dict, results: list, decision: dict,
                      parent_sha_before: str, token_before: int, started: float,
                      best: dict | None = None) -> dict:
        cfg = state.config
        metrics = dict(state.selection_metrics)
        metrics["primary"] = state.selection_metrics["primary"]
        history_ids = [f"{cfg.key}-r{r['index']}" for r in state.rounds[-3:]]
        strategy = best["proposal"]["strategy"] if best else (proposals[0].strategy if proposals else "retain")
        postprocess = best["postprocess"] if best else state.postprocess
        selected_experiment = best["experiment_id"] if best else None
        change_summary = self._change_summary(cfg, best, decision, state)
        record = {
            "index": round_number,
            "id": f"{cfg.key}-r{round_number}",
            "task": cfg.key,
            "status": "done",
            "decision": decision["decision"],
            "decision_reason": decision.get("reason", ""),
            "delta": decision.get("delta", 0.0),
            "hypothesis": (proposals[0].hypothesis if proposals else ""),
            "change_summary": change_summary,
            "strategy": strategy,
            "postprocess": postprocess.get("type", "none"),
            "postprocess_spec": postprocess,
            "metrics": {k: metrics[k] for k in
                        ("macro_f1", "accuracy", "rmse", "mae", "nasa_score", "primary", "rows")
                        if k in metrics},
            "validation_metrics": {cfg.primary_metric: metrics["primary"]},
            "parent_checkpoint_sha256": parent_sha_before,
            "candidate_checkpoint_sha256": best["checkpoint_sha256"] if best else None,
            "candidate_code_sha256": best["code_sha256"] if best else None,
            "checkpoint_sha256": state.parent_sha,
            "code_sha256": code_sha256(HARNESS_ROOT),
            "history_ids": history_ids,
            "diagnosis": diagnosis,
            "drift": {k: drift.get(k) for k in ("severity", "score", "feature_shift", "ks_max", "causes")},
            "approved_plans": [p.__dict__ for p in proposals],
            "review": review,
            "experiments": results,
            "chosen_experiment": selected_experiment,
            "cpu_seconds": round(sum(float(r.get("cpu_seconds", 0.0)) for r in results), 3),
            "wall_seconds": round(time.time() - started, 3),
            "llm": {"tokens_used": self.budget.used - token_before,
                    "calls": self.budget.calls, "roles": list(coordinator_roles(proposals))},
            "created_at": utcnow_iso(),
        }
        state.rounds.append(record)
        self.recorder.round(record)
        self._write_state("running")
        return record

    def _change_summary(self, cfg: TaskConfig, best: dict | None, decision: dict,
                        state: TaskState) -> str:
        if best is None:
            return "本轮无可执行实验（候选全部失败或被拒绝），保留父模型。"
        proposal = best["proposal"]
        parts = [f"{proposal['strategy']}",
                 f"postprocess={best['postprocess'].get('type', 'none')}",
                 f"{cfg.primary_metric}={best['metrics']['selection']['primary']:.6f}",
                 f"决策={decision['decision']}"]
        if proposal.get("hypothesis"):
            parts.append(proposal["hypothesis"][:120])
        return " | ".join(parts)

    def _fallback_summary(self, state: TaskState) -> str:
        cfg = state.config
        promotions = [r for r in state.rounds if r["decision"] == "promote"]
        return (f"{cfg.key} 任务共 {len(state.rounds)} 轮，{len(promotions)} 次提升；"
                f"{cfg.primary_metric} {cfg.baseline_primary:.6f} -> "
                f"{state.selection_metrics['primary']:.6f}；"
                f"当前检查点 {state.parent_sha[:20]}...。")

    # -- helpers -----------------------------------------------------------
    def _history(self, state: TaskState) -> list[dict]:
        return [{"index": r["index"], "decision": r["decision"], "strategy": r["strategy"],
                 "hypothesis": r["hypothesis"][:160],
                 "change_summary": r["change_summary"][:200],
                 "metrics": r["validation_metrics"],
                 "delta": r.get("delta", 0.0)} for r in state.rounds[-5:]]

    def _budget_payload(self) -> dict:
        return {"tokens_used": self.budget.used, "tokens_cap": self.budget.cap,
                "remaining": self.budget.remaining(),
                "experiments_used": self.experiments_used,
                "experiments_cap": self.experiments_cap_total,
                "wall_seconds": round(time.time() - self.started_at, 1),
                "wall_cap": self.wall_cap}

    # -- summary -----------------------------------------------------------
    def write_summary(self) -> dict:
        datasets = [self._workspace_report(state) for state in self.tasks.values()]
        summary = {
            "run_id": self.run_id,
            "created_at": _iso_now(),
            "tasks": [c.key for c in self.configs],
            "llm": {"backend": self.runtime.backend if self.runtime.available() else "deterministic",
                    "available": self.runtime.available(),
                    "model": self.llm_config.model, "base_url": self.llm_config.base_url,
                    "budget": self.budget.to_dict()},
            "budget": {"cpu_seconds": round(self.global_cpu, 2), "wall_seconds": round(time.time() - self.started_at, 2),
                       "experiments_used": self.experiments_used,
                       "experiments_cap": self.experiments_cap_total},
            "datasets": datasets,
            "artifacts": {"state": str(self.recorder.state_path),
                          "records": str(self.recorder.records_path),
                          "rounds": str(self.recorder.rounds_path)},
        }
        atomic_write_json(self.run_dir / "summary.json", summary)
        atomic_write_json(self.run_dir / "workspace_report.json", {"datasets": datasets})
        if self.swan_run is not None:
            try:
                self.swan_run.finish()
            except Exception:
                pass
        return summary

    def _workspace_report(self, state: TaskState) -> dict:
        cfg = state.config
        code_now = code_sha256(HARNESS_ROOT)
        artifact_sha = "sha256:" + hashlib.sha256(json.dumps({
            "predict_sha256": sha256_file(HARNESS_ROOT / "rsi" / "predict.py")
            if (HARNESS_ROOT / "rsi" / "predict.py").is_file() else None,
            "checkpoint": state.parent_sha,
            "postprocess": state.postprocess,
        }, sort_keys=True).encode()).hexdigest()
        return {
            "key": cfg.key,
            "record_complete": len(state.rounds) >= self.rounds,
            "rounds": state.rounds,
            "prediction_count": sum(int((r.get("metrics") or {}).get("rows", 0)) for r in state.rounds[-1:])
                                if state.rounds else int((state.selection_metrics or {}).get("rows", 0)),
            "cpu_accounting_status": "measured",
            "cpu_seconds": round(state.executor.cpu_seconds, 2),
            "wall_seconds": round((state.finished_at or time.time()) - state.started_at, 2),
            "metrics": {**state.selection_metrics, "report": state.report_metrics,
                        "probe": state.probe_metrics},
            "baseline_checkpoint_sha256": cfg.checkpoint_path and state.baseline_sha_actual,
            "baseline_code_sha256": code_now,
            "checkpoint_sha256": state.parent_sha,
            "code_sha256": code_now,
            "artifact_sha256": artifact_sha,
            "evaluation_config_id": os.environ.get("RSI_EVALUATION_CONFIG_ID", "local-dev"),
            "evaluation_config_digest": os.environ.get("RSI_EVALUATION_CONFIG_DIGEST", "local-dev"),
            "postprocess": state.postprocess,
            "panel": state.panel,
        }


def coordinator_roles(proposals) -> list[str]:
    return ["diagnostician", "experimenter", "critic"]
