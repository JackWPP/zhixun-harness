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
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rsi.agent.config import TokenBudget, load_llm_config
from rsi.agent.coordinator import Coordinator
from rsi.agent.goals import GoalLoop
from rsi.agent.guard import Guard
from rsi.agent.kernel import Kernel
from rsi.agent.runtime import PIRuntime
from rsi.data import Dataset, load_jsonl_dataset
from rsi.drift import analyse_covariates
from rsi.executor import ExperimentProposal, Executor
from rsi.inference import apply_postprocess, predict_for_split, resolve_postprocess
from rsi.memory import (MemoryReviewer, MemoryStore, Recall, SkillCurator,
                        SkillStore, StateDB)
from rsi.memory.skill_candidates import SkillCandidateStore
from rsi.metrics import from_predictions, primary
from rsi.model import ExtendedModel, checkpoint_sha256, code_sha256
from rsi.planner import (fallback_proposals, postprocess_catalog_text,
                         strategy_catalog_text)
from rsi.records import Recorder, atomic_write_json, utcnow_iso
from rsi.selection import best_candidate, gate, relative_change
from rsi.splits import build_splits, sha256_file
from rsi.tasks import TaskConfig, resolve_tasks
from rsi.agent.views import agent_view

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
    controller: str = "pipeline"
    goal: dict = field(default_factory=dict)
    agent_stats: dict = field(default_factory=dict)
    memory_info: dict = field(default_factory=dict)
    extended: Dataset | None = None
    extended_path: Path | None = None
    validation: Dataset | None = None
    degraded: bool = False


class KernelHost:
    """Supervisor callbacks exposed to the autonomous kernel.

    The kernel can *act*, but the supervisor remains the only writer of the
    parent checkpoint, the experiment counter and the round record.
    """

    def __init__(self, runner: "Runner", state: TaskState, drift: dict, context: dict):
        self.runner = runner
        self.state = state
        self.drift = drift
        self._context = context

    def context(self) -> dict:
        return self._context

    def parent_primary(self) -> float:
        return float(self.state.selection_metrics["primary"])

    def run_experiment(self, proposal: ExperimentProposal) -> dict:
        return self.runner._execute_proposal(self.state, proposal, self.drift)

    def record_panel(self, entries: list[dict]) -> None:
        # KernelTool panels are counted through KernelEnv.summary(); avoid double count.
        self.runner._record_panel(self.state, entries, count=False)



class Runner:
    def __init__(self, keys=("A", "B", "C"), rounds: int = 10, llm_mode: str = "auto",
                 seed: int = 42, state_dir: str = ".scratch/rsi-runs",
                 wall_cap: float = 21600.0, cpu_cap: float = 36000.0,
                 experiments_cap: int = 40, swanlab: bool = False,
                 run_dir: Path | None = None, verbose: bool = True,
                 controller: str = "auto", memory: bool = True,
                 memory_dir: str | None = None, db_path: str | None = None,
                 round_turns: int = 20, judge: bool = True):
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
        self.round_turns = int(round_turns)
        self.judge_enabled = bool(judge)
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
        self.swanlab_error: str | None = None
        # -- controller selection (kernel needs a live LLM) ----------------
        if controller not in ("auto", "kernel", "pipeline"):
            raise ValueError(f"unknown controller {controller!r}")
        self.requested_controller = controller
        self.llm_ready = self.runtime.available()
        if controller == "auto":
            self.controller = "kernel" if self.llm_ready else "pipeline"
        elif controller == "kernel" and not self.llm_ready:
            self.controller = "pipeline"
        else:
            self.controller = controller
        # -- persistent memory / records (optional, stdlib only) -----------
        self.memory_enabled = bool(memory)
        # Memory is persistent across runs by default (Hermes-style MEMORY.md +
        # TASK_<X>.md snapshot); use --memory-dir to isolate an ablation run.
        self.memory_root = Path(memory_dir or os.environ.get("RSI_MEMORY_DIR")
                                or (self.state_dir / "memory"))
        self.db_path = Path(db_path or os.environ.get("RSI_DB_PATH")
                            or (self.state_dir / "rsi_state.db"))
        self.memory = MemoryStore(self.memory_root) if self.memory_enabled else None
        self.skills = self._build_skill_store() if self.memory_enabled else None
        self.skill_candidates = (SkillCandidateStore(
            self.skills, self.run_dir / "skill-candidates",
            max_candidates=int(os.environ.get("RSI_SKILL_CANDIDATE_CAP", 3)))
            if self.skills is not None else None)
        self.db = StateDB(self.db_path) if self.memory_enabled else None
        if self.db is not None and hasattr(self.recorder, "attach_db"):
            self.recorder.attach_db(self.db)
        self.recall = Recall(self.db, self.memory, self.skills) if self.memory_enabled else None
        self.reviewer = (MemoryReviewer(self.memory, self.db, self.skills, self.runtime, self.run_id)
                         if self.memory_enabled else None)
        self.curator = SkillCurator(self.skills, self.db) if self.memory_enabled else None
        self.agent_stats = {"kernel_rounds": 0, "kernel_failures": 0, "pipeline_rounds": 0,
                            "delegates": 0, "panels": 0, "memory_writes": 0,
                            "skill_writes": 0, "kernel_tokens": 0}
        self._init_swanlab()

    def _build_skill_store(self):
        """Give each persistent run state its own writable skill library.

        Seed skills are copied from the repository once; agent-created or
        patched skills therefore cannot mutate the submitted source tree.
        """
        seed_root = HARNESS_ROOT / "skills"
        writable_root = self.memory_root / "skills"
        writable_root.mkdir(parents=True, exist_ok=True)
        for name in SkillStore(seed_root).names():
            source = seed_root / name / "SKILL.md"
            target = writable_root / name / "SKILL.md"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        return SkillStore(writable_root, on_use=self._note_skill_use)

    # -- memory helpers ----------------------------------------------------
    def _note_skill_use(self, name: str) -> None:
        if self.db is not None:
            try:
                self.db.record_skill_use(name, self.run_id)
            except Exception:
                pass


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
                                         mode=os.environ.get("SWANLAB_MODE", "offline"),
                                         log_dir=str(self.run_dir / "swanlab"))
            self.log("info", "-", f"SwanLab {os.environ.get('SWANLAB_MODE', 'offline')} run initialised")
        except Exception as exc:  # pragma: no cover - optional dependency
            self.swanlab_error = f"{exc.__class__.__name__}: {exc}"
            self.log("warn", "-", f"SwanLab disabled: {exc}")

    def _swan_log(self, data: dict) -> None:
        if self.swan_run is None:
            return
        try:
            self.swan_run.log(data)
        except Exception:
            pass

    def _swan_metrics(self, state: TaskState, round_number: int, decision: dict,
                      results: list[dict]) -> dict:
        """Log model lifecycle evidence, not only the driver-agent's outcome."""
        candidates = []
        for result in results:
            metrics = result.get("metrics") or {}
            candidates.append({
                "experiment_id": result.get("experiment_id"),
                "proposal": result.get("proposal"),
                "parent_checkpoint_sha256": result.get("parent_checkpoint"),
                "candidate_checkpoint_sha256": result.get("checkpoint_sha256"),
                "code_sha256": result.get("code_sha256"),
                "selection": metrics.get("selection"),
                "report": metrics.get("report"),
                "probe": metrics.get("probe"),
                "cpu_seconds": result.get("cpu_seconds"),
                "wall_seconds": result.get("wall_seconds"),
                "error": result.get("error"),
            })
        return {
            "task": state.config.key,
            "round": round_number,
            "decision": decision.get("decision"),
            "decision_reason": decision.get("reason"),
            "champion_checkpoint_sha256": state.parent_sha,
            "postprocess": state.postprocess,
            "metrics": {"selection": state.selection_metrics,
                        "report": state.report_metrics,
                        "probe": state.probe_metrics},
            "candidates": candidates,
        }

    # -- evaluation --------------------------------------------------------
    def evaluate(self, model: ExtendedModel, dataset: Dataset | None,
                 postprocess: dict, config: TaskConfig,
                 full: Dataset | None = None) -> dict | None:
        if dataset is None or len(dataset) == 0:
            return None
        if full is not None and (postprocess or {}).get("type") == "damage_shrink":
            predicted = predict_for_split(postprocess, model, full, dataset)
        else:
            predicted = apply_postprocess(postprocess, model, dataset.x)
        values = from_predictions(config.task, predicted, dataset.y, config.classes)
        values["primary"] = primary(config.task, values)
        values["rows"] = int(len(dataset))
        return values

    # -- drift -------------------------------------------------------------
    def drift_report(self, state: TaskState, round_index: int) -> dict:
        # Preserve public validation row order and keep this view label blind.
        # Labels remain available only to the separate evaluator/selection gate.
        n = len(state.selection)
        start = min(n - 1, int(n * round_index / max(1, self.rounds)))
        end = min(n, max(start + 1, int(n * (round_index + 1) / max(1, self.rounds))))
        target_x = state.selection.x[start:end]
        payload = analyse_covariates(state.train.x, target_x)
        payload["window_rows"] = int(end - start)
        payload["window_start"] = int(start)
        payload["window_end"] = int(end)
        payload["round_index"] = round_index + 1
        payload["reference"] = "train"
        return payload

    # -- records / state ---------------------------------------------------
    def _write_state(self, status: str = "running") -> None:
        memory_state = {"enabled": self.memory_enabled,
                        "dir": str(self.memory_root) if self.memory_enabled else None,
                        "db": str(self.db_path) if self.db is not None else None}
        if self.memory_enabled and self.memory is not None:
            memory_state["entries"] = len(self.memory.entries())
            memory_state["usage"] = self.memory.usage()
        state = {
            "schema_version": 2,
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
                "agent": {"controller": self.controller, **self.agent_stats},
                "memory": memory_state,
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
        strategy_stats = []
        if self.db is not None:
            try:
                strategy_stats = self.db.strategy_stats(cfg.key)
            except Exception:
                strategy_stats = []
        return {
            "key": cfg.key, "name": cfg.name, "name_en": cfg.name_en, "task": cfg.task,
            "primary_metric": cfg.primary_metric, "direction": cfg.direction,
            "baseline": cfg.baseline_primary,
            "best": state.best_primary or None,
            "current": current,
            "status": state.status,
            "round_index": len(state.rounds),
            "rounds_total": self.rounds,
            "controller": state.controller,
            "degraded": state.degraded,
            "goal": state.goal,
            "agent": state.agent_stats,
            "memory": state.memory_info,
            "checkpoint_path": str(state.parent_path),
            "checkpoint_sha256": state.parent_sha,
            "postprocess": state.postprocess,
            "strategy_stats": strategy_stats,
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
        self.log("info", "-", f"controller={self.controller} memory={'on' if self.memory_enabled else 'off'} "
                              f"memory_dir={self.memory_root if self.memory else '-'} db={self.db_path if self.db else '-'}")
        if self.db is not None:
            try:
                self.db.add_run(self.run_id, utcnow_iso(), controller=self.controller,
                                llm_model=self.llm_config.model if self.llm_ready else "",
                                tasks=",".join(c.key for c in self.configs), seed=self.seed)
            except Exception as exc:
                self.log("warn", "-", f"sqlite index unavailable: {exc}")
        self._write_state("running")
        try:
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
            if self.db is not None:
                try:
                    self.db.finish_run(self.run_id, "done",
                                       json.dumps({"tasks": [c.key for c in self.configs]}))
                except Exception:
                    pass
        finally:
            self._write_state("done")
            self.runtime.close()
            if self.db is not None:
                self.db.close()
        run_totals = {"cpu": round(self.global_cpu, 1), "tokens": self.budget.used}
        self.log("info", "-", f"run finished: {_compact(run_totals)}")
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
        full_validation = splits.get("validation")
        parent_path = Path(config.checkpoint_path)
        parent_sha = checkpoint_sha256(parent_path)
        baseline_sha = parent_sha
        model = ExtendedModel.from_checkpoint(parent_path)
        postprocess = {"type": "none"}
        sel_metrics = self.evaluate(model, selection, postprocess, config, full=full_validation)
        rep_metrics = self.evaluate(model, report, postprocess, config, full=full_validation)
        probe_metrics = self.evaluate(model, probe, postprocess, config)
        executor = Executor(key, config.task, config.classes, splits["paths"],
                            HARNESS_ROOT, task_dir, timeout_seconds=int(os.environ.get("RSI_EXPERIMENT_TIMEOUT", 1800)))
        state = TaskState(config=config, splits=splits, train=train, validation=full_validation,
                          calibration=calibration,
                          selection=selection, report=report, probe=probe,
                          extended=splits.get("extended"), extended_path=splits.get("extended_path"),
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
        # -- autonomous controller setup -----------------------------------
        state.controller = self.controller
        memory_snapshot = ""
        if self.recall is not None:
            try:
                memory_snapshot = self.recall.memory_snapshot(key)
            except Exception as exc:
                self.log("warn", key, f"memory snapshot failed: {exc}")
        state.memory_info = {"enabled": self.memory_enabled,
                             "entries": len(self.memory.entries()) if self.memory else 0,
                             "snapshot_chars": len(memory_snapshot)}
        goal_loop = GoalLoop(task_key=key, run_id=self.run_id,
                             db=self.db if self.memory_enabled else None,
                             rounds_target=self.rounds)
        if self.judge_enabled:
            goal_loop.start()
        guard = Guard(config,
                      max_experiments_per_round=int(os.environ.get("RSI_EXPERIMENTS_PER_ROUND", 3)),
                      max_experiments_per_task=self.experiments_cap)
        kernel = None
        if state.controller == "kernel" and self.llm_ready:
            kernel = Kernel(config, self.runtime, guard, memory=self.memory,
                            skills=self.skills, skill_candidates=self.skill_candidates,
                            recall=self.recall,
                            recorder=self.recorder, run_id=self.run_id,
                            max_turns=self.round_turns,
                            round_timeout=float(os.environ.get("RSI_ROUND_TIMEOUT", 1800)),
                            experiment_timeout=float(os.environ.get("RSI_EXPERIMENT_TIMEOUT", 900)),
                            verbose=self.verbose)
            self.log("info", key, f"autonomous kernel enabled (max_turns={self.round_turns}, "
                                  f"controller={state.controller})")
        kernel_failures = 0

        for round_index in range(len(state.rounds), self.rounds):
            remaining = self.wall_cap - (time.time() - self.started_at)
            # Time-aware degradation: when the wall budget is tight, shrink the
            # search space instead of overrunning the platform limit.  The run is
            # marked degraded so the report can explain the reduced plan.
            if remaining < 1500 and self.rounds - len(state.rounds) > 0:
                state.degraded = True
                guard.max_experiments_per_round = 1
                if kernel is not None:
                    kernel.max_turns = min(kernel.max_turns, 8)
                    kernel.round_timeout = min(kernel.round_timeout, max(300.0, remaining - 300.0))
                self.log("warn", key, f"wall budget tight ({remaining:.0f}s left), degraded search: "
                                     f"1 experiment/round, kernel turns<=8")
            if time.time() - self.started_at > self.wall_cap * 0.95:
                state.status = "budget_stop"
                self.log("warn", key, "wall-clock budget nearly exhausted, stopping rounds")
                break
            if state.executor.experiments >= self.experiments_cap:
                state.status = "budget_stop"
                self.log("warn", key, "per-task experiment budget exhausted, stopping rounds")
                break
            handled = False
            if kernel is not None:
                outcome = self._run_kernel_round(state, kernel, coordinator, round_index,
                                                goal_loop, memory_snapshot)
                if outcome is not None:
                    handled = True
                    if outcome.ok:
                        kernel_failures = 0
                    else:
                        kernel_failures += 1
                        if kernel_failures >= 2:
                            self.log("warn", key, "kernel failed twice; task switches to pipeline fallback")
                            kernel = None
                            state.controller = "pipeline"
            if not handled:
                self._run_round(state, coordinator, round_index)
                self.agent_stats["pipeline_rounds"] += 1
                state.agent_stats = dict(self.agent_stats)
            # -- goal loop judge (deterministic, fail-open) ----------------
            if self.judge_enabled and goal_loop.state and state.rounds:
                try:
                    last = state.rounds[-1]
                    done, reason = goal_loop.judge(
                        decision=last.get("decision"), round_record=last,
                        budget_remaining=self.budget.remaining(),
                        wall_remaining=self.wall_cap - (time.time() - self.started_at),
                        degraded=(state.status == "budget_stop"))
                    state.goal = dict(goal_loop.state.to_dict(),
                                      judge={"done": done, "reason": reason})
                except Exception as exc:
                    state.goal = {"error": f"{exc.__class__.__name__}: {exc}"}
        else:
            state.status = "done"
        if state.status == "running":
            state.status = "done"
        state.finished_at = time.time()
        state.tokens_used = self.budget.used - token_start
        self.log("info", key, f"task finished: {config.primary_metric}={state.selection_metrics['primary']:.6f} "
                              f"({len(state.rounds)} rounds, {state.executor.experiments} experiments, "
                              f"controller={state.controller})")
        # -- close the task: distill durable memory, curate skills --------
        review_info: dict = {}
        if self.reviewer is not None:
            try:
                review_info = self.reviewer.review(key, agent_view(state.rounds))
            except Exception as exc:
                review_info = {"error": f"{exc.__class__.__name__}: {exc}"}
        curated: list = []
        if self.curator is not None:
            try:
                curated = self.curator.curate()
            except Exception as exc:
                curated = [{"error": f"{exc.__class__.__name__}: {exc}"}]
        state.memory_info.update({
            "review": review_info, "curated": curated,
            "entries_after": len(self.memory.entries()) if self.memory else 0})

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
            "controller": state.controller, "agent": state.agent_stats,
            "goal": state.goal, "memory": state.memory_info,
            "llm": dict(coordinator_usage := {"tokens_used": state.tokens_used,
                                              "calls": self.budget.calls,
                                              "roles": coordinator.used_roles}),
            "checkpoint": {"path": str(state.parent_path), "sha256": state.parent_sha,
                           "baseline_sha256": baseline_sha},
            "postprocess": state.postprocess,
        })
        self._write_state("running")

    def _strategy_stats(self, key: str) -> list[dict]:
        try:
            return self.db.strategy_stats(key) if self.db is not None else []
        except Exception:
            return []

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
        stats = self._strategy_stats(key)
        context = coordinator.build_context(round_number, self.rounds, drift,
                                            state.selection_metrics, history,
                                            self._budget_payload())
        context["stats"] = stats
        self._write_state("running")

        diagnosis = coordinator.diagnose(context)
        self.log("info", key, f"round {round_number}: diagnosis={diagnosis['severity']} "
                              f"source={diagnosis['source']} causes={diagnosis['causes'][:2]}")
        context = coordinator.build_context(round_number, self.rounds, drift,
                                            state.selection_metrics, history,
                                            self._budget_payload())
        context["stats"] = stats
        if state.force_explore:
            proposals = fallback_proposals(key, round_number + 5, stats=stats)[:2]
            state.force_explore = False
            self.log("info", key, "round: forced exploration proposal set after panel")
        else:
            proposals = coordinator.propose(context, diagnosis)
        from rsi.policy import deny_reason
        proposals = [p for p in proposals[:3]
                     if not ((p.params or {}).get("data")
                             and deny_reason(str(p.params["data"]), state.config))]
        if not proposals:
            proposals = [ExperimentProposal("retain", {}, {"type": "none"},
                                            hypothesis="policy-filtered fallback",
                                            expected_effect="keep parent",
                                            risk="no candidate")]
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
            results.append(self._execute_proposal(state, proposal, drift, round_number))
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
            self._apply_promotion(state, best)
            self.log("info", key, f"round {round_number}: promoted -> {state.parent_sha[:22]}... "
                                  f"{state.config.primary_metric}={state.best_primary:.6f}")
        else:
            state.stall_rounds += 1
        record = self._record_round(state, round_number, drift, diagnosis, proposals, review,
                                    results, decision.to_dict(), parent_sha_before, token_before,
                                    started, best=best)
        self._swan_log(self._swan_metrics(
            state, round_number, decision.to_dict(), results))
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

    # -- experiments (shared by kernel and pipeline) -----------------------
    def _execute_proposal(self, state: TaskState, proposal: ExperimentProposal,
                          drift: dict, round_number: int = 0) -> dict:
        key = state.config.key
        self.log("info", key, f"round {round_number}: executing {proposal.strategy} "
                              f"{_compact(proposal.params, 160)} pp={_compact(proposal.postprocess, 120)}")
        result = state.executor.run(
            state.parent_path, proposal,
            calibration=state.splits["paths"]["calibration"],
            selection=state.splits["paths"]["selection"],
            report=state.splits["paths"]["report"],
            train=state.splits["paths"]["train"],
            drift={k: drift.get(k) for k in ("severity", "score", "causes")},
            drift_probe=state.splits["paths"].get("drift_probe"),
            extended_train=state.extended_path,
            seed=self.seed + round_number)
        self.experiments_used += 1
        self.global_cpu += result.cpu_seconds
        status = "ok" if result.ok else f"failed: {result.error}"
        self.log("info" if result.ok else "warn", key,
                 f"round {round_number}: {result.experiment_id} {status}"
                 + (f" selection_primary={result.metrics['selection']['primary']:.6f}"
                    if result.ok and result.metrics.get("selection") else ""))
        # Keep the complete result for audit artifacts. Agent-facing tool
        # responses apply the selection-only projection in rsi.agent.views.
        return result.to_dict()

    def _apply_promotion(self, state: TaskState, best: dict) -> None:
        state.parent_path = Path(best["candidate_checkpoint"])
        state.parent_sha = best["checkpoint_sha256"]
        state.postprocess = best["postprocess"]
        model = ExtendedModel.from_checkpoint(state.parent_path)
        state.selection_metrics = self.evaluate(model, state.selection, state.postprocess,
                                                 state.config, full=state.validation)
        state.report_metrics = self.evaluate(model, state.report, state.postprocess,
                                              state.config, full=state.validation)
        if state.probe is not None:
            state.probe_metrics = self.evaluate(model, state.probe, state.postprocess, state.config)
        state.best_primary = state.selection_metrics["primary"]
        state.best_round = len(state.rounds) + 1
        state.stall_rounds = 0

    def _record_panel(self, state: TaskState, entries: list[dict],
                      count: bool = True) -> None:
        if count:
            self.agent_stats["panels"] += 1
            state.agent_stats = dict(self.agent_stats)
        state.panel.append({
            "trigger": "kernel convened panel",
            "round_index": len(state.rounds),
            "verdict": next((e.get("standpoint", "") for e in entries
                             if e.get("role") == "arbiter"), "retain"),
            "messages": [{k: e.get(k) for k in
                          ("role", "standpoint", "evidence", "confidence", "source", "ok")}
                         for e in entries],
        })

    # -- autonomous kernel round -------------------------------------------
    def _kernel_brief(self, state: TaskState, round_number: int, drift: dict,
                      history: list[dict], memory_snapshot: str,
                      goal_loop: GoalLoop | None) -> dict:
        cfg = state.config
        return {
            "task": {"key": cfg.key, "name": cfg.name, "name_en": cfg.name_en,
                     "task": cfg.task, "primary_metric": cfg.primary_metric,
                     "direction": cfg.direction,
                     "baseline_primary": cfg.baseline_primary, "margin": cfg.margin},
            "goal": (goal_loop.state.to_dict() if goal_loop is not None
                     and goal_loop.state else {}),
            "round_index": round_number, "rounds_total": self.rounds,
            "metrics": dict(state.selection_metrics),
            "budget": self._budget_payload(),
            "memory": memory_snapshot,
            "lessons": self.recall.lessons_text(cfg.key) if self.recall else "",
            "stats": self.recall.stats_text(cfg.key) if self.recall else "",
            "history": history,
            "drift": {k: drift.get(k) for k in
                      ("severity", "score", "feature_shift", "max_feature_shift",
                       "ks_max", "causes", "evidence")},
            "catalog": strategy_catalog_text() + "\n" + postprocess_catalog_text(),
            "skills": self.skills.index_text() if self.skills else "",
        }

    def _run_kernel_round(self, state: TaskState, kernel: Kernel, coordinator: Coordinator,
                          round_index: int, goal_loop: GoalLoop, memory_snapshot: str):
        key = state.config.key
        round_number = round_index + 1
        started = time.time()
        token_before = self.budget.used
        parent_sha_before = state.parent_sha
        self.log("info", key, f"round {round_number}/{self.rounds}: kernel observes drift")
        drift = self.drift_report(state, round_index)
        state.drift = drift
        history = self._history(state)
        context = coordinator.build_context(round_number, self.rounds, drift,
                                            state.selection_metrics, history,
                                            self._budget_payload())
        brief = self._kernel_brief(state, round_number, drift, history, memory_snapshot, goal_loop)
        host = KernelHost(self, state, drift, context)
        self._write_state("running")
        outcome = kernel.run_round(host, brief, round_number,
                                   experiments_total=state.executor.experiments)
        if not outcome.results and outcome.decision is None:
            self.agent_stats["kernel_failures"] += 1
            return None
        env = outcome.env or {}
        self.agent_stats["kernel_rounds"] += 1
        self.agent_stats["delegates"] += int(env.get("delegates", 0))
        self.agent_stats["panels"] += int(env.get("panels", 0))
        self.agent_stats["memory_writes"] += int(env.get("memory_writes", 0))
        self.agent_stats["skill_writes"] += int(env.get("skill_writes", 0))
        usage = outcome.usage or {}
        self.agent_stats["kernel_tokens"] += int(usage.get("input", 0)) + int(usage.get("output", 0))
        state.agent_stats = dict(self.agent_stats)

        proposals = [ExperimentProposal(
            strategy=p.get("strategy", "retain"),
            params=p.get("params") or {},
            postprocess=p.get("postprocess") or {},
            hypothesis=p.get("hypothesis", ""),
            expected_effect=p.get("expected_effect", ""),
            risk=p.get("risk", "")) for p in outcome.proposals]
        results = outcome.results
        best = best_candidate(results, state.config.direction) if results else None
        decision = dict(outcome.decision or {
            "decision": "retain", "reason": outcome.error or "kernel produced no decision",
            "delta": 0.0, "source": "supervisor"})
        promote_target = None
        if decision.get("decision") == "promote":
            chosen_id = decision.get("experiment_id")
            promote_target = next((r for r in results if r.get("experiment_id") == chosen_id), None) or best
            if promote_target is None or not promote_target.get("ok") or not promote_target.get("candidate_checkpoint"):
                promote_target = None
                decision = {"decision": "retain", "delta": 0.0, "source": "supervisor",
                            "reason": "kernel promote downgraded: no successful candidate"}
        if promote_target is not None:
            self._apply_promotion(state, promote_target)
            self.log("info", key, f"round {round_number}: promoted -> {state.parent_sha[:22]}... "
                                  f"{state.config.primary_metric}={state.selection_metrics['primary']:.6f}")
        else:
            state.stall_rounds += 1

        diagnosis = {"summary": (outcome.finished or {}).get("summary", ""),
                     "severity": drift.get("severity"),
                     "causes": list(drift.get("causes") or []),
                     "focus": "kernel", "confidence": None,
                     "evidence": {"drift_score": drift.get("score"),
                                  "tool_calls": outcome.tool_calls},
                     "source": "kernel"}
        review = {"source": "kernel",
                  "note": "autonomous agent self-review; deterministic guard verified the decision",
                  "approvals": list(range(len(proposals))), "vetoes": [],
                  "violations": env.get("violations", [])}
        agent_record = outcome.to_record()
        agent_record["brief"] = {
            "memory_chars": len(memory_snapshot or ""),
            "lessons_chars": len(brief.get("lessons") or ""),
            "skills_chars": len(brief.get("skills") or ""),
            "history_rounds": len(brief.get("history") or []),
        }
        record = self._record_round(
            state, round_number, drift, diagnosis, proposals, review, results, decision,
            parent_sha_before, token_before, started,
            best=(promote_target or best), roles=["kernel"], agent=agent_record)
        self._swan_log(self._swan_metrics(
            state, round_number, {**decision, "controller": "kernel"}, results))
        if state.stall_rounds >= 2 and len(state.panel) < 2 and self.budget.remaining() > 20000:
            issue = (f"连续 {state.stall_rounds} 轮未提升（当前 {state.config.primary_metric}="
                     f"{state.selection_metrics['primary']:.6f}，基线 {state.config.baseline_primary:.6f}，"
                     f"漂移={drift.get('severity')}）")
            self.log("warn", key, f"round {round_number}: stall detected, convening panel")
            entries = coordinator.panel(context, issue)
            self._record_panel(state, entries)
            state.force_explore = True
            state.stall_rounds = 0
        self.log("info", key, f"round {round_number}: kernel done decision={decision['decision']} "
                              f"tokens={int(usage.get('input', 0)) + int(usage.get('output', 0))} "
                              f"trace={len(outcome.trace)} tool_calls={outcome.tool_calls}")
        return outcome

    # -- round record ------------------------------------------------------
    def _record_round(self, state: TaskState, round_number: int, drift: dict, diagnosis: dict,
                      proposals: list, review: dict, results: list, decision: dict,
                      parent_sha_before: str, token_before: int, started: float,
                      best: dict | None = None, roles: list[str] | None = None,
                      agent: dict | None = None) -> dict:
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
                    "calls": self.budget.calls,
                    "roles": list(roles) if roles is not None else list(coordinator_roles(proposals))},
            "agent": agent or {},
            "created_at": utcnow_iso(),
        }
        state.rounds.append(record)
        self.recorder.round(record)
        if self.db is not None:
            try:
                self.db.add_round(self.run_id, cfg.key, round_number, decision["decision"],
                                  strategy=strategy, postprocess=postprocess.get("type", "none"),
                                  primary=metrics.get("primary"), delta=decision.get("delta"),
                                  hypothesis=record["hypothesis"],
                                  change_summary=change_summary, payload=record)
                for item in results:
                    self.db.add_experiment(
                        self.run_id, cfg.key, round_number,
                        str(item.get("experiment_id") or "unknown"),
                        strategy=str((item.get("proposal") or {}).get("strategy", "")),
                        postprocess=str((item.get("postprocess") or {}).get("type", "none")),
                        ok=bool(item.get("ok")),
                        primary=((item.get("metrics") or {}).get("selection") or {}).get("primary"),
                        delta=decision.get("delta"),
                        cpu_seconds=item.get("cpu_seconds"),
                        payload=item)
                self.db.record_strategy(cfg.key, strategy, postprocess.get("type", "none"),
                                        decision.get("delta"),
                                        decision["decision"] == "promote")
                self.db.add_event(self.run_id, cfg.key, "info",
                                  f"round {round_number}: {decision['decision']} "
                                  f"delta={decision.get('delta', 0.0):+.6f}")
            except Exception as exc:
                self.log("warn", cfg.key, f"sqlite write failed: {exc.__class__.__name__}: {exc}")
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
            "swanlab": {"enabled": self.swan_run is not None,
                        "mode": os.environ.get("SWANLAB_MODE", "offline") if self.swanlab else None,
                        "error": self.swanlab_error,
                        "local_dir": str(self.run_dir / "swanlab") if self.swanlab else None},
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
            "controller": state.controller,
            "degraded": state.degraded,
            "agent": state.agent_stats,
            "goal": state.goal,
            "memory": state.memory_info,
            "strategy_stats": (self.db.strategy_stats(cfg.key) if self.db is not None else []),
        }


def coordinator_roles(proposals) -> list[str]:
    return ["diagnostician", "experimenter", "critic"]
