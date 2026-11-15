"""Experiment executor: validates agent proposals, runs them isolated, records evidence.

Every experiment runs in a separate Python process so a numeric crash cannot take
down the controller. The child only ever writes inside the experiment directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rsi.model import code_sha256, checkpoint_sha256
from rsi.strategies import POSTPROCESS_TYPES, STRATEGIES


class InvalidExperiment(ValueError):
    pass


@dataclass
class ExperimentProposal:
    strategy: str
    params: dict = field(default_factory=dict)
    postprocess: dict = field(default_factory=dict)
    hypothesis: str = ""
    expected_effect: str = ""
    risk: str = ""
    notes: str = ""


def _coerce_number(value, low, high, kind):
    if isinstance(value, bool):
        raise InvalidExperiment("numeric parameter may not be a boolean")
    number = kind(value)
    if not low <= number <= high:
        raise InvalidExperiment(f"parameter out of range [{low}, {high}]: {number}")
    return number


def validate_params(strategy_name, params):
    if strategy_name not in STRATEGIES:
        raise InvalidExperiment(f"strategy not allowed: {strategy_name}")
    spec = STRATEGIES[strategy_name]
    unknown = set(params) - set(spec.params)
    if unknown:
        raise InvalidExperiment(f"unknown parameters for {strategy_name}: {sorted(unknown)}")
    clean = {}
    for name, rule in spec.params.items():
        value = params.get(name, rule.get("default"))
        if rule["type"] == "float":
            clean[name] = _coerce_number(value, rule["min"], rule["max"], float)
        elif rule["type"] == "int":
            clean[name] = _coerce_number(value, rule["min"], rule["max"], int)
        elif rule["type"] == "choice":
            if value not in rule["choices"]:
                raise InvalidExperiment(f"{name} must be one of {rule['choices']}")
            clean[name] = value
        elif rule["type"] == "bool":
            if not isinstance(value, bool):
                raise InvalidExperiment(f"{name} must be a boolean")
            clean[name] = value
        else:  # pragma: no cover - registry is static
            raise InvalidExperiment("bad parameter rule")
    return clean


def validate_postprocess(postprocess, task):
    spec = dict(postprocess or {"type": "none"})
    kind = spec.get("type", "none")
    if kind not in POSTPROCESS_TYPES:
        raise InvalidExperiment(f"postprocess not allowed: {kind}")
    if kind == "none":
        return {"type": "none"}
    if kind == "recenter_blend" or kind == "recenter":
        kind = "recenter"
        weight = float(spec.get("weight", 1.0))
        if not 0.0 <= weight <= 1.0:
            raise InvalidExperiment("recenter weight must be within [0, 1]")
        mode = spec.get("mode", "batch_mean")
        if mode not in ("batch_mean", "batch_std"):
            raise InvalidExperiment("recenter mode must be batch_mean or batch_std")
        return {"type": "recenter", "mode": mode, "weight": weight, "window": int(spec.get("window", 4096))}
    if kind == "shrink":
        alpha = float(spec.get("alpha", 0.0))
        if task != "regression":
            raise InvalidExperiment("shrink calibration is only meaningful for regression")
        if not 0.0 <= alpha <= 1.0:
            raise InvalidExperiment("shrink alpha must be within [0, 1]")
        return {"type": "shrink", "alpha": alpha}
    raise InvalidExperiment(f"unsupported postprocess: {kind}")


def validate_proposal(proposal, task):
    if not isinstance(proposal, ExperimentProposal):
        raise InvalidExperiment("proposal must be an ExperimentProposal")
    params = validate_params(proposal.strategy, proposal.params)
    postprocess = validate_postprocess(proposal.postprocess, task)
    return ExperimentProposal(proposal.strategy, params, postprocess,
                              str(proposal.hypothesis)[:2000], str(proposal.expected_effect)[:1000],
                              str(proposal.risk)[:1000], str(proposal.notes)[:1000])


@dataclass
class ExperimentResult:
    ok: bool
    experiment_id: str
    proposal: dict
    parent_checkpoint: str
    candidate_checkpoint: str | None
    checkpoint_sha256: str | None
    code_sha256: str
    metrics: dict
    postprocess: dict
    cpu_seconds: float
    wall_seconds: float
    error: str | None = None
    notes: dict = field(default_factory=dict)
    log_path: str | None = None

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


class Executor:
    def __init__(self, task_key, task, classes, paths, harness_root, state_dir, timeout_seconds=1800):
        self.task_key = task_key
        self.task = task
        self.classes = list(classes)
        self.paths = paths
        self.harness_root = Path(harness_root)
        self.state_dir = Path(state_dir)
        self.timeout_seconds = timeout_seconds
        self.experiments = 0
        self.cpu_seconds = 0.0

    def run(self, parent_checkpoint, proposal, *, calibration, selection, report, train,
            drift=None, drift_probe=None, extended_train=None, seed=42):
        proposal = validate_proposal(proposal, self.task)
        self.experiments += 1
        experiment_id = f"exp-{self.experiments:04d}"
        job_dir = self.state_dir / "experiments" / experiment_id
        if job_dir.exists():
            raise RuntimeError(f"experiment directory already exists: {job_dir}")
        job_dir.mkdir(parents=True, mode=0o700)
        job = {
            "experiment_id": experiment_id,
            "task_key": self.task_key,
            "task": self.task,
            "classes": self.classes,
            "parent_checkpoint": str(parent_checkpoint),
            "train": str(self.paths["train"]),
            "calibration": str(calibration),
            "selection": str(selection),
            "report": str(report),
            "drift_probe": str(drift_probe) if drift_probe else None,
            "extended_train": str(extended_train) if extended_train else None,
            "seed": int(seed),
            "proposal": {
                "strategy": proposal.strategy,
                "params": proposal.params,
                "postprocess": proposal.postprocess,
                "hypothesis": proposal.hypothesis,
                "expected_effect": proposal.expected_effect,
                "risk": proposal.risk,
                "notes": proposal.notes,
            },
            "drift": drift,
            "candidate_checkpoint": str(job_dir / "candidate.npz"),
            "result": str(job_dir / "result.json"),
        }
        job_path = job_dir / "job.json"
        job_path.write_text(json.dumps(job, indent=1))
        started = time.monotonic()
        log_path = job_dir / "worker.log"
        python_path = str(self.harness_root)
        inherited = os.environ.get("PYTHONPATH")
        if inherited:
            python_path = f"{python_path}{os.pathsep}{inherited}"
        env = {**os.environ, "PYTHONPATH": python_path, "PYTHONHASHSEED": "0",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            with log_path.open("wb") as log:
                completed = subprocess.run([sys.executable, "-m", "rsi.worker", "--job", str(job_dir)],
                                           cwd=str(self.harness_root), env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            return ExperimentResult(False, experiment_id, job["proposal"], str(parent_checkpoint), None, None,
                                    code_sha256(self.harness_root), {}, job["proposal"]["postprocess"],
                                    0.0, time.monotonic() - started, "timeout", {}, str(log_path))
        wall = time.monotonic() - started
        result_path = Path(job["result"])
        if completed.returncode != 0 or not result_path.is_file():
            tail = log_path.read_text(errors="replace")[-2000:]
            return ExperimentResult(False, experiment_id, job["proposal"], str(parent_checkpoint), None, None,
                                    code_sha256(self.harness_root), {}, job["proposal"]["postprocess"],
                                    0.0, wall, f"worker failed (exit {completed.returncode})", {"log_tail": tail}, str(log_path))
        raw = json.loads(result_path.read_text())
        self.cpu_seconds += float(raw.get("cpu_seconds", 0.0))
        return ExperimentResult(
            ok=bool(raw.get("ok")), experiment_id=experiment_id, proposal=job["proposal"],
            parent_checkpoint=str(raw.get("parent_checkpoint", parent_checkpoint)),
            candidate_checkpoint=raw.get("candidate_checkpoint"),
            checkpoint_sha256=raw.get("checkpoint_sha256"),
            code_sha256=raw.get("code_sha256", code_sha256(self.harness_root)),
            metrics=raw.get("metrics", {}), postprocess=raw.get("postprocess", {"type": "none"}),
            cpu_seconds=float(raw.get("cpu_seconds", 0.0)), wall_seconds=wall,
            error=raw.get("error"), notes=raw.get("notes", {}), log_path=str(log_path))
