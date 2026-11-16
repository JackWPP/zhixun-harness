"""Deterministic safety boundary around the autonomous kernel.

The LLM may *propose* and *act*, but every action that can change the industrial
model, spend budget or touch memory passes through this guard:

* strategies/params/postprocess must exist in the frozen catalog;
* training data may only be the official ``train`` split. The validation-derived
  calibration/selection/report splits must never be fitted (competition rule:
  验证/隐藏集不能参与拟合), and probe metrics never enter context;
* one decision per round, at most one promotion per round, and the gate is
  recomputed here from the frozen selection metrics (the LLM can only be more
  conservative, never force a promotion);
* text written to memory/skills is scanned for injection and credential
  patterns;
* every action is returned as an audit record for records.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from rsi.executor import ExperimentProposal, InvalidExperiment, validate_proposal
from rsi.selection import GateDecision, gate
from rsi.tasks import TaskConfig

from .views import heldout_text

ALLOWED_TRAIN_DATA = {"train"}  # see rsi.policy for opt-in "extended"
DENIED_ACTIONS = {"train+calibration", "train+all-public", "validation", "hidden"}

_PROBE_METRIC_KEYS = ("probe", "hidden", "external")
_SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key|secret|password|token)\s*[=:]\s*\S{12,}", re.I),
    re.compile(r"trio://", re.I),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
]


class GuardViolation(Exception):
    """Raised when an agent action is outside the allowed envelope."""


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "reason": self.reason, "payload": self.payload}


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Guard:
    def __init__(self, task: TaskConfig, *, max_experiments_per_round: int = 3,
                 max_experiments_per_task: int = 40,
                 max_round_tokens: int | None = None):
        self.task = task
        self.max_experiments_per_round = int(max_experiments_per_round)
        self.max_experiments_per_task = int(max_experiments_per_task)
        self.max_round_tokens = max_round_tokens
        self.violations: list[dict] = []

    # -- proposals ---------------------------------------------------------
    def validate_catalog(self, proposal: ExperimentProposal) -> ExperimentProposal:
        """Validate strategy/params/postprocess/data without touching caps."""
        try:
            clean = validate_proposal(proposal, self.task)
        except InvalidExperiment as exc:
            raise GuardViolation(str(exc)) from exc
        from rsi.policy import deny_reason
        data = (clean.params or {}).get("data")
        if data:
            reason = deny_reason(str(data), self.task)
            if reason:
                raise GuardViolation(reason)
        self._scan_text(" ".join([clean.hypothesis, clean.expected_effect, clean.risk]),
                        "proposal text")
        return clean

    def validate_experiment(self, proposal: ExperimentProposal,
                            experiments_this_round: int,
                            experiments_total: int) -> ExperimentProposal:
        if experiments_this_round >= self.max_experiments_per_round:
            raise GuardViolation(
                f"per-round experiment cap reached ({self.max_experiments_per_round})")
        if experiments_total >= self.max_experiments_per_task:
            raise GuardViolation(
                f"per-task experiment cap reached ({self.max_experiments_per_task})")
        return self.validate_catalog(proposal)

    # -- decisions ---------------------------------------------------------
    def check_decision(self, decision: str, chosen_experiment: str | None,
                       candidates: dict[str, dict], parent_primary: float,
                       used_experiments_this_round: int,
                       parent_metrics: dict | None = None) -> GateDecision:
        if decision not in ("promote", "retain", "rollback"):
            raise GuardViolation(f"unknown decision: {decision}")
        if used_experiments_this_round == 0:
            raise GuardViolation("cannot decide a round without at least one experiment")
        if decision == "promote":
            if not chosen_experiment or chosen_experiment not in candidates:
                raise GuardViolation("promote requires a chosen experiment id")
            candidate = candidates[chosen_experiment]
            if not candidate.get("ok"):
                raise GuardViolation("chosen experiment failed and cannot be promoted")
            selection = (candidate.get("metrics") or {}).get("selection") or {}
            if "primary" not in selection:
                raise GuardViolation("chosen experiment has no selection metric")
            deterministic = gate(self.task.task, float(parent_primary),
                                 float(selection["primary"]), self.task.margin,
                                 parent_metrics, selection)
            if deterministic.decision != "promote":
                raise GuardViolation(
                    f"gate rejects promotion: {deterministic.reason} "
                    f"(delta={deterministic.delta:+.6f}, margin={self.task.margin})")
            return deterministic
        if decision == "rollback":
            if chosen_experiment and chosen_experiment not in candidates:
                raise GuardViolation("rollback references an unknown experiment")
            return GateDecision("rollback", "agent chose rollback (conservative)", 0.0)
        return GateDecision("retain", "agent chose retain (conservative)", 0.0)

    # -- prompt/memory text ------------------------------------------------
    def scan_text(self, text: str, where: str = "text") -> str:
        """Return sanitized text; raise when it contains secrets/injection."""
        self._scan_text(text, where)
        if heldout_text(text):
            raise GuardViolation(f"{where} refers to held-out evaluation evidence")
        cleaned = text
        for key in _PROBE_METRIC_KEYS:
            cleaned = re.sub(rf"\b{key}[_a-z]*\s*[:=]\s*[-+0-9.]+", "", cleaned, flags=re.I)
        return cleaned

    def _scan_text(self, text: str, where: str) -> None:
        text = text or ""
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise GuardViolation(f"{where} contains a secret/credential pattern")
        from rsi.memory.store import scan_content
        threat = scan_content(text)
        if threat:
            raise GuardViolation(f"{where} blocked by injection guard: {threat}")

    def check_skill_text(self, text: str) -> None:
        self._scan_text(text, "skill")
        if heldout_text(text):
            raise GuardViolation("skill refers to held-out evaluation evidence")

    # -- audit -------------------------------------------------------------
    def audit(self, action: str, payload: dict | None = None) -> dict:
        record = {"action": action, "digest": _digest(payload or {}),
                  "allowed": True, "at": __import__("datetime").datetime.utcnow().isoformat()}
        return record

    def reject(self, action: str, reason: str, payload: dict | None = None) -> dict:
        record = {"action": action, "digest": _digest(payload or {}),
                  "allowed": False, "reason": reason}
        self.violations.append(record)
        return record
