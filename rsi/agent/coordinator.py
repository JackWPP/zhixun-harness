"""Coordinator: six-role LLM collaboration with deterministic safety net.

The coordinator decides *when* to spend LLM tokens. Routine rounds use three
calls (diagnostician -> experimenter -> critic); the arbiter is only consulted at
a boundary; the panel is only convened when the runner reports a stall. Every LLM
submission is validated before it can become an experiment, so a hallucinating
model can never escape the whitelisted strategy catalog or the budget.
"""
from __future__ import annotations

import json

from rsi.executor import ExperimentProposal, InvalidExperiment, validate_proposal
from rsi.planner import fallback_proposals, postprocess_catalog_text, strategy_catalog_text
from rsi.planner import default_plans
from rsi.tasks import TaskConfig

from . import prompts
from .runtime import PIRuntime, RoleResult

ROLES = ("diagnostician", "experimenter", "critic", "arbiter")
PANEL_ROLES = ("diagnostician", "experimenter", "critic", "arbiter")


def _safe_json(text, fallback=None):
    if isinstance(text, dict):
        return text
    try:
        value = json.loads(text)
    except Exception:
        return fallback
    return value if isinstance(value, dict) else fallback


class Coordinator:
    def __init__(self, task: TaskConfig, runtime: PIRuntime | None = None, recorder=None):
        self.task = task
        self.runtime = runtime
        self.recorder = recorder
        self.used_roles: list[str] = []

    # -- helpers ----------------------------------------------------------
    def _llm_ready(self) -> bool:
        if self.runtime is None:
            return False
        if not self.runtime.available():
            return False
        return not self.runtime.budget.exhausted()

    def _record(self, role: str, result: RoleResult, **extra) -> None:
        if self.recorder is not None:
            self.recorder.llm_call(self.task.key, role, result.usage,
                                   latency=round(result.latency, 2), ok=result.ok,
                                   error=result.error, tool_calls=result.tool_calls, **extra)

    def _call(self, role: str, system: str, user: str, toolset: str,
              context: dict) -> RoleResult | None:
        self.used_roles.append(role)
        result = self.runtime.call(role, system, user, toolset, context)
        self._record(role, result)
        return result

    # -- 1. diagnosis -----------------------------------------------------
    def diagnose(self, context: dict) -> dict:
        drift = context.get("drift", {}) or {}
        if self._llm_ready():
            result = self._call("diagnostician", prompts.SYSTEM_PROMPTS["diagnostician"],
                                prompts.diagnosis_prompt(context), "diagnosis", context)
            if result is not None:
                for payload in result.submitted:
                    if payload.get("kind") != "diagnosis":
                        continue
                    if payload.get("summary") and payload.get("severity"):
                        return {
                            "summary": str(payload["summary"])[:1200],
                            "severity": str(payload["severity"]),
                            "causes": [str(c) for c in payload.get("causes", [])][:4],
                            "focus": [str(c) for c in payload.get("focus", [])][:3],
                            "evidence": str(payload.get("evidence", ""))[:800],
                            "confidence": float(payload.get("confidence", 0.5)),
                            "source": "llm",
                        }
        return self._fallback_diagnosis(drift)

    def _fallback_diagnosis(self, drift: dict) -> dict:
        severity = str(drift.get("severity", "stable"))
        causes = [str(c) for c in (drift.get("causes") or [])][:4]
        evidence = json.dumps(drift.get("evidence", {}), ensure_ascii=False)[:600]
        focus = []
        for item in (drift.get("evidence", {}) or {}).get("top_shifted_features", [])[:3]:
            if isinstance(item, dict):
                focus.append(str(item.get("feature", item.get("name", ""))))
            else:
                focus.append(str(item))
        if not focus:
            focus = ["feature_distribution" if severity != "stable" else "no_action"]
        summary = (drift.get("summary") or
                   f"数据流漂移严重度={severity}；证据={evidence}。")
        return {"summary": str(summary)[:1200], "severity": severity, "causes": causes,
                "focus": focus[:3], "evidence": evidence,
                "confidence": 0.55 if severity != "stable" else 0.7,
                "source": "deterministic"}

    # -- 2. experiment proposals -----------------------------------------
    def propose(self, context: dict, diagnosis: dict) -> list[ExperimentProposal]:
        if self._llm_ready():
            result = self._call("experimenter", prompts.SYSTEM_PROMPTS["experimenter"],
                                prompts.plan_prompt(context, diagnosis), "plan", context)
            proposals = []
            if result is not None:
                for payload in result.submitted:
                    if payload.get("kind") != "plan":
                        continue
                    params = _safe_json(payload.get("params_json"), {}) or {}
                    postprocess = _safe_json(payload.get("postprocess_json"), {}) or {}
                    proposal = ExperimentProposal(
                        strategy=str(payload.get("strategy", "")),
                        params=params, postprocess=postprocess,
                        hypothesis=str(payload.get("hypothesis", ""))[:2000],
                        expected_effect=str(payload.get("expected_effect", ""))[:1000],
                        risk=str(payload.get("risk", ""))[:1000],
                        notes="llm",
                    )
                    try:
                        proposals.append(validate_proposal(proposal, self.task.task))
                    except (InvalidExperiment, TypeError, ValueError):
                        continue
                    if len(proposals) >= 3:
                        break
            if proposals:
                return proposals
        return fallback_proposals(self.task.key, context.get("round_index", 0),
                                      stats=context.get("stats"))

    # -- 3. review --------------------------------------------------------
    def criticize(self, context: dict, proposals: list[ExperimentProposal]) -> dict:
        plans = [p.__dict__ for p in proposals]
        if self._llm_ready():
            result = self._call("critic", prompts.SYSTEM_PROMPTS["critic"],
                                prompts.review_prompt(context, plans), "review", context)
            if result is not None:
                for payload in result.submitted:
                    if payload.get("kind") != "review":
                        continue
                    approvals = [int(i) for i in payload.get("approvals", [])
                                 if isinstance(i, int) and 0 <= i < len(proposals)]
                    vetoes = [str(v) for v in payload.get("vetoes", [])][:6]
                    if approvals:
                        return {"approvals": sorted(set(approvals)), "vetoes": vetoes,
                                "note": str(payload.get("note", ""))[:600],
                                "source": "llm"}
        approvals = list(range(min(2, len(proposals))))
        return {"approvals": approvals, "vetoes": [], "source": "deterministic",
                "note": "no vetoes: proposals already validated against the catalog"}

    # -- 4. arbitration (boundary only) ----------------------------------
    def arbitrate(self, context: dict, proposals: list[ExperimentProposal],
                  candidates: list[dict]) -> dict:
        if self._llm_ready() and len(candidates) >= 2:
            result = self._call("arbiter", prompts.SYSTEM_PROMPTS["arbiter"],
                                prompts.verdict_prompt(context, [p.__dict__ for p in proposals],
                                                       candidates), "verdict", context)
            if result is not None:
                for payload in result.submitted:
                    if payload.get("kind") != "verdict":
                        continue
                    return {"decision": str(payload.get("decision", "")),
                            "chosen": int(payload.get("chosen", 0)),
                            "reason": str(payload.get("reason", ""))[:800],
                            "confidence": float(payload.get("confidence", 0.5)),
                            "source": "llm"}
        return {"source": "deterministic", "reason": "boundary arbitration skipped"}

    # -- 5. panel (stall only) -------------------------------------------
    def panel(self, context: dict, issue: str) -> list[dict]:
        entries = []
        for role in PANEL_ROLES:
            entry = {"role": role, "standpoint": "", "evidence": "", "confidence": 0.0,
                     "source": "deterministic"}
            if self._llm_ready():
                system = prompts.SYSTEM_PROMPTS["panel"].format(role=role)
                result = self._call(role, system,
                                    prompts.standpoint_prompt(context, role, issue),
                                    "standpoint", context)
                if result is not None:
                    for payload in result.submitted:
                        if payload.get("kind") != "standpoint":
                            continue
                        entry = {"role": role,
                                 "standpoint": str(payload.get("standpoint", ""))[:900],
                                 "evidence": str(payload.get("evidence", ""))[:600],
                                 "confidence": float(payload.get("confidence", 0.5)),
                                 "source": "llm"}
                        break
            if not entry["standpoint"]:
                entry["standpoint"] = self._deterministic_standpoint(role, context, issue)
                entry["confidence"] = 0.4
            entries.append(entry)
        if self.recorder is not None:
            self.recorder.panel(self.task.key, entries, issue=issue)
        return entries

    @staticmethod
    def _deterministic_standpoint(role: str, context: dict, issue: str) -> str:
        drift = context.get("drift", {}) or {}
        if role == "diagnostician":
            return f"卡点={issue}；当前漂移={drift.get('severity', 'unknown')}，应先确认漂移是否仍在加剧。"
        if role == "experimenter":
            return "建议切换到更保守的参数网格（更小学习率、更少轮数），并保留后处理校准作为保底。"
        if role == "critic":
            return "连续无改善说明现有假设空间可能错误，应优先 review 泄漏风险与评估噪声，而非继续调参。"
        return "建议维持当前最优检查点（retain），只允许不改动父模型的推理期校准候选。"

    # -- 6. reporter ------------------------------------------------------
    def report(self, context: dict, round_summary: dict) -> str:
        if self._llm_ready():
            result = self._call("reporter", prompts.SYSTEM_PROMPTS["reporter"],
                                prompts.reporter_prompt(context, round_summary), "reporter",
                                context)
            if result is not None and result.ok and result.text:
                return result.text.strip()[:1200]
        return str(round_summary.get("fallback_summary", ""))[:1200]

    # -- context ----------------------------------------------------------
    def build_context(self, round_index: int, rounds_total: int, drift: dict,
                      metrics: dict, history: list[dict], budget: dict) -> dict:
        return {
            "task": {"key": self.task.key, "name_en": self.task.name_en,
                     "task": self.task.task, "primary_metric": self.task.primary_metric,
                     "direction": self.task.direction,
                     "baseline_primary": self.task.baseline_primary,
                     "margin": self.task.margin},
            "round_index": round_index, "rounds_total": rounds_total,
            "drift": drift, "metrics": metrics, "history": history[-5:],
            "budget": budget,
            "catalog": {"strategies": strategy_catalog_text(),
                        "postprocess": postprocess_catalog_text()},
        }
