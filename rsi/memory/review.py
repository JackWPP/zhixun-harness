"""End-of-task review: distill rounds into durable lessons and memory entries.

Deterministic by default so it works with ``--llm off`` and never blocks a
run; an optional LLM reviewer can later propose better wording, but every
proposal still goes through the memory guard.
"""
from __future__ import annotations

from .db import StateDB
from .skills import SkillStore
from .store import MemoryStore


def _selection_primary(record: dict):
    metrics = record.get("metrics") or {}
    candidates = [metrics.get("selection") or {}, metrics,
                  record.get("validation_metrics") or {}]
    for source in candidates:
        for key in ("primary", "macro_f1", "rmse"):
            if source.get(key) is not None:
                try:
                    return float(source[key])
                except (TypeError, ValueError):
                    continue
    return None


def _strategy_label(record: dict) -> str:
    strategy = record.get("strategy") or "retain"
    postprocess = record.get("postprocess") or "none"
    if postprocess in ("", "none"):
        return strategy
    return f"{strategy}+{postprocess}"


def _delta_text(delta) -> str:
    try:
        return f"{float(delta):+.4f}"
    except (TypeError, ValueError):
        return "n/a"


class MemoryReviewer:
    def __init__(self, memory: MemoryStore | None, db: StateDB | None,
                 skills: SkillStore | None = None, runtime=None, run_id: str = ""):
        self.memory = memory
        self.db = db
        self.skills = skills
        self.runtime = runtime
        self.run_id = run_id

    # -- deterministic distillation ---------------------------------------
    def distill(self, task: str, rounds: list[dict], max_lessons: int = 4) -> list[dict]:
        lessons: list[dict] = []
        seen: set[str] = set()
        promoted = 0
        retained_streak = 0

        def push(text: str, scope: str, evidence: dict, wins: int = 0) -> None:
            text = (text or "").strip()
            if not text or text in seen or len(lessons) >= max_lessons:
                return
            seen.add(text)
            lessons.append({"task": task, "text": text, "scope": scope,
                            "evidence": evidence, "wins": wins})

        for record in rounds:
            decision = (record.get("decision") or "").strip()
            index = record.get("index", record.get("round_index", 0))
            label = _strategy_label(record)
            delta = _delta_text(record.get("delta"))
            primary = _selection_primary(record)
            hypothesis = (record.get("hypothesis") or "").strip()
            evidence = {"round": index, "decision": decision, "delta": delta,
                        "primary": primary, "strategy": label}
            if decision == "promote":
                promoted += 1
                retained_streak = 0
                push(f"{task}: {label} promoted in round {index} "
                     f"(Δ{delta}, selection primary={primary}); "
                     f"hypothesis: {hypothesis[:100]}",
                     scope=label, evidence=evidence, wins=1)
            elif decision == "rollback":
                retained_streak = 0
                push(f"{task}: avoid {label} without new evidence — rolled back in "
                     f"round {index} (Δ{delta})",
                     scope=label, evidence=evidence)
            elif decision == "retain":
                retained_streak += 1
                if retained_streak >= 4:
                    push(f"{task}: {retained_streak} consecutive retentions by round {index}; "
                         "re-explore an orthogonal strategy or stop spending experiments",
                         scope="stall", evidence=evidence)

        if not lessons:
            push(f"{task}: no candidate beat the gate in {len(rounds)} rounds; "
                 "keep the parent checkpoint and improve drift calibration instead of "
                 "fine-tuning on the same distribution",
                 scope="baseline", evidence={"rounds": len(rounds), "promotions": promoted})
        return lessons

    # -- persist -----------------------------------------------------------
    def review(self, task: str, rounds: list[dict]) -> dict:
        lessons = self.distill(task, rounds)
        db_ids: list[int] = []
        if self.db:
            for lesson in lessons:
                try:
                    db_ids.append(self.db.add_lesson(
                        lesson["task"], lesson["text"], scope=lesson["scope"],
                        evidence=str(lesson["evidence"]), wins=int(lesson["wins"])))
                except Exception:
                    continue
        memory_updates: list[str] = []
        if self.memory:
            for lesson in lessons:
                if int(lesson.get("wins") or 0) < 1:
                    continue
                result = self.memory.add(lesson["text"], target="task", task_key=task)
                if result.ok:
                    memory_updates.append(lesson["text"])
                if len(memory_updates) >= 2:
                    break
        return {"task": task, "lessons": lessons, "db_ids": db_ids,
                "memory_updates": memory_updates}
