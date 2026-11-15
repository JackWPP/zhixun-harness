"""Recall: turn the SQLite index + memory files into compact prompt context."""
from __future__ import annotations

from .db import StateDB
from .skills import SkillStore
from .store import MemoryStore


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


class Recall:
    def __init__(self, db: StateDB | None, memory: MemoryStore | None,
                 skills: SkillStore | None = None):
        self.db = db
        self.memory = memory
        self.skills = skills

    # -- pieces ------------------------------------------------------------
    def lessons(self, task: str, strategy: str | None = None, limit: int = 5) -> list[dict]:
        if not self.db:
            return []
        scoped = self.db.lessons(task, limit=limit, scope_contains=strategy) if strategy else []
        glob = self.db.lessons(task, limit=limit)
        seen: set[int] = set()
        merged: list[dict] = []
        for item in scoped + glob:
            if int(item["id"]) in seen:
                continue
            seen.add(int(item["id"]))
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    def lessons_text(self, task: str, strategy: str | None = None, limit: int = 4,
                     max_chars: int = 900) -> str:
        items = self.lessons(task, strategy, limit)
        if not items:
            return "(no lessons recorded yet)"
        lines = []
        for item in items:
            wins = item.get("wins") or 0
            suffix = f" [wins={wins}]" if wins else ""
            lines.append(f"- {_clip(item['text'], 240)}{suffix}")
        return _clip("\n".join(lines), max_chars)

    def stats_text(self, task: str, limit: int = 6) -> str:
        if not self.db:
            return "(no strategy statistics yet)"
        rows = self.db.strategy_stats(task, limit)
        if not rows:
            return "(no strategy statistics yet)"
        lines = []
        for row in rows:
            postprocess = row.get("postprocess_type") or "none"
            lines.append(
                f"- {row['strategy']} + {postprocess}: trials={row['trials']}, "
                f"win_rate={row['win_rate']:.0%}, mean_delta={row['mean_delta']:+.4f}, "
                f"best={row['best_delta']:+.4f}")
        return "\n".join(lines)

    def search_text(self, query: str, limit: int = 10) -> str:
        if not self.db:
            return "(records index unavailable)"
        rows = self.db.search(query, limit)
        if not rows:
            return f"(no records matching {query!r})"
        lines = []
        for row in rows:
            lines.append(f"- [{row['kind']}] {row['task'] or '-'} {row['ref'] or ''}: "
                         f"{_clip(row['text'] or '', 180)}")
        return "\n".join(lines)

    def memory_snapshot(self, task: str) -> str:
        if not self.memory:
            return ""
        return self.memory.snapshot(task)

    # -- brief -------------------------------------------------------------
    def brief(self, task: str, strategy: str | None = None,
              max_chars: int = 1600) -> str:
        sections = []
        snapshot = self.memory_snapshot(task)
        if snapshot:
            sections.append(snapshot)
        lessons = self.lessons_text(task, strategy)
        sections.append(f"## Lessons from previous rounds/runs\n{lessons}")
        stats = self.stats_text(task)
        sections.append(f"## Strategy statistics (public data only)\n{stats}")
        text = "\n\n".join(sections)
        return _clip(text, max_chars)
