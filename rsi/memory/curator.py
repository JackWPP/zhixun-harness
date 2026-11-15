"""Background curator: keep the skill library small and useful."""
from __future__ import annotations

from .db import StateDB
from .skills import SkillStore


class SkillCurator:
    def __init__(self, skills: SkillStore, db: StateDB | None,
                 min_runs_before_archive: int = 2):
        self.skills = skills
        self.db = db
        self.min_runs = int(min_runs_before_archive)

    def usage_report(self) -> list[dict]:
        usage = {}
        if self.db:
            usage = {row["skill"]: row for row in self.db.skill_usage()}
        report = []
        for skill in self.skills.list():
            row = usage.get(skill.name, {})
            report.append({
                "name": skill.name,
                "description": skill.description,
                "seed": skill.seed,
                "uses": int(row.get("uses", 0) or 0),
                "last_used": row.get("last_used"),
            })
        report.sort(key=lambda item: (not item["seed"], -item["uses"], item["name"]))
        return report

    def archived(self) -> list[str]:
        root = self.skills.archive_root
        if not root.is_dir():
            return []
        return sorted(child.name for child in root.iterdir() if child.is_dir())

    def curate(self) -> list[dict]:
        """Archive agent-created skills that were never used across runs."""
        if not self.db or self.db.run_count() < self.min_runs:
            return []
        usage = {row["skill"]: int(row["uses"] or 0) for row in self.db.skill_usage()}
        archived = []
        for skill in self.skills.list():
            if skill.seed:
                continue
            if usage.get(skill.name, 0) > 0:
                continue
            try:
                result = self.skills.archive(skill.name)
            except Exception as exc:  # pragma: no cover - defensive
                archived.append({"name": skill.name, "error": str(exc)})
                continue
            archived.append({"name": skill.name, "path": result.get("path", "")})
        return archived
