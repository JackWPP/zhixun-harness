"""Curated skills (procedural memory) with progressive disclosure.

A skill is a ``SKILL.md`` file with a small YAML frontmatter and a body that
follows a fixed template (When / Procedure / Pitfalls / Verification).  The
agent sees a compact index (Level 0) and loads a full skill on demand (Level 1),
which keeps the system prompt small.

Skills are *data*, never executable code: the store rejects fenced shell
commands, imports of dangerous modules and external URLs.  Seed skills ship
with the repository; agent-created skills can be archived by the curator when
they go unused.
"""
from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .store import MemoryError_, atomic_write_text, scan_content

MAX_SKILL_BYTES = 16_384
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{2,40}$")

SEED_SKILLS = (
    "drift-diagnosis",
    "experiment-design",
    "calibration-playbook",
    "leakage-audit",
    "stall-recovery",
    "reporter",
    "sequential-rsi",
)

_BLOCKED_PATTERNS = [
    re.compile(r"```\s*(bash|sh|shell|zsh)", re.I),
    re.compile(r"\b(subprocess|os\.system|shutil\.rmtree|eval\(|exec\(|__import__)\b"),
    re.compile(r"(curl|wget)\s+https?://", re.I),
    re.compile(r"https?://\S+", re.I),
]


@dataclass
class Skill:
    name: str
    description: str
    version: str = "1"
    tags: list[str] = field(default_factory=list)
    body: str = ""
    path: str = ""
    seed: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "version": self.version, "tags": list(self.tags),
                "seed": self.seed, "path": self.path}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise MemoryError_("SKILL.md must start with a '---' frontmatter block")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise MemoryError_("unterminated SKILL.md frontmatter") from exc
    meta: dict = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise MemoryError_(f"invalid frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    body = "\n".join(lines[end + 1:]).strip()
    return meta, body


def validate_skill_text(text: str, expected_name: str | None = None) -> tuple[dict, str]:
    if len(text.encode("utf-8")) > MAX_SKILL_BYTES:
        raise MemoryError_(f"skill exceeds {MAX_SKILL_BYTES} bytes")
    threat = scan_content(text)
    if threat:
        raise MemoryError_(f"skill blocked by guard: {threat}")
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(text):
            raise MemoryError_(f"skill contains blocked content: {pattern.pattern[:40]}")
    meta, body = parse_frontmatter(text)
    name = (meta.get("name") or "").strip()
    if not NAME_RE.match(name):
        raise MemoryError_("frontmatter 'name' must match ^[a-z][a-z0-9_-]{2,40}$")
    if expected_name and name != expected_name:
        raise MemoryError_(f"frontmatter name {name!r} != directory {expected_name!r}")
    description = (meta.get("description") or "").strip()
    if not description:
        raise MemoryError_("frontmatter 'description' is required")
    if len(description) > 240:
        raise MemoryError_("description must be <= 240 chars")
    if not body:
        raise MemoryError_("skill body is empty")
    return meta, body


def render_skill(name: str, description: str, body: str, version: str = "1",
                 tags: list[str] | None = None) -> str:
    tag_line = ", ".join(tags or [])
    return (f"---\nname: {name}\ndescription: {description}\nversion: {version}\n"
            f"tags: {tag_line}\n---\n\n{body.strip()}\n")


class SkillStore:
    def __init__(self, root: str | Path, on_use=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_root = self.root / ".archive"
        self.on_use = on_use

    # -- reading -----------------------------------------------------------
    def names(self) -> list[str]:
        names = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and (child / "SKILL.md").is_file():
                names.append(child.name)
        return names

    def load(self, name: str) -> Skill:
        if not NAME_RE.match(name or ""):
            raise MemoryError_(f"invalid skill name: {name!r}")
        path = self.root / name / "SKILL.md"
        if not path.is_file():
            raise MemoryError_(f"unknown skill: {name}")
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        tags = [tag.strip() for tag in (meta.get("tags") or "").split(",") if tag.strip()]
        return Skill(name=meta.get("name", name),
                     description=meta.get("description", ""),
                     version=meta.get("version", "1"),
                     tags=tags, body=body, path=str(path),
                     seed=name in SEED_SKILLS)

    def list(self) -> list[Skill]:
        skills = []
        for name in self.names():
            try:
                skills.append(self.load(name))
            except MemoryError_:
                continue
        return skills

    def index_text(self) -> str:
        skills = self.list()
        if not skills:
            return "(no skills)"
        lines = []
        for skill in skills:
            tag_text = f" [tags: {', '.join(skill.tags)}]" if skill.tags else ""
            lines.append(f"- {skill.name}: {skill.description} (v{skill.version}){tag_text}")
        return "\n".join(lines)

    def read_text(self, name: str) -> str:
        skill = self.load(name)
        if self.on_use:
            try:
                self.on_use(name)
            except Exception:
                pass
        header = (f"# SKILL: {skill.name} (v{skill.version})\n"
                  f"{skill.description}\n{'─' * 42}\n")
        return header + skill.body

    # -- writing -----------------------------------------------------------
    def create(self, name: str, text: str) -> dict:
        if not NAME_RE.match(name or ""):
            raise MemoryError_(f"invalid skill name: {name!r}")
        directory = self.root / name
        if directory.exists():
            raise MemoryError_(f"skill already exists: {name}")
        meta, body = validate_skill_text(text, expected_name=name)
        directory.mkdir(parents=True)
        atomic_write_text(directory / "SKILL.md", render_skill(
            name, meta.get("description", ""), body,
            version=meta.get("version", "1"),
            tags=[tag.strip() for tag in (meta.get("tags") or "").split(",") if tag.strip()]))
        return self.load(name).to_dict()

    def patch(self, name: str, old_string: str, new_string: str) -> dict:
        skill = self.load(name)
        raw = (self.root / name / "SKILL.md").read_text(encoding="utf-8")
        if not old_string or raw.count(old_string) != 1:
            raise MemoryError_("old_string must appear exactly once in SKILL.md")
        updated = raw.replace(old_string, new_string)
        validate_skill_text(updated, expected_name=name)
        atomic_write_text(self.root / name / "SKILL.md", updated)
        return self.load(name).to_dict()

    def delete(self, name: str, archive: bool = True, force: bool = False) -> dict:
        skill = self.load(name)
        if skill.seed and not force:
            raise MemoryError_("seed skills are protected from agent deletion")
        source = self.root / name
        target = self.archive_root / f"{name}-{int(time.time())}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        return {"name": name, "archived": archive, "path": str(target)}

    # -- curator hooks -----------------------------------------------------
    def archive(self, name: str) -> dict:
        return self.delete(name, archive=True, force=True)

    def usage_stats(self) -> dict:
        stats = {}
        for skill in self.list():
            stats[skill.name] = {"seed": skill.seed, "tags": skill.tags}
        return stats
