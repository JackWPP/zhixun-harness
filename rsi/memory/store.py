"""Bounded, curated, cross-run memory (Hermes-style MEMORY.md).

Design notes
------------
* The agent keeps a small ``MEMORY.md`` (and one ``TASK_<key>.md`` per task).
  Entries are separated by ``\\n§\\n`` and shown to the model as a *frozen
  snapshot* at the start of a task; writes during the run are persisted to disk
  immediately and echoed back through tool results, but never mutate the
  system prompt of the running session (this protects the LLM prefix cache and
  keeps behaviour predictable).
* Capacity is enforced hard: adding an entry above the limit is rejected and
  the current entries are returned so the model can curate.
* Writes are atomic (temp file + fsync + ``os.replace``) and serialized with a
  lock file. External edits are detected with a round-trip check; a ``.bak``
  copy is written and the write is refused instead of silently clobbering it.
* Content is scanned for prompt-injection / credential-exfiltration patterns
  because it is injected into the system prompt.
"""
from __future__ import annotations

import fcntl
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ENTRY_SEP = "\n§\n"
TARGETS = ("memory", "task")


class MemoryError_(Exception):
    """Structured memory failure (kept name short for JSON payloads)."""


@dataclass
class MemoryResult:
    ok: bool
    action: str
    target: str
    entries: list[str] = field(default_factory=list)
    usage: str = ""
    error: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        payload = {"ok": self.ok, "action": self.action, "target": self.target,
                   "entries": self.entries, "usage": self.usage,
                   "entry_count": len(self.entries)}
        if self.error:
            payload["error"] = self.error
        if self.note:
            payload["note"] = self.note
        return payload


# Patterns that must never reach the system prompt.  This is intentionally
# conservative and not a substitute for the platform's own defenses.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I),
    re.compile(r"system\s+prompt\s*:", re.I),
    re.compile(r"<\s*/?\s*(system|assistant|developer)\s*>", re.I),
    re.compile(r"(api[_-]?key|secret|password|token)\s*[=:]\s*\S{16,}", re.I),
    re.compile(r"(hidden|private)\s+(test|label|answer)s?", re.I),
    re.compile(r"(curl|wget)\s+https?://", re.I),
    re.compile(r"authorized_keys|\.ssh/", re.I),
]
_INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e]")


def scan_content(text: str) -> str | None:
    """Return a threat label if the text must not enter the prompt."""
    if _INVISIBLE.search(text):
        return "invisible_unicode"
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return f"pattern:{pattern.pattern[:60]}"
    return None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def parse_entries(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    entries = [item.strip() for item in text.split("§")]
    return [item for item in entries if item]


def serialize_entries(entries: list[str]) -> str:
    if not entries:
        return ""
    return ENTRY_SEP.join(entries) + "\n"


class MemoryStore:
    def __init__(self, root: str | Path, memory_limit: int = 4000,
                 task_limit: int = 2000):
        self.root = Path(root)
        self.memory_limit = int(memory_limit)
        self.task_limit = int(task_limit)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def _path(self, target: str, task_key: str | None = None) -> Path:
        if target == "memory":
            return self.root / "MEMORY.md"
        if target == "task":
            if not task_key:
                raise MemoryError_("task target requires a task key")
            return self.root / f"TASK_{task_key}.md"
        raise MemoryError_(f"unknown target: {target}")

    def limit_for(self, target: str) -> int:
        return self.memory_limit if target == "memory" else self.task_limit

    # -- io ----------------------------------------------------------------
    def _lock(self, path: Path):
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _read_raw(self, path: Path) -> str:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def _detect_drift(self, path: Path, raw: str) -> str | None:
        """Return backup path if the file was modified outside the tool."""
        if not raw.strip():
            return None
        parsed = parse_entries(raw)
        roundtrip = serialize_entries(parsed)
        limit = self.memory_limit if path.name == "MEMORY.md" else self.task_limit
        bad = raw != roundtrip
        bad = bad or any(len(entry) > limit for entry in parsed)
        if not bad:
            return None
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        try:
            backup.write_text(raw, encoding="utf-8")
        except OSError:
            return None
        return str(backup)

    def _write(self, path: Path, entries: list[str]) -> None:
        atomic_write_text(path, serialize_entries(entries))

    def entries(self, target: str = "memory", task_key: str | None = None) -> list[str]:
        return parse_entries(self._read_raw(self._path(target, task_key)))

    def usage(self, target: str = "memory", task_key: str | None = None) -> str:
        entries = self.entries(target, task_key)
        used = len(serialize_entries(entries))
        limit = self.limit_for(target)
        pct = int(round(100 * used / limit)) if limit else 0
        return f"{pct}% — {used}/{limit} chars"

    def read(self, target: str = "memory", task_key: str | None = None) -> str:
        return self._read_raw(self._path(target, task_key))

    # -- mutations ---------------------------------------------------------
    def _mutate(self, action: str, target: str, task_key: str | None,
                mutate) -> MemoryResult:
        path = self._path(target, task_key)
        limit = self.limit_for(target)
        with self._lock(path):
            raw = self._read_raw(path)
            backup = self._detect_drift(path, raw)
            if backup:
                return MemoryResult(False, action, target, parse_entries(raw),
                                    self.usage(target, task_key),
                                    error=(f"external drift detected; backup at {backup}; "
                                           "refusing to overwrite, integrate entries via add()"))
            entries = parse_entries(raw)
            try:
                new_entries, note = mutate(entries)
            except MemoryError_ as exc:
                return MemoryResult(False, action, target, entries,
                                    self.usage(target, task_key), error=str(exc))
            if new_entries is not None:
                payload = serialize_entries(new_entries)
                if len(payload) > limit:
                    return MemoryResult(
                        False, action, target, entries, self.usage(target, task_key),
                        error=(f"target {target} at {len(payload)}/{limit} chars; "
                               "replace or remove entries first"))
                self._write(path, new_entries)
                entries = new_entries
            return MemoryResult(True, action, target, entries,
                                self.usage(target, task_key), note=note)

    def add(self, content: str, target: str = "memory",
            task_key: str | None = None) -> MemoryResult:
        content = (content or "").strip()
        if not content:
            return MemoryResult(False, "add", target, self.entries(target, task_key),
                                self.usage(target, task_key), error="empty content")
        threat = scan_content(content)
        if threat:
            return MemoryResult(False, "add", target, self.entries(target, task_key),
                                self.usage(target, task_key),
                                error=f"content blocked by memory guard: {threat}")

        def mutate(entries):
            if content in entries:
                return None, "duplicate entry ignored"
            return entries + [content], None

        return self._mutate("add", target, task_key, mutate)

    def replace(self, old_text: str, content: str, target: str = "memory",
                task_key: str | None = None) -> MemoryResult:
        content = (content or "").strip()
        threat = scan_content(content)
        if threat:
            return MemoryResult(False, "replace", target, self.entries(target, task_key),
                                self.usage(target, task_key),
                                error=f"content blocked by memory guard: {threat}")

        def mutate(entries):
            matches = [i for i, entry in enumerate(entries) if old_text and old_text in entry]
            if not matches:
                raise MemoryError_(f"no entry contains {old_text!r}")
            if len(matches) > 1:
                raise MemoryError_(f"{old_text!r} matches {len(matches)} entries; be more specific")
            updated = list(entries)
            updated[matches[0]] = content
            return updated, None

        return self._mutate("replace", target, task_key, mutate)

    def remove(self, old_text: str, target: str = "memory",
               task_key: str | None = None) -> MemoryResult:
        def mutate(entries):
            matches = [i for i, entry in enumerate(entries) if old_text and old_text in entry]
            if not matches:
                raise MemoryError_(f"no entry contains {old_text!r}")
            if len(matches) > 1:
                raise MemoryError_(f"{old_text!r} matches {len(matches)} entries; be more specific")
            updated = [entry for i, entry in enumerate(entries) if i != matches[0]]
            return updated, None

        return self._mutate("remove", target, task_key, mutate)

    # -- prompt rendering --------------------------------------------------
    def render(self, target: str = "memory", task_key: str | None = None,
               title: str | None = None) -> str:
        entries = self.entries(target, task_key)
        if not entries:
            return ""
        header = title or ("MEMORY (project notes)" if target == "memory"
                           else f"TASK MEMORY ({task_key})")
        return (f"{'═' * 46}\n{header} [{self.usage(target, task_key)}]\n"
                f"{'═' * 46}\n" + ENTRY_SEP.join(entries))

    def snapshot(self, task_key: str | None = None) -> str:
        blocks = [self.render("memory", title="MEMORY (project notes)")]
        if task_key:
            blocks.append(self.render("task", task_key, title=f"TASK MEMORY ({task_key})"))
        return "\n\n".join(block for block in blocks if block)
