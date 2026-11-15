"""Run records: append-only JSONL audit trail + atomic dashboard state.

Two artifacts are produced under the run directory:

``records.jsonl``
    one line per event / per completed round. Rounds carry the platform's
    workspace-report fields (hypothesis, change summary, metrics, decision,
    parent/candidate checkpoint digests, history ids).
``state.json``
    the live snapshot consumed by ``rsi.dashboard`` (schema v1), written
    atomically via a temporary file + ``os.replace``.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utcnow_iso(dt=None) -> str:
    return (dt or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class Recorder:
    def __init__(self, state_dir, run_id: str):
        self.state_dir = Path(state_dir)
        self.run_id = run_id
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.state_dir / "records.jsonl"
        self.state_path = self.state_dir / "state.json"
        self.rounds_path = self.state_dir / "rounds.jsonl"

    # -- event / record stream -------------------------------------------------
    def _append(self, record: dict) -> dict:
        record = {"ts": utcnow_iso(), **record}
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def event(self, level: str, task: str, message: str, **extra) -> dict:
        return self._append({"type": "event", "run_id": self.run_id, "level": level,
                             "task": task, "message": message, **extra})

    def round(self, record: dict) -> dict:
        record = {"type": "round", "run_id": self.run_id, **record}
        saved = self._append(record)
        with self.rounds_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return saved

    def llm_call(self, task: str, role: str, usage: dict, **extra) -> dict:
        return self._append({"type": "llm_call", "run_id": self.run_id, "task": task,
                             "role": role, "usage": usage, **extra})

    def panel(self, task: str, entries, **extra) -> dict:
        return self._append({"type": "panel", "run_id": self.run_id, "task": task,
                             "entries": entries, **extra})

    # -- state -----------------------------------------------------------------
    def write_state(self, state: dict) -> None:
        atomic_write_json(self.state_path, state)

    def read_rounds(self, task: str | None = None) -> list[dict]:
        if not self.rounds_path.is_file():
            return []
        rounds = []
        for line in self.rounds_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if task is None or record.get("task") == task:
                rounds.append(record)
        return rounds

    def read_records(self) -> list[dict]:
        if not self.records_path.is_file():
            return []
        return [json.loads(line) for line in
                self.records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
