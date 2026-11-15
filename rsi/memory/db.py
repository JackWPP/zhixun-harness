"""SQLite state for cross-run memory, traces and strategy statistics.

* JSONL files (``records.jsonl`` / ``rounds.jsonl``) stay the audit source of
  truth; this database is a queryable index and can be rebuilt from them.
* WAL mode keeps the dashboard able to read while the runner writes.
* FTS5 backs ``search_records`` when available; otherwise a LIKE fallback is
  used so the harness runs on any stdlib SQLite build.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
    run_id TEXT PRIMARY KEY,
    started_at TEXT,
    finished_at TEXT,
    status TEXT,
    controller TEXT,
    llm_model TEXT,
    tasks TEXT,
    seed INTEGER,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS rounds(
    run_id TEXT,
    task TEXT,
    round_index INTEGER,
    decision TEXT,
    strategy TEXT,
    postprocess TEXT,
    primary_value REAL,
    delta REAL,
    hypothesis TEXT,
    change_summary TEXT,
    payload_json TEXT,
    created_at TEXT,
    PRIMARY KEY(run_id, task, round_index)
);
CREATE TABLE IF NOT EXISTS experiments(
    run_id TEXT,
    task TEXT,
    round_index INTEGER,
    exp_id TEXT,
    strategy TEXT,
    postprocess TEXT,
    ok INTEGER,
    primary_value REAL,
    delta REAL,
    cpu_seconds REAL,
    payload_json TEXT,
    PRIMARY KEY(run_id, task, exp_id)
);
CREATE TABLE IF NOT EXISTS llm_calls(
    run_id TEXT,
    task TEXT,
    role TEXT,
    ok INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    latency REAL,
    error TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS lessons(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT,
    scope TEXT,
    text TEXT,
    evidence TEXT,
    wins INTEGER DEFAULT 0,
    uses INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS lessons_unique ON lessons(task, text);
CREATE TABLE IF NOT EXISTS strategy_stats(
    task TEXT,
    strategy TEXT,
    postprocess_type TEXT,
    trials INTEGER DEFAULT 0,
    promotions INTEGER DEFAULT 0,
    mean_delta REAL DEFAULT 0,
    best_delta REAL DEFAULT 0,
    updated_at TEXT,
    PRIMARY KEY(task, strategy, postprocess_type)
);
CREATE TABLE IF NOT EXISTS skill_usage(
    skill TEXT,
    run_id TEXT,
    uses INTEGER DEFAULT 0,
    last_used TEXT,
    PRIMARY KEY(skill, run_id)
);
CREATE TABLE IF NOT EXISTS events(
    run_id TEXT,
    task TEXT,
    level TEXT,
    message TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS goals(
    session TEXT PRIMARY KEY,
    state_json TEXT,
    updated_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class StateDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self.has_fts = self._init_fts()
        self.conn.commit()

    def _init_fts(self) -> bool:
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5("
                "kind, ref UNINDEXED, task UNINDEXED, text)")
            return True
        except sqlite3.OperationalError:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS session_text("
                "kind TEXT, ref TEXT, task TEXT, text TEXT)")
            return False

    # -- low level ---------------------------------------------------------
    def _exec(self, sql: str, params=()):
        with self.conn:
            return self.conn.execute(sql, params)

    def _rows(self, sql: str, params=()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params))

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # -- runs --------------------------------------------------------------
    def add_run(self, run_id: str, started_at: str, controller: str = "",
                llm_model: str = "", tasks: str = "", seed: int = 42) -> None:
        self._exec(
            "INSERT INTO runs(run_id, started_at, status, controller, llm_model, tasks, seed) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET "
            "controller=excluded.controller, llm_model=excluded.llm_model, tasks=excluded.tasks",
            (run_id, started_at, "running", controller, llm_model, tasks, int(seed)))

    def finish_run(self, run_id: str, status: str, summary_json: str = "") -> None:
        self._exec("UPDATE runs SET finished_at=?, status=?, summary_json=? WHERE run_id=?",
                   (_now(), status, summary_json, run_id))

    def run_count(self) -> int:
        return int(self._rows("SELECT COUNT(*) AS n FROM runs")[0]["n"])

    def recent_run_ids(self, limit: int = 5) -> list[str]:
        return [row["run_id"] for row in self._rows(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT ?", (int(limit),))]

    # -- rounds / experiments ---------------------------------------------
    def add_round(self, run_id: str, task: str, round_index: int, decision: str,
                  strategy: str = "", postprocess: str = "", primary: float | None = None,
                  delta: float | None = None, hypothesis: str = "",
                  change_summary: str = "", payload: dict | None = None) -> None:
        self._exec(
            "INSERT INTO rounds(run_id, task, round_index, decision, strategy, postprocess, "
            "primary_value, delta, hypothesis, change_summary, payload_json, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id, task, round_index) DO UPDATE SET "
            "decision=excluded.decision, strategy=excluded.strategy, "
            "postprocess=excluded.postprocess, primary_value=excluded.primary_value, delta=excluded.delta, "
            "hypothesis=excluded.hypothesis, change_summary=excluded.change_summary, "
            "payload_json=excluded.payload_json",
            (run_id, task, int(round_index), decision, strategy, postprocess,
             primary, delta, hypothesis, change_summary, _json(payload or {}), _now()))

    def add_experiment(self, run_id: str, task: str, round_index: int, exp_id: str,
                       strategy: str = "", postprocess: str = "", ok: bool = False,
                       primary: float | None = None, delta: float | None = None,
                       cpu_seconds: float | None = None, payload: dict | None = None) -> None:
        self._exec(
            "INSERT INTO experiments(run_id, task, round_index, exp_id, strategy, postprocess, "
            "ok, primary_value, delta, cpu_seconds, payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, task, exp_id) DO UPDATE SET ok=excluded.ok, "
            "primary_value=excluded.primary_value, delta=excluded.delta, cpu_seconds=excluded.cpu_seconds, "
            "payload_json=excluded.payload_json",
            (run_id, task, int(round_index), exp_id, strategy, postprocess,
             int(bool(ok)), primary, delta, cpu_seconds, _json(payload or {})))

    def rounds(self, task: str | None = None, limit: int = 100) -> list[dict]:
        if task:
            rows = self._rows("SELECT * FROM rounds WHERE task=? ORDER BY round_index LIMIT ?",
                              (task, int(limit)))
        else:
            rows = self._rows("SELECT * FROM rounds ORDER BY task, round_index LIMIT ?",
                              (int(limit),))
        return [dict(row) for row in rows]

    # -- llm calls ---------------------------------------------------------
    def add_llm_call(self, run_id: str, task: str, role: str, ok: bool,
                     usage: dict | None = None, latency: float = 0.0,
                     error: str | None = None) -> None:
        usage = usage or {}
        self._exec(
            "INSERT INTO llm_calls(run_id, task, role, ok, input_tokens, output_tokens, "
            "reasoning_tokens, latency, error, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (run_id, task, role, int(bool(ok)), int(usage.get("input", 0) or 0),
             int(usage.get("output", 0) or 0), int(usage.get("reasoning", 0) or 0),
             float(latency or 0.0), error, _now()))

    def llm_totals(self, run_id: str | None = None) -> dict:
        if run_id:
            row = self._rows("SELECT COUNT(*) AS calls, COALESCE(SUM(input_tokens),0) AS i, "
                             "COALESCE(SUM(output_tokens),0) AS o, "
                             "COALESCE(SUM(reasoning_tokens),0) AS r, "
                             "COALESCE(SUM(ok),0) AS ok FROM llm_calls WHERE run_id=?",
                             (run_id,))[0]
        else:
            row = self._rows("SELECT COUNT(*) AS calls, COALESCE(SUM(input_tokens),0) AS i, "
                             "COALESCE(SUM(output_tokens),0) AS o, "
                             "COALESCE(SUM(reasoning_tokens),0) AS r, "
                             "COALESCE(SUM(ok),0) AS ok FROM llm_calls")[0]
        return {"calls": row["calls"], "input": row["i"], "output": row["o"],
                "reasoning": row["r"], "ok": row["ok"]}

    # -- lessons -----------------------------------------------------------
    def add_lesson(self, task: str, text: str, scope: str = "", evidence: str = "",
                   wins: int = 0) -> int:
        text = (text or "").strip()
        if not text:
            raise ValueError("lesson text is empty")
        now = _now()
        existing = self._rows("SELECT id FROM lessons WHERE task=? AND text=?", (task, text))
        if existing:
            lesson_id = int(existing[0]["id"])
            self._exec("UPDATE lessons SET wins=MAX(wins,?), updated_at=? WHERE id=?",
                       (int(wins), now, lesson_id))
            return lesson_id
        cursor = self._exec(
            "INSERT INTO lessons(task, scope, text, evidence, wins, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?)", (task, scope, text, evidence, int(wins), now, now))
        return int(cursor.lastrowid)

    def lessons(self, task: str, limit: int = 5, scope_contains: str | None = None) -> list[dict]:
        sql = "SELECT * FROM lessons WHERE task=?"
        params: list = [task]
        if scope_contains:
            sql += " AND (scope LIKE ? OR scope='')"
            params.append(f"%{scope_contains}%")
        sql += " ORDER BY wins DESC, updated_at DESC LIMIT ?"
        params.append(int(limit))
        return [dict(row) for row in self._rows(sql, tuple(params))]

    def bump_lessons(self, ids: list[int]) -> None:
        for lesson_id in ids:
            self._exec("UPDATE lessons SET uses=uses+1, updated_at=? WHERE id=?",
                       (_now(), int(lesson_id)))

    # -- strategy stats ----------------------------------------------------
    def record_strategy(self, task: str, strategy: str, postprocess_type: str,
                        delta: float | None, promoted: bool) -> None:
        key = (task, strategy, postprocess_type)
        row = self._rows("SELECT * FROM strategy_stats WHERE task=? AND strategy=? "
                         "AND postprocess_type=?", key)
        now = _now()
        if not row:
            trials = 1
            promotions = int(bool(promoted))
            mean_delta = float(delta or 0.0)
            best_delta = float(delta or 0.0)
            self._exec(
                "INSERT INTO strategy_stats(task, strategy, postprocess_type, trials, promotions, "
                "mean_delta, best_delta, updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (*key, trials, promotions, mean_delta, best_delta, now))
            return
        current = row[0]
        trials = int(current["trials"]) + 1
        promotions = int(current["promotions"]) + int(bool(promoted))
        mean_delta = ((float(current["mean_delta"]) * int(current["trials"]))
                      + float(delta or 0.0)) / trials
        best_delta = max(float(current["best_delta"]), float(delta or 0.0))
        self._exec("UPDATE strategy_stats SET trials=?, promotions=?, mean_delta=?, "
                   "best_delta=?, updated_at=? WHERE task=? AND strategy=? AND postprocess_type=?",
                   (trials, promotions, mean_delta, best_delta, now, *key))

    def strategy_stats(self, task: str, limit: int = 20) -> list[dict]:
        rows = self._rows("SELECT * FROM strategy_stats WHERE task=? "
                          "ORDER BY trials DESC, best_delta DESC LIMIT ?",
                          (task, int(limit)))
        result = []
        for row in rows:
            item = dict(row)
            trials = max(1, int(item["trials"]))
            item["win_rate"] = round(int(item["promotions"]) / trials, 4)
            result.append(item)
        return result

    # -- skills ------------------------------------------------------------
    def record_skill_use(self, skill: str, run_id: str = "") -> None:
        self._exec(
            "INSERT INTO skill_usage(skill, run_id, uses, last_used) VALUES(?,?,1,?) "
            "ON CONFLICT(skill, run_id) DO UPDATE SET uses=uses+1, last_used=excluded.last_used",
            (skill, run_id, _now()))

    def skill_uses(self, skill: str) -> int:
        row = self._rows("SELECT COALESCE(SUM(uses),0) AS n FROM skill_usage WHERE skill=?",
                         (skill,))[0]
        return int(row["n"])

    def skill_usage(self) -> list[dict]:
        return [dict(row) for row in self._rows(
            "SELECT skill, COALESCE(SUM(uses),0) AS uses, MAX(last_used) AS last_used "
            "FROM skill_usage GROUP BY skill ORDER BY uses DESC")]

    # -- events ------------------------------------------------------------
    def add_event(self, run_id: str, task: str, level: str, message: str) -> None:
        self._exec("INSERT INTO events(run_id, task, level, message, created_at) "
                   "VALUES(?,?,?,?,?)", (run_id, task, level, message, _now()))

    def events(self, run_id: str | None = None, limit: int = 100) -> list[dict]:
        if run_id:
            rows = self._rows("SELECT * FROM events WHERE run_id=? ORDER BY rowid DESC LIMIT ?",
                              (run_id, int(limit)))
        else:
            rows = self._rows("SELECT * FROM events ORDER BY rowid DESC LIMIT ?", (int(limit),))
        return [dict(row) for row in rows]

    # -- goals -------------------------------------------------------------
    def save_goal(self, session: str, state: dict) -> None:
        self._exec("INSERT INTO goals(session, state_json, updated_at) VALUES(?,?,?) "
                   "ON CONFLICT(session) DO UPDATE SET state_json=excluded.state_json, "
                   "updated_at=excluded.updated_at", (session, _json(state), _now()))

    def load_goal(self, session: str) -> dict | None:
        rows = self._rows("SELECT state_json FROM goals WHERE session=?", (session,))
        if not rows:
            return None
        try:
            return json.loads(rows[0]["state_json"])
        except (TypeError, ValueError):
            return None

    # -- full text search --------------------------------------------------
    def index(self, kind: str, ref: str, task: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self.has_fts:
            self._exec("INSERT INTO session_fts(kind, ref, task, text) VALUES(?,?,?,?)",
                       (kind, ref, task, text))
        else:
            self._exec("INSERT INTO session_text(kind, ref, task, text) VALUES(?,?,?,?)",
                       (kind, ref, task, text))

    def search(self, query: str, limit: int = 20) -> list[dict]:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", query or "")[:10]
        if not tokens:
            return []
        if self.has_fts:
            match = " ".join(f'"{token}"*' for token in tokens)
            try:
                rows = self._rows(
                    "SELECT kind, ref, task, snippet(session_fts, 3, '[', ']', '…', 12) AS text "
                    "FROM session_fts WHERE session_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, int(limit)))
                return [dict(row) for row in rows]
            except sqlite3.OperationalError:
                pass
        source = "session_fts" if self.has_fts else "session_text"
        clauses = " OR ".join("text LIKE ?" for _ in tokens)
        rows = self._rows(f"SELECT kind, ref, task, substr(text,1,240) AS text FROM {source} "
                          f"WHERE {clauses} LIMIT ?",
                          tuple(f"%{token}%" for token in tokens) + (int(limit),))
        return [dict(row) for row in rows]

    # -- rebuild from JSONL ------------------------------------------------
    def replay_from_jsonl(self, rounds_path: str | Path) -> int:
        path = Path(rounds_path)
        if not path.is_file():
            return 0
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("type") != "round" and "round_index" not in record:
                continue
            self.add_round(record.get("run_id", ""), record.get("task", ""),
                           record.get("round_index", record.get("index", 0)),
                           record.get("decision", ""), record.get("strategy", ""),
                           record.get("postprocess", ""), record.get("primary"),
                           record.get("delta"), record.get("hypothesis", ""),
                           record.get("change_summary", ""), record)
            count += 1
        return count
