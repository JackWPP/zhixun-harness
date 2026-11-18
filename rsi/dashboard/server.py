"""智训引擎 RSI Harness 本地看板：零依赖 HTTP 服务端。

设计要点
--------
* 只依赖 Python 标准库（http.server / json），HTTP/1.1 长连接；
* 每次 ``GET /api/state`` 都检查 state.json 的 mtime+size，变化才重新解析，
  所以 harness 用 ``os.replace`` 原子写文件时看板可以安全地每 2s 轮询；
* 若与 state.json 同目录存在 ``records.jsonl``（或显式 ``--records``），
  会把其中的事件/决策记录合并进事件流（与 state.events 去重）；
* 静态文件按白名单读取，不提供目录浏览，也没有任何外部资源依赖。

对外路由
--------
GET /                 -> static/index.html
GET /app.js           -> 前端脚本
GET /style.css        -> 样式
GET /favicon.svg      -> 图标
GET /api/state        -> 完整 state（schema v2 + 附加 ``_meta``）
GET /api/events       -> 事件流，支持 limit / level / task / q
GET /api/memory       -> 运行/任务记忆、lessons、策略统计、技能使用、记忆文件
GET /api/agent        -> 自主内核统计、目标、每轮 agent 轨迹（?task=A 可只看单任务）
GET /api/records/search -> 在 records.jsonl / rounds.jsonl 中做大小写不敏感子串搜索
GET /api/goal         -> schema v3 软目标 / 批次时间线（GOAL 缺失时降级为空结构）
GET /api/knowledge    -> schema v3 知识命中 / 技能使用 / lessons / 策略统计
GET /api/health       -> 轻量健康检查
其他                  -> 404（JSON）

记忆/内核/软目标接口全部只读且容错：字段缺失、records 文件不存在、
sqlite 打不开时返回空结构或降级说明（v1 / v2 state 同样可用），不会让页面报错。
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

VERSION = "1.2.0"
SCHEMA_VERSION = 3

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "favicon.svg": "image/svg+xml",
}
VALID_LEVELS = ("info", "warn", "error")
MAX_RECORDS_BYTES = 16 * 1024 * 1024
DEFAULT_EVENTS_LIMIT = 50
MAX_EVENTS_LIMIT = 500
DEFAULT_SEARCH_LIMIT = 20
MAX_SEARCH_LIMIT = 100
MAX_SEARCH_LINES = 20_000
MAX_MEMORY_FILE_BYTES = 1024 * 1024
MAX_TRACE_DETAIL = 400
DB_TIMEOUT_SECONDS = 0.35
PROPOSAL_SOURCE_KEYS = ("seed", "agent", "knowledge", "panel")
SOFT_GOAL_NOTE = "软目标，不阻塞完成"


class StateUnavailable(Exception):
    """state.json 不存在或暂不可读。"""

    def __init__(self, kind: str, detail: str, path: Path | None = None):
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.path = path


# --------------------------------------------------------------------------- #
# 时间 / 记录归一化
# --------------------------------------------------------------------------- #
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ts_sort_key(value: object) -> tuple:
    """ISO8601 字符串 -> 可比较元组；非法值排到最后。"""
    if not isinstance(value, str) or not value:
        return (0, "")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (1, dt.timestamp())
    except (ValueError, OverflowError):
        return (0, value)


def _normalise_event(raw: object) -> dict | None:
    """把任意来源（state.events / records.jsonl）的一条记录归一成事件。"""
    if not isinstance(raw, dict):
        return None
    rtype = str(raw.get("type") or raw.get("kind") or "event")
    ts = raw.get("ts") or raw.get("created_at") or raw.get("timestamp") or ""
    task = raw.get("task", raw.get("task_key", ""))
    task = "" if task is None else str(task)

    level = raw.get("level")
    if level not in VALID_LEVELS:
        level = {"error": "error", "warn": "warn", "warning": "warn"}.get(rtype, "info")

    message = raw.get("message") or raw.get("msg") or raw.get("event")
    if not isinstance(message, str) or not message.strip():
        if rtype in ("round", "decision"):
            parts = []
            if task:
                parts.append(f"任务 {task}")
            if raw.get("round_index") is not None:
                parts.append(f"第 {raw['round_index']} 轮")
            if raw.get("decision"):
                parts.append(f"决策 {raw['decision']}")
            if raw.get("strategy"):
                parts.append(f"策略 {raw['strategy']}")
            message = " · ".join(parts) + "（来自 records.jsonl）"
        elif rtype == "run_start":
            message = f"run 开始：{raw.get('run_id', '')}".strip()
        else:
            message = f"{rtype} 记录"
    return {
        "ts": ts if isinstance(ts, str) else str(ts),
        "level": level,
        "task": task,
        "message": message.strip(),
        "source": "records.jsonl",
    }


def _dedupe_events(events: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for event in events:
        key = (event.get("ts"), event.get("level"), event.get("task"), event.get("message"))
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


# --------------------------------------------------------------------------- #
# 记忆 / 内核数据归一化
# --------------------------------------------------------------------------- #
def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _truncate(text: object, limit: int) -> str:
    value = "" if text is None else str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _iso_or_empty(value: object) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _normalise_lesson(raw: object, *, source: str) -> dict | None:
    """把 db 行 / state.review.lessons 归一成 {task,text,wins,...}。"""
    if not isinstance(raw, dict):
        return None
    text = raw.get("text") or raw.get("lesson") or raw.get("message")
    if not isinstance(text, str) or not text.strip():
        return None
    evidence = raw.get("evidence")
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError:
            pass
    return {
        "task": str(raw.get("task") or "").upper() or "—",
        "scope": _truncate(raw.get("scope") or "", 120),
        "text": _truncate(text.strip(), 1200),
        "wins": _safe_int(raw.get("wins")),
        "uses": _safe_int(raw.get("uses")),
        "evidence": evidence if isinstance(evidence, (dict, list)) else (evidence if isinstance(evidence, str) else {}),
        "source": source,
    }


def _normalise_strategy_row(raw: object, fallback_task: str = "") -> dict | None:
    if not isinstance(raw, dict):
        return None
    trials = _safe_int(raw.get("trials"))
    promotions = _safe_int(raw.get("promotions"))
    win_rate = raw.get("win_rate")
    try:
        win_rate = float(win_rate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        win_rate = (promotions / trials) if trials > 0 else None
    if win_rate is not None:
        win_rate = max(0.0, min(1.0, win_rate))
    return {
        "task": str(raw.get("task") or fallback_task).upper(),
        "strategy": str(raw.get("strategy") or "unknown"),
        "postprocess_type": str(raw.get("postprocess_type") or raw.get("postprocess") or "none"),
        "trials": trials,
        "promotions": promotions,
        "mean_delta": raw.get("mean_delta"),
        "best_delta": raw.get("best_delta"),
        "win_rate": win_rate,
        "updated_at": _iso_or_empty(raw.get("updated_at")),
    }


# -- schema v3 归一化 helpers ------------------------------------------------
def _as_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalise_metric_map(raw: object) -> dict:
    """把 {metric: value} 归一为 float；非法值与非字符串键直接忽略。"""
    result: dict[str, float] = {}
    for key, value in _as_dict(raw).items():
        number = _as_float(value)
        if isinstance(key, str) and key.strip() and number is not None:
            result[key.strip()] = number
    return result


def _normalise_docs(raw: object) -> list[dict]:
    """归一知识文档：兼容字符串与 {title,path,hits}。"""
    docs: list[dict] = []
    for item in _as_list(raw):
        if isinstance(item, str):
            title = item.strip()
            if title:
                docs.append({"title": _truncate(title, 400), "path": "", "hits": 0})
        elif isinstance(item, dict):
            title = item.get("title") or item.get("name") or item.get("path") or item.get("doc")
            if title is None:
                continue
            docs.append(
                {
                    "title": _truncate(title, 400),
                    "path": _truncate(item.get("path") or item.get("file") or "", 600),
                    "hits": _safe_int(item.get("hits") or item.get("uses") or item.get("count")),
                }
            )
    return docs[:50]


def _normalise_goal_soft(task: dict) -> dict:
    """读取 task.goal_soft（兼容 task.goal / 平铺写法），缺字段时给出可展示的降级结构。"""
    raw = _as_dict(task.get("goal_soft"))
    goal = _as_dict(task.get("goal"))
    metric = str(task.get("primary_metric") or "")
    direction = str(task.get("direction") or "lower")
    targets = _normalise_metric_map(raw.get("targets"))
    current = _normalise_metric_map(raw.get("current"))

    if not targets:
        target = _as_float(raw.get("target"))
        if target is not None and metric:
            targets = {metric: target}
    if not current:
        value = _as_float(raw.get("current_value"))
        if value is None:
            value = _as_float(task.get("current"))
        if value is not None and metric:
            current = {metric: value}

    gap = _as_float(raw.get("gap"))
    gap_metric = metric or next(iter(targets), "") or next(iter(current), "")
    if gap is None and gap_metric in targets and gap_metric in current:
        # 统一约定：gap > 0 表示距软目标还有缺口（未达标）。
        if direction == "higher":
            gap = targets[gap_metric] - current[gap_metric]
        else:
            gap = current[gap_metric] - targets[gap_metric]
    if gap is not None:
        gap = round(gap, 6)

    degraded_raw = raw.get("degraded")
    if isinstance(degraded_raw, str):
        degraded = degraded_raw.strip().lower() in {"1", "true", "yes", "degraded"}
    else:
        degraded = bool(degraded_raw) if degraded_raw is not None else False

    verdict = raw.get("verdict") or goal.get("last_verdict") or ""
    evidence = raw.get("evidence")
    if isinstance(evidence, (dict, list)):
        evidence_text = json.dumps(evidence, ensure_ascii=False)
    else:
        evidence_text = "" if evidence is None else str(evidence)

    return {
        "available": bool(raw) or bool(goal),
        "targets": targets,
        "current": current,
        "gap": gap,
        "metric": gap_metric,
        "direction": direction,
        "degraded": degraded,
        "verdict": str(verdict or ""),
        "evidence": _truncate(evidence_text, 600),
        "note": SOFT_GOAL_NOTE,
    }


def _normalise_batches(data: dict) -> list[dict]:
    """归一 state.batches（含批前/批后指标与决策），缺字段时返回空列表。"""
    task_metrics = {
        str(_as_dict(task).get("key") or ""): str(_as_dict(task).get("primary_metric") or "")
        for task in _as_list(data.get("tasks"))
    }
    batches: list[dict] = []
    for position, raw in enumerate(_as_list(data.get("batches"))):
        item = _as_dict(raw)
        task_key = str(item.get("task") or "").upper()
        before = _normalise_metric_map(item.get("before_metrics"))
        after = _normalise_metric_map(item.get("after_metrics"))
        fallback_metric = task_metrics.get(task_key, "") or "metric"
        if not before:
            single = _as_float(item.get("before"))
            if single is not None:
                before = {fallback_metric: single}
        if not after:
            single = _as_float(item.get("after"))
            if single is not None:
                after = {fallback_metric: single}
        batches.append(
            {
                "index": _safe_int(item.get("index"), position),
                "task": task_key,
                "before_metrics": before,
                "after_metrics": after,
                "decision": item.get("decision"),
                "note": _truncate(item.get("note") or item.get("reason") or "", 300),
            }
        )
    batches.sort(key=lambda entry: entry["index"])
    return batches


def _normalise_proposal_sources(raw: object) -> dict:
    """归一 round.agent.proposal_sources；缺失时返回空 dict 供前端降级。"""
    source = _as_dict(raw)
    if not source:
        return {}
    return {key: _safe_int(source.get(key)) for key in PROPOSAL_SOURCE_KEYS}


def _normalise_drift(task: dict) -> dict:
    """合并 drift / drift_extra，输出雷达所需的数值与归因（缺失时全部降级）。"""
    drift = _as_dict(task.get("drift"))
    extra = _as_dict(task.get("drift_extra"))
    merged = {**extra, **drift}
    attribution = [str(item) for item in _as_list(merged.get("attribution")) if str(item).strip()][:12]
    causes = [str(item) for item in _as_list(merged.get("causes")) if str(item).strip()][:12]
    return {
        "type": str(merged.get("type") or ""),
        "severity": str(merged.get("severity") or "unknown"),
        "score": _as_float(merged.get("score")),
        "feature_shift": _as_float(merged.get("feature_shift")),
        "ks_max": _as_float(merged.get("ks_max")),
        "attribution": attribution,
        "causes": causes,
        "evidence": _truncate(merged.get("evidence") or "", 600),
    }


def _normalise_knowledge(data: dict, skill_rows: list[dict], memory_files: list[dict]) -> dict:
    """合并 state.knowledge（hits/docs/skills_used）与 sqlite 技能统计。"""
    raw = _as_dict(data.get("knowledge"))
    hits_raw = raw.get("hits")
    if isinstance(hits_raw, list):
        hits = len(hits_raw)
    elif isinstance(hits_raw, dict):
        total = hits_raw.get("total")
        hits = _safe_int(total) if total is not None else sum(_safe_int(value) for value in hits_raw.values())
    else:
        hits = _safe_int(hits_raw)

    docs = _normalise_docs(raw.get("docs"))
    if not docs:
        docs = [{"title": item.get("name", ""), "path": item.get("path", ""), "hits": 0} for item in memory_files]

    name_counts: dict[str, int] = {}
    for item in _as_list(raw.get("skills_used")):
        if isinstance(item, str):
            name = item.strip()
            if name:
                name_counts[name] = name_counts.get(name, 0) + 1
        elif isinstance(item, dict):
            name = str(item.get("skill") or item.get("name") or "").strip()
            if name:
                name_counts[name] = name_counts.get(name, 0) + max(1, _safe_int(item.get("uses") or item.get("count"), 1))
    for row in skill_rows:
        name = str(_as_dict(row).get("skill") or "").strip()
        if name and name not in name_counts:
            name_counts[name] = _safe_int(_as_dict(row).get("uses"))

    skills_used = [
        {"skill": name, "uses": uses}
        for name, uses in sorted(name_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
    return {
        "available": bool(raw) or bool(docs) or bool(skills_used),
        "hits": hits,
        "docs": docs,
        "skills_used": skills_used,
        "hits_detail": _truncate(raw.get("hits_detail") or "", 600),
    }


# --------------------------------------------------------------------------- #
# state / records 读取（带 mtime 缓存）
# --------------------------------------------------------------------------- #
class StateStore:
    """线程安全地读取 state.json，并按 mtime 缓存解析结果。"""

    def __init__(self, state_path: str | Path, records_path: str | Path | None = None,
                 follow_dir: str | Path | None = None):
        self.state_path = Path(state_path).expanduser().resolve()
        self.follow_dir = Path(follow_dir).expanduser().resolve() if follow_dir else None
        self._explicit_records = Path(records_path).expanduser().resolve() if records_path else None
        if self._explicit_records:
            self.records_path = self._explicit_records
        else:
            self.records_path = self.state_path.parent / "records.jsonl"
        self._lock = threading.Lock()
        self._state_sig: tuple | None = None
        self._state_cache: dict | None = None
        self._last_good: dict | None = None
        self._records_sig: tuple | None = None
        self._records_cache: list[dict] = []
        self._records_skipped: int = 0
        # 搜索缓存：path -> (sig, items, skipped_lines, available)
        self._search_lock = threading.Lock()
        self._search_cache: dict[Path, tuple] = {}
        # sqlite 缓存：(path, mtime, size) -> 解析结果
        self._db_lock = threading.Lock()
        self._db_sig: tuple | None = None
        self._db_cache: dict | None = None
        self._db_path_used: Path | None = None

    # -- 内部 ---------------------------------------------------------------
    def _maybe_follow(self) -> None:
        """实时跟随运行根目录下最新的 run（`<dir>/*/state.json`）。

        harness 每次 run 会新建目录；看板常驻时自动切换到最新 run，
        无需重启即可看到下一次实验的进度。
        """
        if self.follow_dir is None:
            return
        try:
            candidates = sorted(
                (p for p in self.follow_dir.glob("*/state.json") if p.is_file()),
                key=lambda p: p.stat().st_mtime_ns,
            )
        except OSError:
            return
        if not candidates:
            return
        latest = candidates[-1].resolve()
        if latest == self.state_path:
            return
        with self._lock:
            self.state_path = latest
            if self._explicit_records is None:
                self.records_path = latest.parent / "records.jsonl"
            self._state_sig = None
            self._state_cache = None
            self._records_sig = None
            self._search_cache.clear()
            print(f"[dashboard] follow -> {latest}")

    def _stat_sig(self, path: Path) -> tuple | None:
        try:
            info = path.stat()
        except OSError:
            return None
        return (info.st_mtime_ns, info.st_size)

    def _load_state(self) -> dict:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise StateUnavailable("missing", "state 文件不存在", self.state_path) from exc
        except OSError as exc:
            raise StateUnavailable("unreadable", f"state 文件无法读取：{exc}", self.state_path) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StateUnavailable(
                "invalid", f"state 文件不是合法 JSON（line {exc.lineno} col {exc.colno}）", self.state_path
            ) from exc
        if not isinstance(data, dict):
            raise StateUnavailable("invalid", "state 根节点必须是 JSON object", self.state_path)
        return data

    def _load_records(self) -> None:
        sig = self._stat_sig(self.records_path)
        if sig == self._records_sig:
            return
        self._records_sig = sig
        self._records_cache = []
        self._records_skipped = 0
        if sig is None or sig[1] > MAX_RECORDS_BYTES:
            return
        events: list[dict] = []
        try:
            with self.records_path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        self._records_skipped += 1
                        continue
                    event = _normalise_event(raw)
                    if event is not None:
                        events.append(event)
        except OSError:
            return
        self._records_cache = events

    # -- 对外 ---------------------------------------------------------------
    def read(self, force: bool = False) -> dict:
        self._maybe_follow()
        with self._lock:
            state_sig = self._stat_sig(self.state_path)
            cache_hit = not force and state_sig is not None and state_sig == self._state_sig and self._state_cache
            if cache_hit:
                data = self._state_cache
            else:
                try:
                    data = self._load_state()
                except StateUnavailable as exc:
                    if self._last_good is not None:
                        self._attach_meta(
                            self._last_good,
                            cache="degraded",
                            state_sig=state_sig or (0, 0),
                            error=f"{exc.detail}；正在使用最近一次成功读取的快照",
                        )
                        return self._last_good
                    raise
                self._state_sig = state_sig
                self._state_cache = data
                self._last_good = data

            self._load_records()
            return self._attach_meta(data, cache="hit" if cache_hit else "miss", state_sig=state_sig or (0, 0))

    def _attach_meta(self, data: dict, *, cache: str, state_sig: tuple, error: str | None = None) -> dict:
        state_events = data.get("events")
        if not isinstance(state_events, list):
            state_events = []
        merged = list(state_events) + list(self._records_cache)
        events = _dedupe_events(merged)
        for event in state_events:
            if isinstance(event, dict):
                event.setdefault("source", "state.json")

        payload = dict(data)
        payload["events"] = events
        payload["ok"] = True
        # 运行中的 run：按 wall clock 重算 elapsed，避免状态文件只在轮末刷新
        # 时看板上的耗时"卡住"。只影响展示，不回写 state。
        run = payload.get("run")
        if isinstance(run, dict) and run.get("status") in {"running", "queued"}:
            started = run.get("started_at")
            if isinstance(started, str):
                try:
                    begin = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    if begin.tzinfo is None:
                        begin = begin.replace(tzinfo=timezone.utc)
                    payload["run"] = dict(run)
                    try:
                        reported = float(run.get("elapsed_seconds") or 0.0)
                    except (TypeError, ValueError):
                        reported = 0.0
                    payload["run"]["elapsed_seconds"] = round(
                        max(reported,
                            max(0.0, (datetime.now(timezone.utc) - begin).total_seconds())), 1)
                except ValueError:
                    pass
        mtime_ns = state_sig[0]
        stale = None
        if mtime_ns:
            stale = max(0.0, time.time() - mtime_ns / 1e9)
        payload["_meta"] = {
            "dashboard_version": VERSION,
            "generated_at": _utcnow_iso(),
            "state_path": str(self.state_path),
            "records_path": str(self.records_path),
            "records_available": self._records_sig is not None,
            "records_events": len(self._records_cache),
            "records_skipped_lines": self._records_skipped,
            "event_count": len(events),
            "schema_version": data.get("schema_version", SCHEMA_VERSION),
            "stale_seconds": None if stale is None else round(stale, 1),
            "cache": cache,
        }
        if error:
            payload["_meta"]["error"] = error
        return payload

    # -- 路径解析 -----------------------------------------------------------
    def _resolve_input_path(self, raw: object, *, want_dir: bool) -> Path | None:
        """把 state 里的相对路径（相对 run 目录 / 上级 / 当前目录）解析成真实路径。"""
        if not isinstance(raw, str) or not raw.strip():
            return None
        candidate = Path(raw).expanduser()
        options: list[Path] = []
        if candidate.is_absolute():
            options.append(candidate)
        else:
            for base in (self.state_path.parent, self.state_path.parent.parent, Path.cwd()):
                options.append(base / candidate)
        for option in options:
            try:
                if option.is_dir() if want_dir else option.is_file():
                    return option.resolve()
            except OSError:
                continue
        return None

    def _resolve_memory_db(self, data: dict) -> Path | None:
        run = _as_dict(data.get("run"))
        memory = _as_dict(run.get("memory"))
        resolved = self._resolve_input_path(memory.get("db"), want_dir=False)
        if resolved is not None:
            return resolved
        for candidate in (self.state_path.parent / "rsi_state.db", self.state_path.parent.parent / "rsi_state.db"):
            try:
                if candidate.is_file():
                    return candidate.resolve()
            except OSError:
                continue
        return None

    def _memory_files(self, data: dict) -> list[dict]:
        run = _as_dict(data.get("run"))
        memory = _as_dict(run.get("memory"))
        memory_dir = self._resolve_input_path(memory.get("dir"), want_dir=True)
        if memory_dir is None:
            for candidate in (self.state_path.parent / "memory", self.state_path.parent.parent / "memory"):
                try:
                    if candidate.is_dir():
                        memory_dir = candidate.resolve()
                        break
                except OSError:
                    continue
        if memory_dir is None:
            return []
        try:
            paths = sorted(p for p in memory_dir.glob("*.md") if p.is_file())
        except OSError:
            return []
        files: list[dict] = []
        for path in paths[:40]:
            try:
                raw = path.read_bytes()[:MAX_MEMORY_FILE_BYTES]
            except OSError:
                continue
            text = raw.decode("utf-8", "replace")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            bullets = [
                line for line in lines
                if line.startswith(("- ", "* "))
                or (len(line) > 2 and line[0].isdigit() and line[1] in ".)")
            ]
            body = [line for line in lines if not line.startswith("#")]
            entries = len(bullets) if bullets else len(body)
            files.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "entries": entries,
                    "characters": len(text),
                    "usage": f"{len(text):,} chars",
                }
            )
        return files

    # -- sqlite（只读 URI，失败降级为空）-----------------------------------
    def _read_db(self, db_path: Path | None) -> dict:
        empty = {"available": False, "error": "", "lessons": [], "strategy_stats": [], "skill_usage": []}
        if db_path is None:
            return dict(empty, error="未找到 rsi_state.db")
        try:
            info = db_path.stat()
        except OSError as exc:
            return dict(empty, error=f"db 文件不可读：{exc}")
        sig = (str(db_path), info.st_mtime_ns, info.st_size)
        with self._db_lock:
            if sig == self._db_sig and self._db_cache is not None:
                return self._db_cache
            result = dict(empty)
            try:
                con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=DB_TIMEOUT_SECONDS)
            except sqlite3.Error as exc:
                result["error"] = f"sqlite 打不开：{exc}"
            else:
                try:
                    con.row_factory = sqlite3.Row
                    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

                    def columns(table: str) -> set[str]:
                        return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}

                    if "lessons" in tables:
                        cols = columns("lessons")
                        wanted = [c for c in ("id", "task", "scope", "text", "evidence", "wins", "uses", "created_at", "updated_at") if c in cols]
                        if "text" in cols and "task" in cols:
                            sql = f"SELECT {', '.join(wanted)} FROM lessons ORDER BY wins DESC, id DESC LIMIT 300"
                            for row in con.execute(sql):
                                lesson = _normalise_lesson({key: row[key] for key in row.keys()}, source="db")
                                if lesson:
                                    result["lessons"].append(lesson)
                    if "strategy_stats" in tables:
                        cols = columns("strategy_stats")
                        wanted = [c for c in ("task", "strategy", "postprocess_type", "trials", "promotions", "mean_delta", "best_delta", "updated_at") if c in cols]
                        if "task" in cols and "strategy" in cols:
                            sql = f"SELECT {', '.join(wanted)} FROM strategy_stats ORDER BY trials DESC LIMIT 300"
                            for row in con.execute(sql):
                                item = _normalise_strategy_row({key: row[key] for key in row.keys()})
                                if item:
                                    result["strategy_stats"].append(item)
                    if "skill_usage" in tables:
                        cols = columns("skill_usage")
                        if "skill" in cols and "uses" in cols:
                            last = ", MAX(last_used) AS last_used" if "last_used" in cols else ""
                            sql = f"SELECT skill, SUM(uses) AS uses{last} FROM skill_usage GROUP BY skill ORDER BY uses DESC LIMIT 30"
                            for row in con.execute(sql):
                                result["skill_usage"].append(
                                    {
                                        "skill": str(row["skill"]),
                                        "uses": _safe_int(row["uses"]),
                                        "last_used": _iso_or_empty(row["last_used"]) if "last_used" in row.keys() else "",
                                    }
                                )
                    result["available"] = True
                except sqlite3.Error as exc:
                    result["error"] = f"sqlite 读取失败：{exc}"
                finally:
                    con.close()
            self._db_sig = sig
            self._db_cache = result
            self._db_path_used = db_path
            return result

    # -- 聚合接口 -----------------------------------------------------------
    def memory_payload(self) -> dict:
        data = self.read()
        run = _as_dict(data.get("run"))
        run_agent = _as_dict(run.get("agent"))
        run_memory = _as_dict(run.get("memory"))
        db_path = self._resolve_memory_db(data)
        db = self._read_db(db_path)

        tasks_out: list[dict] = []
        fallback_lessons: list[dict] = []
        state_strategy: list[dict] = []
        for raw_task in _as_list(data.get("tasks")):
            task = _as_dict(raw_task)
            key = str(task.get("key") or "")
            memory = _as_dict(task.get("memory"))
            goal = _as_dict(task.get("goal"))
            rows: list[dict] = []
            for raw_row in _as_list(task.get("strategy_stats")):
                row = _normalise_strategy_row(raw_row, key)
                if row:
                    rows.append(row)
                    state_strategy.append(row)
            review = _as_dict(memory.get("review"))
            for raw_lesson in _as_list(review.get("lessons")):
                lesson = _normalise_lesson(raw_lesson, source="state")
                if lesson:
                    fallback_lessons.append(lesson)
            tasks_out.append(
                {
                    "key": key,
                    "name": str(task.get("name") or task.get("name_en") or key),
                    "controller": task.get("controller") or run_agent.get("controller") or "pipeline",
                    "status": task.get("status"),
                    "round_index": task.get("round_index"),
                    "rounds_total": task.get("rounds_total"),
                    "memory": memory,
                    "goal": goal,
                    "agent": _as_dict(task.get("agent")),
                    "strategy_stats": rows,
                }
            )

        lessons: list[dict] = []
        seen: set[str] = set()
        for lesson in list(db.get("lessons") or []) + fallback_lessons:
            fingerprint = lesson["text"].strip()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            lessons.append(lesson)

        strategy_stats = state_strategy or list(db.get("strategy_stats") or [])
        memory_files = self._memory_files(data)

        memory = {
            "enabled": run_memory.get("enabled", True),
            "dir": run_memory.get("dir") or (str(self.state_path.parent / "memory")),
            "db": run_memory.get("db") or (str(db_path) if db_path else ""),
            "entries": _safe_int(run_memory.get("entries")),
            "usage": run_memory.get("usage"),
        }
        if not memory["entries"] and memory_files:
            memory["entries"] = sum(_safe_int(item.get("entries")) for item in memory_files)
        usage_text = str(memory.get("usage") or "")
        if memory["entries"] and (not usage_text.strip() or usage_text.startswith("0% — 0/")):
            total_chars = sum(_safe_int(item.get("characters")) for item in memory_files)
            memory["usage"] = f"{total_chars:,} chars ({len(memory_files)} files)"
        if not memory["usage"]:
            total_chars = sum(_safe_int(item.get("characters")) for item in memory_files)
            memory["usage"] = f"{total_chars:,} chars" if memory_files else "—"
        by_name = {item.get("name"): item for item in memory_files}
        for task_out in tasks_out:
            task_memory = _as_dict(task_out.get("memory"))
            if _safe_int(task_memory.get("entries")):
                continue
            item = by_name.get(f"TASK_{task_out.get('key')}.md")
            if item:
                task_memory["entries"] = _safe_int(item.get("entries"))
                task_memory["usage"] = f"{_safe_int(item.get('characters')):,} chars"
                task_out["memory"] = task_memory

        return {
            "ok": True,
            "schema_version": data.get("schema_version", SCHEMA_VERSION),
            "controller": run_agent.get("controller") or "pipeline",
            "run_agent": run_agent,
            "memory": memory,
            "tasks": tasks_out,
            "memory_files": memory_files,
            "lessons": lessons[:200],
            "strategy_stats": strategy_stats[:200],
            "skill_usage": list(db.get("skill_usage") or []),
            "db": {
                "path": str(db_path) if db_path else None,
                "available": bool(db.get("available")),
                "error": db.get("error") or None,
            },
            "generated_at": _utcnow_iso(),
            "_meta": {
                "state_path": str(self.state_path),
                "db_path": str(db_path) if db_path else None,
                "db_available": bool(db.get("available")),
            },
        }

    def agent_payload(self, task_key: str | None = None) -> dict | None:
        data = self.read()
        run = _as_dict(data.get("run"))
        run_agent = _as_dict(run.get("agent"))
        entries: list[dict] = []
        for raw_task in _as_list(data.get("tasks")):
            task = _as_dict(raw_task)
            key = str(task.get("key") or "")
            rounds: list[dict] = []
            for raw_round in _as_list(task.get("rounds")):
                round_item = _as_dict(raw_round)
                agent = _as_dict(round_item.get("agent"))
                usage = _as_dict(agent.get("usage"))
                trace: list[dict] = []
                for raw_step in _as_list(agent.get("trace")):
                    step = _as_dict(raw_step)
                    ok = step.get("ok")
                    trace.append(
                        {
                            "tool": _truncate(step.get("tool") or step.get("name") or "?", 120),
                            "ok": ok if isinstance(ok, bool) else None,
                            "detail": _truncate(step.get("detail") or step.get("message") or "", MAX_TRACE_DETAIL),
                            "at": _iso_or_empty(step.get("at") or step.get("ts")),
                        }
                    )
                latency = agent.get("latency")
                try:
                    latency = float(latency)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    latency = None
                rounds.append(
                    {
                        "index": _safe_int(round_item.get("index")),
                        "id": str(round_item.get("id") or ""),
                        "status": str(round_item.get("status") or ""),
                        "decision": round_item.get("decision"),
                        "agent": {
                            "ok": agent.get("ok"),
                            "fallback": bool(agent.get("fallback")),
                            "error": _truncate(agent.get("error") or "", 400),
                            "decision": agent.get("decision"),
                            "finished": bool(agent.get("finished")),
                            "usage": {
                                "input": _safe_int(usage.get("input")),
                                "output": _safe_int(usage.get("output")),
                                "reasoning": _safe_int(usage.get("reasoning")),
                            },
                            "latency": latency,
                            "tool_calls": _safe_int(agent.get("tool_calls")),
                            "trace": trace,
                            "proposal_sources": _normalise_proposal_sources(agent.get("proposal_sources")),
                            "env": _as_dict(agent.get("env")),
                        },
                    }
                )
            entries.append(
                {
                    "key": key,
                    "name": str(task.get("name") or task.get("name_en") or key),
                    "controller": task.get("controller") or run_agent.get("controller") or "pipeline",
                    "goal": _as_dict(task.get("goal")),
                    "agent": _as_dict(task.get("agent")),
                    "rounds": rounds,
                }
            )

        if task_key:
            wanted = task_key.strip().upper()
            found = next((entry for entry in entries if entry["key"].upper() == wanted), None)
            if found is None:
                return None
            return {
                "ok": True,
                "run_agent": run_agent,
                "task": found["key"],
                "task_agent": found["agent"],
                "goal": found["goal"],
                "controller": found["controller"],
                "rounds": found["rounds"],
                "generated_at": _utcnow_iso(),
            }
        return {"ok": True, "run_agent": run_agent, "tasks": entries, "generated_at": _utcnow_iso()}

    # -- records / rounds 搜索 ---------------------------------------------
    def _load_search_file(self, path: Path) -> tuple[list[dict], int, bool]:
        sig = self._stat_sig(path)
        with self._search_lock:
            cached = self._search_cache.get(path)
            if cached is not None and cached[0] == sig:
                return cached[1], cached[2], cached[3]
            items: list[dict] = []
            skipped = 0
            available = sig is not None and sig[1] <= MAX_RECORDS_BYTES
            if available:
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as fh:
                        lines = deque(fh, maxlen=MAX_SEARCH_LINES)
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            skipped += 1
                            continue
                        if isinstance(raw, dict):
                            items.append(raw)
                except OSError:
                    available = False
                    items = []
            self._search_cache[path] = (sig, items, skipped, available)
            return items, skipped, available

    @staticmethod
    def _search_item(raw: dict, file_name: str) -> dict:
        rtype = str(raw.get("type") or raw.get("kind") or ("round" if file_name == "rounds.jsonl" else "event"))
        ts = _iso_or_empty(raw.get("ts") or raw.get("created_at") or raw.get("timestamp"))
        task = str(raw.get("task") or raw.get("task_key") or "")
        if task == "-":
            task = ""
        round_no = raw.get("round_index", raw.get("index", raw.get("round")))
        try:
            round_no = int(round_no) if round_no is not None else None
        except (TypeError, ValueError):
            round_no = None
        message = raw.get("message") or raw.get("change_summary") or raw.get("hypothesis") or raw.get("detail") or raw.get("summary")
        if not isinstance(message, str) or not message.strip():
            if rtype in ("round", "decision"):
                bits = [f"第 {round_no} 轮" if round_no is not None else "轮次"]
                if raw.get("decision"):
                    bits.append(f"决策 {raw['decision']}")
                if raw.get("strategy"):
                    bits.append(f"策略 {raw['strategy']}")
                message = " · ".join(bits)
            elif rtype in ("llm_call", "llm"):
                message = f"LLM 调用 role={raw.get('role', '?')} ok={raw.get('ok', '?')}"
            elif rtype == "panel":
                message = f"panel 讨论 verdict={raw.get('verdict', '?')}"
            else:
                message = f"{rtype} 记录"
        try:
            haystack = json.dumps(raw, ensure_ascii=False, default=str).lower()
        except (TypeError, ValueError):
            haystack = str(raw).lower()
        return {
            "ts": ts,
            "type": rtype,
            "task": task,
            "message": _truncate(message.strip(), 600),
            "round": round_no,
            "source": file_name,
            "_haystack": haystack,
        }

    def search_records(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
        needle = (query or "").strip().lower()
        files = [self.records_path, self.state_path.parent / "rounds.jsonl"]
        items: list[dict] = []
        seen: set[tuple] = set()
        file_meta: list[dict] = []
        for path in files:
            raws, skipped, available = self._load_search_file(path)
            file_meta.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "available": available,
                    "records": len(raws),
                    "skipped_lines": skipped,
                }
            )
            for raw in raws:
                item = self._search_item(raw, path.name)
                if needle and needle not in item["_haystack"]:
                    continue
                fingerprint = (item["ts"], item["type"], item["task"], item["message"], item["round"])
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                items.append(item)
        items.sort(key=lambda item: _ts_sort_key(item.get("ts")), reverse=True)
        total = len(items)
        selected = items[: max(1, limit)]
        for item in selected:
            item.pop("_haystack", None)
        return {"items": selected, "total": total, "files": file_meta}

    # -- schema v3：软目标 / 知识面板 ----------------------------------------
    def goal_payload(self) -> dict:
        """汇总每任务软目标（目标/当前/gap/degraded/verdict）与批次时间线。"""
        data = self.read()
        run = _as_dict(data.get("run"))
        run_agent = _as_dict(run.get("agent"))
        tasks_out: list[dict] = []
        soft_count = 0
        degraded_count = 0
        for raw_task in _as_list(data.get("tasks")):
            task = _as_dict(raw_task)
            soft = _normalise_goal_soft(task)
            if soft["available"]:
                soft_count += 1
            if soft["degraded"]:
                degraded_count += 1
            tasks_out.append(
                {
                    "key": str(task.get("key") or ""),
                    "name": str(task.get("name") or task.get("name_en") or ""),
                    "primary_metric": task.get("primary_metric"),
                    "direction": task.get("direction") or soft["direction"],
                    "baseline": task.get("baseline"),
                    "best": task.get("best"),
                    "current": task.get("current"),
                    "status": task.get("status"),
                    "round_index": task.get("round_index"),
                    "rounds_total": task.get("rounds_total"),
                    "controller": task.get("controller") or run_agent.get("controller") or "pipeline",
                    "goal_soft": soft,
                    "drift": _normalise_drift(task),
                    "goal": _as_dict(task.get("goal")),
                }
            )

        batches = _normalise_batches(data)
        raw_batch_index = data.get("batch_index")
        batch_index = _safe_int(raw_batch_index) if raw_batch_index is not None else len(batches)

        return {
            "ok": True,
            "schema_version": data.get("schema_version", SCHEMA_VERSION),
            "soft_goal_mode": True,
            "note": SOFT_GOAL_NOTE,
            "batch_index": batch_index,
            "summary": {
                "tasks": len(tasks_out),
                "goal_soft": soft_count,
                "degraded": degraded_count,
                "batches": len(batches),
            },
            "tasks": tasks_out,
            "batches": batches,
            "generated_at": _utcnow_iso(),
            "_meta": {"state_path": str(self.state_path)},
        }

    def knowledge_payload(self) -> dict:
        """汇总知识命中 / 文档 / 技能使用 / lessons / 策略统计（复用 memory 读取）。"""
        data = self.read()
        memory = self.memory_payload()
        knowledge = _normalise_knowledge(
            data,
            list(memory.get("skill_usage") or []),
            list(memory.get("memory_files") or []),
        )
        return {
            "ok": True,
            "schema_version": memory.get("schema_version", SCHEMA_VERSION),
            "knowledge": knowledge,
            "skills_used": knowledge["skills_used"],
            "lessons": memory.get("lessons", []),
            "strategy_stats": memory.get("strategy_stats", []),
            "skill_usage": memory.get("skill_usage", []),
            "memory_files": memory.get("memory_files", []),
            "db": memory.get("db", {}),
            "generated_at": _utcnow_iso(),
            "_meta": memory.get("_meta", {}),
        }


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"ZhixunDashboard/{VERSION}"
    sys_version = ""

    # -- 输出工具 -----------------------------------------------------------
    @property
    def store(self) -> StateStore:
        return self.server.store  # type: ignore[attr-defined]

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str, **extra) -> None:
        payload = {"ok": False, "error": message, "status": int(status)}
        payload.update(extra)
        self._send_json(status, payload)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - 覆盖基类
        if getattr(self.server, "quiet", False):
            return
        sys.stderr.write(f"[dashboard] {self.address_string()} {fmt % args}\n")
        sys.stderr.flush()

    # -- 路由 ---------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802 - 标准库命名
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        split = urlsplit(self.path)
        path = split.path
        query = parse_qs(split.query, keep_blank_values=True)

        if path in ("/", "/index.html"):
            self._serve_static("index.html")
            return
        if path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            return
        if path in ("/app.js", "/style.css", "/favicon.svg"):
            self._serve_static(path.lstrip("/"))
            return
        if path == "/api/state":
            self._serve_state()
            return
        if path == "/api/events":
            self._serve_events(query)
            return
        if path == "/api/memory":
            self._serve_memory()
            return
        if path == "/api/agent":
            self._serve_agent(query)
            return
        if path == "/api/records/search":
            self._serve_records_search(query)
            return
        if path == "/api/goal":
            self._serve_goal()
            return
        if path == "/api/knowledge":
            self._serve_knowledge()
            return
        if path == "/api/health":
            self._serve_health()
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "路径不存在", path=path)

    def _serve_static(self, name: str) -> None:
        content_type = STATIC_FILES.get(name)
        if content_type is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "静态文件不存在", path=name)
            return
        target = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in target.parents:
            self._send_error_json(HTTPStatus.FORBIDDEN, "非法路径")
            return
        try:
            body = target.read_bytes()
        except OSError:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"静态文件缺失：{name}（请检查 rsi/dashboard/static/ 是否完整）",
            )
            return
        prefix = getattr(self.server, "public_prefix", "")
        if prefix and name in ("index.html", "app.js", "style.css"):
            text = body.decode("utf-8")
            for token in ("/favicon.svg", "/style.css", "/app.js", "/api/"):
                text = text.replace(f'"{token}', f'"{prefix}{token}')
            body = text.encode("utf-8")
        self._send(HTTPStatus.OK, body, content_type)

    def _serve_state(self) -> None:
        try:
            payload = self.store.read()
        except StateUnavailable as exc:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                exc.detail,
                kind=exc.kind,
                state_path=str(exc.path) if exc.path else None,
                hint="启动 harness 生成 state.json，或先用 scripts/mock_state.py 生成演示数据",
            )
            return
        self._send_json(HTTPStatus.OK, payload)

    def _serve_events(self, query: dict) -> None:
        raw_limit = (query.get("limit") or [""])[0]
        try:
            limit = int(raw_limit) if raw_limit != "" else DEFAULT_EVENTS_LIMIT
        except ValueError:
            limit = DEFAULT_EVENTS_LIMIT
        limit = max(1, min(limit, MAX_EVENTS_LIMIT))
        level = (query.get("level") or [""])[0].strip().lower()
        if level not in VALID_LEVELS:
            level = ""
        task = (query.get("task") or [""])[0].strip().upper()
        search = (query.get("q") or [""])[0].strip().lower()

        try:
            payload = self.store.read()
        except StateUnavailable as exc:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                exc.detail,
                kind=exc.kind,
                state_path=str(exc.path) if exc.path else None,
            )
            return

        events = [e for e in payload.get("events", []) if isinstance(e, dict)]
        if level:
            events = [e for e in events if e.get("level") == level]
        if task:
            events = [e for e in events if str(e.get("task") or "").upper() == task]
        if search:
            events = [e for e in events if search in str(e.get("message") or "").lower()]
        events.sort(key=lambda e: _ts_sort_key(e.get("ts")), reverse=True)
        total = len(events)
        selected = events[:limit]
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "events": selected,
                "total": total,
                "returned": len(selected),
                "limit": limit,
                "filters": {"level": level or None, "task": task or None, "q": search or None},
                "generated_at": _utcnow_iso(),
                "data_source": payload.get("_meta", {}).get("state_path"),
            },
        )

    def _serve_memory(self) -> None:
        try:
            payload = self.store.memory_payload()
        except StateUnavailable as exc:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                exc.detail,
                kind=exc.kind,
                state_path=str(exc.path) if exc.path else None,
                hint="记忆面板需要 state.json；数据库缺失时会自动降级为空列表",
            )
            return
        self._send_json(HTTPStatus.OK, payload)

    def _serve_agent(self, query: dict) -> None:
        task = (query.get("task") or [""])[0].strip()
        try:
            payload = self.store.agent_payload(task or None)
        except StateUnavailable as exc:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                exc.detail,
                kind=exc.kind,
                state_path=str(exc.path) if exc.path else None,
            )
            return
        if payload is None:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                f"未知任务：{task}（可选 A / B / C）",
                task=task,
                valid_tasks=["A", "B", "C"],
            )
            return
        self._send_json(HTTPStatus.OK, payload)

    def _serve_records_search(self, query: dict) -> None:
        needle = (query.get("q") or [""])[0].strip()
        raw_limit = (query.get("limit") or [""])[0]
        try:
            limit = int(raw_limit) if raw_limit != "" else DEFAULT_SEARCH_LIMIT
        except ValueError:
            limit = DEFAULT_SEARCH_LIMIT
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))
        try:
            result = self.store.search_records(needle, limit)
        except OSError as exc:  # 文件被占用等极端情况
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"搜索失败：{exc}")
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "query": needle,
                "limit": limit,
                "items": result["items"],
                "total": result["total"],
                "returned": len(result["items"]),
                "files": result["files"],
                "generated_at": _utcnow_iso(),
            },
        )

    def _serve_goal(self) -> None:
        try:
            payload = self.store.goal_payload()
        except StateUnavailable as exc:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                exc.detail,
                kind=exc.kind,
                state_path=str(exc.path) if exc.path else None,
                hint="软目标面板需要 state.json；旧 schema 会降级为空结构",
            )
            return
        self._send_json(HTTPStatus.OK, payload)

    def _serve_knowledge(self) -> None:
        try:
            payload = self.store.knowledge_payload()
        except StateUnavailable as exc:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                exc.detail,
                kind=exc.kind,
                state_path=str(exc.path) if exc.path else None,
                hint="知识面板需要 state.json；缺少 knowledge 字段时会降级为空列表",
            )
            return
        self._send_json(HTTPStatus.OK, payload)

    def _serve_health(self) -> None:
        base = {
            "ok": True,
            "version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "uptime_seconds": round(time.monotonic() - self.server.started_monotonic, 1),  # type: ignore[attr-defined]
            "state_path": str(self.store.state_path),
        }
        try:
            payload = self.store.read()
        except StateUnavailable as exc:
            base.update({"ok": False, "state": exc.kind, "detail": exc.detail})
            self._send_json(HTTPStatus.OK, base)
            return
        run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
        base.update(
            {
                "state": "ready",
                "run_status": run.get("status", "unknown"),
                "event_count": payload.get("_meta", {}).get("event_count", 0),
            }
        )
        self._send_json(HTTPStatus.OK, base)


# --------------------------------------------------------------------------- #
# Server 组装
# --------------------------------------------------------------------------- #
class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, store: StateStore, quiet: bool = False, public_prefix: str = ""):
        super().__init__(server_address, DashboardHandler)
        self.store = store
        self.quiet = quiet
        prefix = (public_prefix or "").strip()
        if prefix in ("", "/"):
            prefix = ""
        else:
            if not prefix.startswith("/"):
                prefix = "/" + prefix
            prefix = prefix.rstrip("/")
        self.public_prefix = prefix
        self.started_monotonic = time.monotonic()

    @property
    def bound_port(self) -> int:
        return int(self.server_address[1])

    def listening_url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        return f"http://{display_host}:{port}"


def create_server(
    state_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7861,
    records_path: str | Path | None = None,
    quiet: bool = False,
    follow_dir: str | Path | None = None,
    public_prefix: str = "",
) -> DashboardServer:
    store = StateStore(state_path, records_path=records_path, follow_dir=follow_dir)
    return DashboardServer((host, int(port)), store, quiet=quiet, public_prefix=public_prefix)


def serve(server: DashboardServer) -> None:
    """阻塞运行，Ctrl-C / KeyboardInterrupt 时干净退出。"""
    print(f"[dashboard] 数据源: {server.store.state_path}")
    if server.store.records_path.exists():
        print(f"[dashboard] 事件源: {server.store.state_path} + {server.store.records_path}")
    print(f"[dashboard] listening on {server.listening_url()}")
    sys.stdout.flush()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n[dashboard] 收到 Ctrl-C，正在关闭…")
    finally:
        server.shutdown()
        server.server_close()
        print("[dashboard] 已停止")
        sys.stdout.flush()
