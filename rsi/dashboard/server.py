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
GET /api/state        -> 完整 state（schema v1 + 附加 ``_meta``）
GET /api/events       -> 事件流，支持 limit / level / task / q
GET /api/health       -> 轻量健康检查
其他                  -> 404（JSON）
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

VERSION = "1.0.0"
SCHEMA_VERSION = 1

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
# state / records 读取（带 mtime 缓存）
# --------------------------------------------------------------------------- #
class StateStore:
    """线程安全地读取 state.json，并按 mtime 缓存解析结果。"""

    def __init__(self, state_path: str | Path, records_path: str | Path | None = None):
        self.state_path = Path(state_path).expanduser().resolve()
        if records_path:
            self.records_path = Path(records_path).expanduser().resolve()
        else:
            self.records_path = self.state_path.parent / "records.jsonl"
        self._lock = threading.Lock()
        self._state_sig: tuple | None = None
        self._state_cache: dict | None = None
        self._last_good: dict | None = None
        self._records_sig: tuple | None = None
        self._records_cache: list[dict] = []
        self._records_skipped: int = 0

    # -- 内部 ---------------------------------------------------------------
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

    def __init__(self, server_address, store: StateStore, quiet: bool = False):
        super().__init__(server_address, DashboardHandler)
        self.store = store
        self.quiet = quiet
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
) -> DashboardServer:
    store = StateStore(state_path, records_path=records_path)
    return DashboardServer((host, int(port)), store, quiet=quiet)


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
