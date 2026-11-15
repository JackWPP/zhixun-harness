"""端到端测试：真实启动看板服务，验证页面、API、静态资源与清理。

运行：``./.venv/bin/python -m pytest tests/test_dashboard.py -q``
"""
from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_SCRIPT = REPO_ROOT / "scripts" / "mock_state.py"
EXTERNAL_SCHEMES = ("http://", "https://")


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_get(port: int, path: str, timeout: float = 5.0) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def _wait_until_listening(proc: subprocess.Popen, port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"看板服务提前退出（exit={proc.returncode}）：\n{output}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.05)
    pytest.fail(f"看板服务在 {timeout:.0f}s 内未监听 127.0.0.1:{port}")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def demo_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """用 mock_state.py 命令行生成一份演示 state.json 与 records.jsonl。"""
    tmp_dir = tmp_path_factory.mktemp("dashboard-demo")
    state_path = tmp_dir / "state.json"
    records_path = tmp_dir / "records.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(MOCK_SCRIPT),
            "--out",
            str(state_path),
            "--records-out",
            str(records_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"mock_state.py 生成失败：{result.stderr}\n{result.stdout}"
    assert state_path.is_file(), "mock_state.py 未生成 state.json"
    assert records_path.is_file(), "mock_state.py 未生成 records.jsonl"
    return state_path, records_path


@pytest.fixture(scope="module")
def running_server(demo_files: tuple[Path, Path]) -> int:
    """临时端口启动 ``python -m rsi.dashboard``，yield 端口，退出时清理。"""
    state_path, records_path = demo_files
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rsi.dashboard",
            "--state",
            str(state_path),
            "--records",
            str(records_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        _wait_until_listening(proc, port)
        yield port
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=8)
        if proc.stdout:
            proc.stdout.close()


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
def test_index_page_renders_chinese_shell(running_server: int) -> None:
    status, headers, raw = http_get(running_server, "/")
    assert status == 200
    content_type = headers.get("Content-Type", "")
    assert "text/html" in content_type
    assert "charset=utf-8" in content_type.lower()

    html = raw.decode("utf-8")
    for text in ("智训引擎", "RSI 控制台", "任务矩阵", "全局事件流", "决策与运行摘要", "记忆与自主内核", "MEMORY"):
        assert text in html, f"页面缺少关键中文：{text}"
    assert 'href="/style.css"' in html
    assert 'src="/app.js"' in html
    assert "viewport" in html

    lowered = html.lower()
    for scheme in EXTERNAL_SCHEMES:
        assert scheme not in lowered, f"HTML 不应引用外部资源：{scheme}"


def test_static_assets_are_offline(running_server: int) -> None:
    cases = {
        "/app.js": b"/api/state",
        "/style.css": b":root",
        "/favicon.svg": b"<svg",
    }
    for path, needle in cases.items():
        status, headers, raw = http_get(running_server, path)
        assert status == 200, f"{path} 返回 {status}"
        assert needle in raw, f"{path} 内容异常"
        assert "charset=utf-8" in headers.get("Content-Type", "").lower() or path.endswith(".svg")
        text = raw.decode("utf-8", "replace").lower()
        # SVG 的 xmlns 是命名空间标识而非网络请求；其余静态资源不得出现外部 URL。
        if not path.endswith(".svg"):
            for scheme in EXTERNAL_SCHEMES:
                assert scheme not in text, f"{path} 引用了外部资源：{scheme}"

    status, _headers, _raw = http_get(running_server, "/favicon.ico")
    assert status == 204


# --------------------------------------------------------------------------- #
# /api/state
# --------------------------------------------------------------------------- #
def test_state_api_returns_canonical_schema(running_server: int) -> None:
    status, headers, raw = http_get(running_server, "/api/state")
    assert status == 200
    assert "application/json" in headers.get("Content-Type", "")

    payload = json.loads(raw.decode("utf-8"))
    assert payload["ok"] is True
    assert payload["schema_version"] in {1, 2}
    assert payload["run"]["run_id"]
    assert payload["run"]["status"] in {"running", "finished", "failed", "stopped"}
    assert payload["run"]["budget"]["tokens_cap"] > 0

    tasks = {task["key"]: task for task in payload["tasks"]}
    assert set(tasks) == {"A", "B", "C"}
    assert tasks["A"]["primary_metric"] == "macro_f1"
    assert tasks["A"]["direction"] == "higher"
    assert tasks["B"]["primary_metric"] == "rmse"
    assert tasks["B"]["direction"] == "lower"

    task_a = tasks["A"]
    assert len(task_a["rounds"]) >= 6
    assert task_a["panel"], "任务 A 缺少 panel 讨论数据"
    first_panel = task_a["panel"][0]
    assert first_panel["messages"]
    assert {"role", "standpoint", "evidence", "confidence"} <= set(first_panel["messages"][0])

    round_fields = {
        "index",
        "status",
        "decision",
        "hypothesis",
        "change_summary",
        "strategy",
        "postprocess",
        "metrics",
        "candidate_checkpoint_sha256",
    }
    for one_round in task_a["rounds"]:
        assert round_fields <= set(one_round), f"轮次缺少字段：{round_fields - set(one_round)}"
        assert one_round["decision"] in {"promote", "retain", "rollback", None}
        assert len(one_round["candidate_checkpoint_sha256"]) == 64

    meta = payload["_meta"]
    assert meta["state_path"].endswith("state.json")
    assert meta["schema_version"] in {1, 2}
    assert meta["records_available"] is True
    assert meta["records_events"] > 0
    assert meta["event_count"] == len(payload["events"])
    assert payload["events"], "state 事件流为空"


def test_state_api_reflects_file_update(running_server: int, demo_files: tuple[Path, Path]) -> None:
    """原子替换 state.json 后，下一次请求应读到新内容（模拟 harness 实时写入）。"""
    state_path, _records_path = demo_files
    before = json.loads(http_get(running_server, "/api/state")[2].decode("utf-8"))
    original = state_path.read_bytes()
    try:
        updated = json.loads(original.decode("utf-8"))
        updated["run"]["run_id"] = "rsi-test-live-update"
        updated["run"]["elapsed_seconds"] = before["run"]["elapsed_seconds"] + 7.5
        state_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

        status, _headers, raw = http_get(running_server, "/api/state")
        assert status == 200
        after = json.loads(raw.decode("utf-8"))
        assert after["run"]["run_id"] == "rsi-test-live-update"
        assert after["run"]["elapsed_seconds"] > before["run"]["elapsed_seconds"]
        assert after["_meta"]["cache"] in {"miss", "hit"}
    finally:
        state_path.write_bytes(original)
        # 恢复后再次请求，确保服务重新缓存原内容
        restored = json.loads(http_get(running_server, "/api/state")[2].decode("utf-8"))
        assert restored["run"]["run_id"] == before["run"]["run_id"]


# --------------------------------------------------------------------------- #
# /api/events
# --------------------------------------------------------------------------- #
def test_events_api_limit_and_order(running_server: int) -> None:
    status, headers, raw = http_get(running_server, "/api/events?limit=5")
    assert status == 200
    assert "application/json" in headers.get("Content-Type", "")

    payload = json.loads(raw.decode("utf-8"))
    assert payload["ok"] is True
    events = payload["events"]
    assert 1 <= len(events) <= 5
    assert payload["total"] >= 40, "mock 应产生 40+ 条事件"
    assert payload["returned"] == len(events)
    assert payload["limit"] == 5

    for event in events:
        assert {"ts", "level", "task", "message"} <= set(event)
        assert event["level"] in {"info", "warn", "error"}
        assert event["message"]

    timestamps = [event["ts"] for event in events]
    assert timestamps == sorted(timestamps, reverse=True), "事件应按时间倒序返回"


def test_events_api_filters(running_server: int) -> None:
    status, _headers, raw = http_get(running_server, "/api/events?limit=100&level=error")
    assert status == 200
    errors = json.loads(raw.decode("utf-8"))["events"]
    assert errors, "mock 中应包含 error 级事件"
    assert all(event["level"] == "error" for event in errors)

    status, _headers, raw = http_get(running_server, "/api/events?limit=100&task=B")
    assert status == 200
    task_b = json.loads(raw.decode("utf-8"))["events"]
    assert task_b
    assert all(str(event["task"]).upper() == "B" for event in task_b)


# --------------------------------------------------------------------------- #
# 错误路径与健康检查
# --------------------------------------------------------------------------- #
def test_unknown_path_returns_json_404(running_server: int) -> None:
    status, headers, raw = http_get(running_server, "/no-such-page")
    assert status == 404
    assert "application/json" in headers.get("Content-Type", "")
    payload = json.loads(raw.decode("utf-8"))
    assert payload["ok"] is False
    assert payload["error"]
    assert payload["path"] == "/no-such-page"


def test_health_endpoint(running_server: int) -> None:
    status, _headers, raw = http_get(running_server, "/api/health")
    assert status == 200
    payload = json.loads(raw.decode("utf-8"))
    assert payload["ok"] is True
    assert payload["state"] == "ready"
    assert payload["schema_version"] in {1, 2}
    assert payload["run_status"] in {"running", "finished", "failed", "stopped"}
    assert payload["event_count"] >= 40


# --------------------------------------------------------------------------- #
# schema v2：记忆 / 自主内核 / 记录检索
# --------------------------------------------------------------------------- #
def _build_demo_bundle(state_path: Path, include_db: bool = True) -> None:
    """调用 rsi.dashboard.demo.write_demo_bundle 生成一份 v2 演示数据包。"""
    code = (
        "import sys\n"
        "from rsi.dashboard.demo import write_demo_bundle\n"
        "write_demo_bundle(sys.argv[1], include_db=sys.argv[2] == '1')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(state_path), "1" if include_db else "0"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"demo 数据包生成失败：{result.stderr}\n{result.stdout}"
    assert state_path.is_file(), "demo 数据包缺少 state.json"
    assert (state_path.parent / "records.jsonl").is_file(), "demo 数据包缺少 records.jsonl"
    assert (state_path.parent / "rounds.jsonl").is_file(), "demo 数据包缺少 rounds.jsonl"
    assert (state_path.parent / "memory").is_dir(), "demo 数据包缺少 memory/"


def _start_dashboard(state_path: Path, records_path: Path | None = None) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "rsi.dashboard",
        "--state",
        str(state_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--quiet",
    ]
    if records_path is not None:
        cmd.extend(["--records", str(records_path)])
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        _wait_until_listening(proc, port)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        raise
    return proc, port


def _stop_dashboard(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=8)
    if proc.stdout:
        proc.stdout.close()


@pytest.fixture(scope="module")
def v2_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp_dir = tmp_path_factory.mktemp("dashboard-v2")
    state_path = tmp_dir / "state.json"
    _build_demo_bundle(state_path, include_db=True)
    return tmp_dir, state_path


@pytest.fixture(scope="module")
def v2_server(v2_bundle: tuple[Path, Path]) -> int:
    _tmp_dir, state_path = v2_bundle
    proc, port = _start_dashboard(state_path)
    try:
        yield port
    finally:
        _stop_dashboard(proc)


@pytest.fixture(scope="module")
def v2_nodb_server(tmp_path_factory: pytest.TempPathFactory) -> int:
    tmp_dir = tmp_path_factory.mktemp("dashboard-v2-nodb")
    state_path = tmp_dir / "state.json"
    _build_demo_bundle(state_path, include_db=False)
    proc, port = _start_dashboard(state_path)
    try:
        yield port
    finally:
        _stop_dashboard(proc)


def test_memory_api_reads_v2_state_and_sqlite(v2_server: int) -> None:
    status, headers, raw = http_get(v2_server, "/api/memory")
    assert status == 200
    assert "application/json" in headers.get("Content-Type", "")
    payload = json.loads(raw.decode("utf-8"))

    assert payload["ok"] is True
    assert payload["schema_version"] == 2
    assert payload["controller"] in {"kernel", "pipeline"}

    run_agent = payload["run_agent"]
    assert run_agent["controller"] == payload["controller"]
    assert run_agent["kernel_rounds"] >= 1
    assert run_agent["kernel_tokens"] >= 0

    memory = payload["memory"]
    assert memory["enabled"] is True
    assert "%" in str(memory["usage"]), "MEMORY 用量应包含百分比"
    assert memory["dir"]

    assert {task["key"] for task in payload["tasks"]} == {"A", "B", "C"}
    task_a = next(task for task in payload["tasks"] if task["key"] == "A")
    assert task_a["goal"]["rounds_done"] >= 6
    assert task_a["goal"]["subgoals"]

    assert payload["db"]["available"] is True, "演示包应包含可读的 rsi_state.db"
    assert payload["_meta"]["db_available"] is True

    lessons = payload["lessons"]
    assert lessons, "应从 sqlite 读取 lessons"
    assert any(lesson.get("source") == "db" for lesson in lessons)
    for lesson in lessons[:5]:
        assert {"task", "text", "wins"} <= set(lesson)
        assert lesson["text"]

    strategies = payload["strategy_stats"]
    assert strategies, "策略统计为空"
    for row in strategies[:5]:
        assert {"strategy", "trials", "promotions", "win_rate"} <= set(row)
        assert 0.0 <= float(row["win_rate"]) <= 1.0

    assert payload["skill_usage"], "技能使用统计为空"
    assert payload["memory_files"], "memory/ 目录快照为空"
    assert any(item["name"] == "MEMORY.md" for item in payload["memory_files"])


def test_memory_api_degrades_without_sqlite(v2_nodb_server: int) -> None:
    status, _headers, raw = http_get(v2_nodb_server, "/api/memory")
    assert status == 200
    payload = json.loads(raw.decode("utf-8"))
    assert payload["ok"] is True
    assert payload["db"]["available"] is False
    # 数据库缺失时，lessons 应回退到 state.json 中的经验快照而不是直接为空。
    assert payload["lessons"], "无 sqlite 时应从 state.json 回退 lessons"
    assert all(lesson.get("source") == "state" for lesson in payload["lessons"])
    assert payload["tasks"], "降级时任务记忆仍应可用"


def test_agent_api_exposes_kernel_trace(v2_server: int) -> None:
    status, headers, raw = http_get(v2_server, "/api/agent?task=A")
    assert status == 200
    assert "application/json" in headers.get("Content-Type", "")
    payload = json.loads(raw.decode("utf-8"))

    assert payload["ok"] is True
    assert payload["task"] == "A"
    assert payload["controller"] in {"kernel", "pipeline"}
    goal = payload["goal"]
    assert goal["goal"]
    assert {"rounds_done", "rounds_target", "subgoals"} <= set(goal)

    rounds = payload["rounds"]
    assert len(rounds) >= 6, "任务 A 应有 6 个已完成轮次"
    traced = [item for item in rounds if item["agent"]["trace"]]
    assert traced, "至少一轮应包含内核工具轨迹"
    for item in rounds:
        assert {"index", "status", "decision", "agent"} <= set(item)
        agent = item["agent"]
        assert {"ok", "usage", "latency", "tool_calls", "trace"} <= set(agent)
        assert {"input", "output", "reasoning"} <= set(agent["usage"])
    step = traced[0]["agent"]["trace"][0]
    assert {"tool", "ok", "detail", "at"} <= set(step)

    status, _headers, raw = http_get(v2_server, "/api/agent")
    assert status == 200
    everything = json.loads(raw.decode("utf-8"))
    assert {task["key"] for task in everything["tasks"]} == {"A", "B", "C"}

    status, _headers, raw = http_get(v2_server, "/api/agent?task=Z")
    assert status == 400
    error = json.loads(raw.decode("utf-8"))
    assert error["ok"] is False
    assert "未知任务" in error["error"]


def test_records_search_api(v2_server: int) -> None:
    status, headers, raw = http_get(v2_server, "/api/records/search?q=promote&limit=5")
    assert status == 200
    assert "application/json" in headers.get("Content-Type", "")
    payload = json.loads(raw.decode("utf-8"))

    assert payload["ok"] is True
    items = payload["items"]
    assert 1 <= len(items) <= 5
    assert payload["total"] >= len(items)
    timestamps = [item["ts"] for item in items]
    assert timestamps == sorted(timestamps, reverse=True), "检索结果应按时间倒序"
    for item in items:
        assert {"ts", "type", "task", "message"} <= set(item)
        assert "_haystack" not in item, "内部检索字段不应泄漏"
    file_names = {item.get("name") for item in payload["files"]}
    assert "records.jsonl" in file_names
    assert "rounds.jsonl" in file_names

    # 空 q：返回最近记录（最多 20 条）。
    status, _headers, raw = http_get(v2_server, "/api/records/search")
    assert status == 200
    recent = json.loads(raw.decode("utf-8"))
    assert 1 <= len(recent["items"]) <= 20

    # 大小写不敏感。
    _status, _headers, raw = http_get(v2_server, "/api/records/search?q=PROMOTE&limit=5")
    assert json.loads(raw.decode("utf-8"))["total"] == payload["total"]

    # 无匹配。
    status, _headers, raw = http_get(v2_server, "/api/records/search?q=no-such-keyword-xyz")
    assert status == 200
    empty = json.loads(raw.decode("utf-8"))
    assert empty["items"] == []
    assert empty["total"] == 0
