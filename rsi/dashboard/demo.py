"""schema v2 演示数据包：在 ``scripts/mock_state.py`` 的 v1 场景上叠加"记忆/自主内核"字段。

生成内容（全部落盘到 state.json 同目录，零第三方依赖）::

    <dir>/state.json          # schema_version = 2
    <dir>/records.jsonl       # event / panel / llm_call 行
    <dir>/rounds.jsonl        # round 行（供 /api/records/search 检索）
    <dir>/memory/MEMORY.md    # 全局 curated 记忆
    <dir>/memory/TASK_A.md    # 任务级快照
    <dir>/rsi_state.db        # sqlite：lessons / strategy_stats / skill_usage

用法::

    ./.venv/bin/python -m rsi.dashboard                    # 自动生成 .scratch/ 演示包
    ./.venv/bin/python -m rsi.dashboard --refresh-demo     # 强制重建
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 3
DEFAULT_DEMO_STATE = ".scratch/demo_state.json"
MEMORY_CAP_CHARS = 4000
REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_SCRIPT = REPO_ROOT / "scripts" / "mock_state.py"

CONTROLLER_BY_TASK = {"A": "kernel", "B": "kernel", "C": "pipeline"}

# schema v3：软目标（仅用于展示，不阻塞完成）与批次 / 知识演示数据。
SOFT_GOAL_TARGETS = {"A": 0.76, "B": 600.0, "C": 50.0}
SOFT_GOAL_DEGRADED = {"A": True, "B": False, "C": False}
SOFT_GOAL_EVIDENCE = {
    "A": "Acetaldehyde 跨批次 F1=0，Toluene 验证样本仅 5 个；severe 漂移下软目标缺口主要来自这两类，未阻塞完成。",
    "B": "验证集 90.2% 来自 Bearing3_4 单轴承，方差偏大；RMSE 仍高于软目标，按计划保留 rollback。",
    "C": "同分布随机切分下已接近基线天花板，软目标仅作方向参考，shrink 后处理可能过度保守。",
}
DRIFT_TYPE = {"A": "covariate_shift", "B": "load_regime_shift", "C": "none"}
DRIFT_ATTRIBUTION = {
    "A": ["批次温湿度偏移", "传感器老化", "Acetaldehyde 跨批次不可分"],
    "B": ["Bearing3_4 工况变化", "单轴承验证方差"],
    "C": ["无显著漂移"],
}
KNOWLEDGE_DOCS = [
    {"title": "漂移诊断手册 / batch covariate shift", "path": "docs/03-drift.md", "hits": 14},
    {"title": "后处理策略对照（recenter / shrink）", "path": "docs/05-postprocess.md", "hits": 9},
    {"title": "小验证集方差与分组重估", "path": "docs/04-validation.md", "hits": 7},
    {"title": "SQLite lessons / strategy_stats schema", "path": "docs/ui_design.md#memory", "hits": 5},
    {"title": "checkpoint 回滚与门控判据", "path": "docs/06-gating.md", "hits": 2},
]
PROPOSAL_SOURCES_SEQ = [
    {"seed": 1, "agent": 0, "knowledge": 0, "panel": 0},
    {"seed": 0, "agent": 2, "knowledge": 1, "panel": 0},
    {"seed": 1, "agent": 1, "knowledge": 2, "panel": 1},
    {"seed": 0, "agent": 1, "knowledge": 2, "panel": 2},
    {"seed": 1, "agent": 2, "knowledge": 1, "panel": 1},
]

GOAL_TEXT = {
    "A": "把 A 气体传感器六类识别 macro-F1 从基线 0.7140 提升到 ≥0.740，并证明 batch 漂移处理可复现。",
    "B": "把 B 滚动轴承 RUL 的 RMSE 从基线 681.16 分钟降到 ≤620，同时控制单轴承验证方差。",
    "C": "在 C 涡扇发动机 RUL 上保持 RMSE ≤50 循环，验证低风险后处理不会伤害同分布表现。",
}
SUBGOALS = {
    "A": ["量化解漂移：对 batch2 做 KS / 特征位移诊断", "冻结归一化后短程续训，确认漂移可回收", "处理 Acetaldehyde 跨批次失效（换损失/加特征）", "固化 promote 判据与回滚阈值"],
    "B": ["确认 RMSE 方差来源：按轴承分组重估", "用 SWA / ridge 降低小验证集方差", "避免对单轴承过拟合，保留 rollback 能力"],
    "C": ["建立同分布基线复现（window_sizes=[1]）", "验证 shrink 后处理收益，避免过度保守"],
}
LESSONS = [
    {
        "task": "A",
        "scope": "classification/drift",
        "text": "batch 2 上冻结 scaler 的 continue_train 可稳定回收 ~+0.008 macro-F1，成本低于重训。",
        "wins": 3,
        "uses": 4,
        "evidence": {"round": 5, "decision": "promote", "strategy": "continue_train"},
    },
    {
        "task": "A",
        "scope": "postprocess/recenter",
        "text": "severe 漂移下 recenter(batch_mean) 比重拟合归一化更稳，小批统计噪声是主要风险。",
        "wins": 2,
        "uses": 3,
        "evidence": {"round": 5, "decision": "promote", "strategy": "continue_train"},
    },
    {
        "task": "B",
        "scope": "validation-variance",
        "text": "验证集 90% 来自单轴承，SWA 的方差缩减收益有限，需按轴承分组重估。",
        "wins": 2,
        "uses": 2,
        "evidence": {"round": 3, "decision": "promote", "strategy": "swa"},
    },
    {
        "task": "A",
        "scope": "class-imbalance",
        "text": "Acetaldehyde 跨批次 F1 恒为 0，继续调学习率无意义，应改损失/换特征。",
        "wins": 1,
        "uses": 2,
        "evidence": {"round": 3, "decision": "promote", "strategy": "swa"},
    },
    {
        "task": "C",
        "scope": "regression/plateau",
        "text": "同分布随机切分下 ridge_last_layer 已接近上限，shrink 后处理容易过度保守。",
        "wins": 1,
        "uses": 1,
        "evidence": {"round": 1, "decision": "promote", "strategy": "continue_train"},
    },
    {
        "task": "B",
        "scope": "reporting",
        "text": "RMSE 被少数长寿命轴承放大，报告应同时给出 MAE 与分组误差。",
        "wins": 0,
        "uses": 2,
        "evidence": {"round": 2, "decision": "retain", "strategy": "ridge_last_layer"},
    },
]
SKILL_USAGE = [
    {"skill": "drift_diagnose", "uses": 6},
    {"skill": "memory_query", "uses": 5},
    {"skill": "experiment_plan", "uses": 4},
    {"skill": "panel_review", "uses": 4},
    {"skill": "checkpoint_guard", "uses": 3},
    {"skill": "metric_compare", "uses": 2},
]

TRACE_TOOLS = {
    "A": ["memory.query", "task.inspect", "drift.diagnose", "metrics.compare", "plan.propose", "experiment.run", "critic.review", "decision.commit"],
    "B": ["memory.query", "task.inspect", "metrics.compare", "plan.propose", "experiment.run", "decision.commit"],
    "C": ["metrics.compare", "plan.propose", "experiment.run", "decision.commit"],
}


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _load_mock_module():
    """按文件路径加载 scripts/mock_state.py（scripts/ 不是包）。"""
    if not MOCK_SCRIPT.is_file():
        raise FileNotFoundError(f"找不到 mock 数据生成器：{MOCK_SCRIPT}")
    spec = importlib.util.spec_from_file_location("zhixun_mock_state_demo", MOCK_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 mock 数据生成器：{MOCK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _iso_between(start: str, end: str, ratio: float) -> str:
    """在线性时间内插值一个 ISO 时间（演示用，容忍解析失败）。"""
    try:
        a = datetime.fromisoformat(start)
        b = datetime.fromisoformat(end)
        return (a + (b - a) * ratio).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return start


def _last_ts(tasks: list[dict], fallback: str) -> str:
    latest = fallback
    for task in tasks:
        for rnd in task.get("rounds", []):
            ts = rnd.get("created_at")
            if isinstance(ts, str) and ts > latest:
                latest = ts
    return latest


def _metric_improvement(task: dict, value: float | None, baseline: float) -> float | None:
    if value is None or not isinstance(value, (int, float)):
        return None
    return round(float(value) - float(baseline), 4) if task.get("direction") == "higher" else round(float(baseline) - float(value), 4)


def _postprocess_type(spec: str | None) -> str:
    if not spec:
        return "none"
    return str(spec).split(":", 1)[0]


# --------------------------------------------------------------------------- #
# 各任务衍生字段
# --------------------------------------------------------------------------- #
def _task_rounds_agent(task: dict, controller: str, started_at: str, last_ts: str) -> list[dict]:
    key = str(task.get("key") or "")
    rounds = [r for r in task.get("rounds", []) if isinstance(r, dict)]
    total = max(1, len(rounds))
    tools = TRACE_TOOLS.get(key, TRACE_TOOLS["C"])
    for offset, rnd in enumerate(rounds):
        index = int(rnd.get("index") or offset + 1)
        created = str(rnd.get("created_at") or started_at)
        base_ratio = (offset + 1) / (total + 1)
        trace: list[dict] = []
        if controller == "kernel":
            for step_no, tool in enumerate(tools):
                ok = not (key == "A" and index == 4 and tool == "experiment.run")
                trace.append(
                    {
                        "tool": tool,
                        "ok": ok,
                        "detail": {
                            "memory.query": f"召回 {2 + index % 3} 条任务记忆（strategy={rnd.get('strategy')}）",
                            "task.inspect": f"{key} 任务证据打包：{task.get('primary_metric')} 基线 {task.get('baseline')}",
                            "drift.diagnose": f"severity={task.get('drift', {}).get('severity')} score={task.get('drift', {}).get('score')}，触发回滚保护",
                            "metrics.compare": f"候选 {rnd.get('metrics')} vs 已接受检查点",
                            "plan.propose": f"提出策略 {rnd.get('strategy')} + {rnd.get('postprocess')}",
                            "experiment.run": "执行实验并写回 metrics / checkpoint",
                            "critic.review": "校验无数据泄漏、预算与回滚条件",
                            "decision.commit": f"写入决策 {rnd.get('decision')}",
                        }.get(tool, tool),
                        "at": _iso_between(started_at, last_ts, min(0.99, base_ratio + step_no * 0.01)),
                    }
                )
        usage = {
            "input": 3800 + index * 260 + offset * 40,
            "output": 720 + index * 55,
            "reasoning": 310 + index * 38,
        }
        rnd["agent"] = {
            "ok": bool(rnd.get("status") == "done" or rnd.get("status") == "running"),
            "fallback": bool(key == "A" and index == 4),
            "error": "实验指标未超过已接受检查点，已自动回滚" if key == "A" and index == 4 else "",
            "decision": rnd.get("decision"),
            "finished": rnd.get("status") == "done",
            "usage": usage,
            "latency": round(18.4 + index * 3.1 + (2.6 if controller == "kernel" else 0.8), 1),
            "tool_calls": len(trace),
            "trace": trace,
            "proposal_sources": dict(PROPOSAL_SOURCES_SEQ[(index - 1) % len(PROPOSAL_SOURCES_SEQ)]),
            "env": {"controller": controller, "device": "cpu", "seed": 42 + index, "budget_left_tokens": max(0, 4000000 - 120000 * index)},
        }
    return rounds


def _task_goal(task: dict, run_id: str, started_at: str) -> dict:
    key = str(task.get("key") or "")
    rounds = [r for r in task.get("rounds", []) if isinstance(r, dict)]
    done = [r for r in rounds if r.get("status") == "done"]
    last = done[-1] if done else {}
    finished = task.get("status") == "done"
    return {
        "goal": GOAL_TEXT.get(key, ""),
        "task_key": key,
        "run_id": run_id,
        "goal_id": f"{key}-{run_id}",
        "status": "done" if finished else "running",
        "turns_used": len(done),
        "max_turns": int(task.get("rounds_total") or 10),
        "rounds_done": len(done),
        "rounds_target": int(task.get("rounds_total") or 10),
        "consecutive_parse_failures": 0,
        "subgoals": SUBGOALS.get(key, []),
        "last_verdict": last.get("decision") or "pending",
        "last_reason": "全部轮次完成，已固化最优检查点" if finished else "仍在预算内推进，下一轮优先处理漂移/方差",
        "created_at": started_at,
        "updated_at": str(last.get("created_at") or started_at),
        "judge": {
            "done": bool(finished),
            "reason": "达到轮次目标并保留证据" if finished else "尚未达到目标，继续执行",
        },
    }


def _task_goal_soft(task: dict) -> dict:
    """v3 软目标：目标 / 当前 / 缺口 / 降级标记 / verdict；仅展示，不阻塞完成。"""
    key = str(task.get("key") or "")
    metric = str(task.get("primary_metric") or "")
    direction = str(task.get("direction") or "lower")
    target = SOFT_GOAL_TARGETS.get(key)
    current_value = task.get("current")
    if current_value is None:
        for rnd in reversed([r for r in task.get("rounds", []) if isinstance(r, dict)]):
            metrics = rnd.get("metrics")
            if isinstance(metrics, dict) and metrics.get(metric) is not None:
                current_value = metrics.get(metric)
                break
    targets: dict[str, float] = {metric: target} if target is not None and metric else {}
    current: dict[str, float] = (
        {metric: round(float(current_value), 4)} if current_value is not None and metric else {}
    )
    gap = None
    if target is not None and current_value is not None:
        raw_gap = (target - float(current_value)) if direction == "higher" else (float(current_value) - target)
        gap = round(max(0.0, raw_gap), 4)
    if gap is None:
        verdict = "unknown"
    elif gap <= 0:
        verdict = "met"
    else:
        verdict = "missed"
    return {
        "targets": targets,
        "current": current,
        "gap": gap,
        "degraded": bool(SOFT_GOAL_DEGRADED.get(key, False)),
        "verdict": verdict,
        "evidence": SOFT_GOAL_EVIDENCE.get(key, ""),
        "note": "软目标，不阻塞完成",
    }


def _demo_batches(tasks: list[dict]) -> list[dict]:
    """构造 v3 批次时间线：每批记录前/后指标与决策。"""
    batches: list[dict] = []
    for task in tasks:
        key = str(task.get("key") or "")
        metric = str(task.get("primary_metric") or "")
        if not metric:
            continue
        try:
            previous = float(task.get("baseline"))
        except (TypeError, ValueError):
            previous = None
        for rnd in task.get("rounds", []):
            if not isinstance(rnd, dict) or rnd.get("status") != "done":
                continue
            metrics = rnd.get("metrics")
            if not isinstance(metrics, dict) or metrics.get(metric) is None:
                continue
            try:
                value = float(metrics[metric])
            except (TypeError, ValueError):
                continue
            batches.append(
                {
                    "index": len(batches) + 1,
                    "task": key,
                    "before_metrics": {} if previous is None else {metric: round(previous, 4)},
                    "after_metrics": {metric: round(value, 4)},
                    "decision": rnd.get("decision"),
                    "note": str(rnd.get("change_summary") or "")[:160],
                }
            )
            if rnd.get("decision") == "promote":
                previous = value
    return batches


def _demo_knowledge() -> dict:
    lessons_hits = sum(int(lesson.get("uses") or 0) for lesson in LESSONS)
    return {
        "hits": sum(int(doc.get("hits") or 0) for doc in KNOWLEDGE_DOCS) + lessons_hits,
        "hits_detail": f"知识库召回 {len(KNOWLEDGE_DOCS)} 篇文档 / lessons 引用 {lessons_hits} 次",
        "docs": [dict(doc) for doc in KNOWLEDGE_DOCS],
        "skills_used": [dict(item) for item in SKILL_USAGE],
    }


def _task_memory(task: dict, lessons_by_task: dict[str, list[dict]]) -> dict:
    key = str(task.get("key") or "")
    lessons = lessons_by_task.get(key, [])
    return {
        "enabled": True,
        "entries": len(lessons),
        "snapshot_chars": 0,  # 写入 memory 文件后回填
        "review": {
            "task": key,
            "lessons": [
                {
                    "task": lesson["task"],
                    "text": lesson["text"],
                    "scope": lesson["scope"],
                    "evidence": lesson["evidence"],
                    "wins": lesson["wins"],
                    "uses": lesson["uses"],
                }
                for lesson in lessons
            ],
            "db_ids": [],
            "memory_updates": [f"{key}: 追加 {len(lessons)} 条经验"],
        },
        "curated": [lesson["text"] for lesson in lessons[:2]],
        "entries_after": len(lessons),
    }


def _strategy_stats(task: dict) -> list[dict]:
    key = str(task.get("key") or "")
    direction = task.get("direction")
    baseline = float(task.get("baseline") or 0.0)
    metric = str(task.get("primary_metric") or "score")
    buckets: dict[tuple[str, str], dict] = {}
    for rnd in task.get("rounds", []):
        if not isinstance(rnd, dict) or rnd.get("status") != "done":
            continue
        strategy = str(rnd.get("strategy") or "unknown")
        post_type = _postprocess_type(rnd.get("postprocess"))
        metrics = rnd.get("metrics") or {}
        value = metrics.get(metric)
        delta = _metric_improvement(task, value, baseline) if direction else None
        bucket = buckets.setdefault(
            (strategy, post_type),
            {"task": key, "strategy": strategy, "postprocess_type": post_type, "trials": 0, "promotions": 0, "deltas": []},
        )
        bucket["trials"] += 1
        if rnd.get("decision") == "promote":
            bucket["promotions"] += 1
        if delta is not None:
            bucket["deltas"].append(delta)
    rows: list[dict] = []
    for bucket in buckets.values():
        deltas = bucket.pop("deltas")
        bucket["mean_delta"] = round(sum(deltas) / len(deltas), 4) if deltas else None
        bucket["best_delta"] = round(max(deltas), 4) if deltas else None
        bucket["win_rate"] = round(bucket["promotions"] / bucket["trials"], 4) if bucket["trials"] else 0.0
        bucket["updated_at"] = str(task.get("rounds", [{}])[-1].get("created_at") or "")
        rows.append(bucket)
    rows.sort(key=lambda row: (-row["win_rate"], -row["trials"]))
    return rows


# --------------------------------------------------------------------------- #
# 记忆文件 / sqlite / jsonl
# --------------------------------------------------------------------------- #
def _memory_texts(lessons_by_task: dict[str, list[dict]], tasks: list[dict]) -> dict[str, str]:
    lines = ["# 全局记忆（curated）", "", "运行级计划与跨任务约束：", ""]
    for task in tasks:
        key = str(task.get("key") or "")
        lines.append(f"- [{key}] 目标：{GOAL_TEXT.get(key, '')}（当前 {task.get('current')} / 基线 {task.get('baseline')}）")
    lines += [
        "- 预算护栏：实验次数与 token 上限同时约束，接近 80% 时只做低风险策略。",
        "- 回滚规则：候选指标未超过已接受检查点或出现数据泄漏时立即 rollback。",
    ]
    texts = {"MEMORY.md": "\n".join(lines) + "\n"}
    for task in tasks:
        key = str(task.get("key") or "")
        task_lessons = lessons_by_task.get(key, [])
        body = [f"# 任务 {key} 记忆快照", ""]
        if task_lessons:
            body.append("## 已验证经验")
            body.append("")
            for lesson in task_lessons:
                body.append(f"- (wins={lesson['wins']}) {lesson['text']}")
        else:
            body.append("- 暂无任务级经验，等待第一轮决策。")
        body += ["", "## 最近证据", "", f"- 当前指标 {task.get('current')}，状态 {task.get('status')}，已完成轮次 {task.get('round_index')}。"]
        texts[f"TASK_{key}.md"] = "\n".join(body) + "\n"
    return texts


def _write_db(path: Path, lessons: list[dict], strategy_rows: list[dict], run_id: str, started_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        con = sqlite3.connect(str(tmp))
        try:
            con.executescript(
                """
                CREATE TABLE lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT NOT NULL,
                    scope TEXT,
                    text TEXT NOT NULL,
                    evidence TEXT,
                    wins INTEGER DEFAULT 0,
                    uses INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE strategy_stats (
                    task TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    postprocess_type TEXT,
                    trials INTEGER DEFAULT 0,
                    promotions INTEGER DEFAULT 0,
                    mean_delta REAL,
                    best_delta REAL,
                    updated_at TEXT
                );
                CREATE TABLE skill_usage (
                    skill TEXT NOT NULL,
                    run_id TEXT,
                    uses INTEGER DEFAULT 0,
                    last_used TEXT
                );
                """
            )
            now = started_at
            con.executemany(
                "INSERT INTO lessons(task, scope, text, evidence, wins, uses, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        lesson["task"],
                        lesson["scope"],
                        lesson["text"],
                        json.dumps(lesson["evidence"], ensure_ascii=False),
                        lesson["wins"],
                        lesson["uses"],
                        now,
                        now,
                    )
                    for lesson in lessons
                ],
            )
            con.executemany(
                "INSERT INTO strategy_stats(task, strategy, postprocess_type, trials, promotions, mean_delta, best_delta, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        row["task"],
                        row["strategy"],
                        row["postprocess_type"],
                        row["trials"],
                        row["promotions"],
                        row["mean_delta"],
                        row["best_delta"],
                        row["updated_at"],
                    )
                    for row in strategy_rows
                ],
            )
            con.executemany(
                "INSERT INTO skill_usage(skill, run_id, uses, last_used) VALUES (?,?,?,?)",
                [(item["skill"], run_id, item["uses"], now) for item in SKILL_USAGE],
            )
            con.commit()
        finally:
            con.close()
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _records_lines(state: dict) -> tuple[str, str]:
    run = state.get("run") or {}
    run_id = str(run.get("run_id") or "")
    events = [e for e in state.get("events") or [] if isinstance(e, dict)]
    records: list[dict] = []
    for event in events:
        records.append(
            {
                "ts": event.get("ts"),
                "type": "event",
                "run_id": run_id,
                "level": event.get("level", "info"),
                "task": event.get("task") or "-",
                "message": event.get("message", ""),
            }
        )
    rounds_lines: list[dict] = []
    for task in state.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        for rnd in task.get("rounds") or []:
            if not isinstance(rnd, dict):
                continue
            rows = {
                "ts": rnd.get("created_at"),
                "type": "round",
                "run_id": run_id,
                "task": task.get("key"),
                "index": rnd.get("index"),
                "id": rnd.get("id"),
                "status": rnd.get("status"),
                "decision": rnd.get("decision"),
                "strategy": rnd.get("strategy"),
                "postprocess": rnd.get("postprocess"),
                "metrics": rnd.get("metrics"),
                "hypothesis": rnd.get("hypothesis"),
                "change_summary": rnd.get("change_summary"),
                "created_at": rnd.get("created_at"),
            }
            rounds_lines.append(rows)
            if rnd.get("decision"):
                records.append(
                    {
                        "ts": rnd.get("created_at"),
                        "type": "decision",
                        "run_id": run_id,
                        "task": task.get("key"),
                        "round": rnd.get("index"),
                        "decision": rnd.get("decision"),
                        "message": f"{task.get('key')} 第 {rnd.get('index')} 轮决策 {rnd.get('decision')}：{rnd.get('change_summary')}",
                    }
                )
        for panel in task.get("panel") or []:
            if not isinstance(panel, dict):
                continue
            records.append(
                {
                    "ts": (task.get("rounds") or [{}])[-1].get("created_at"),
                    "type": "panel",
                    "run_id": run_id,
                    "task": task.get("key"),
                    "round": panel.get("round_index"),
                    "verdict": panel.get("verdict"),
                    "message": f"{task.get('key')} panel verdict={panel.get('verdict')}，{len(panel.get('messages') or [])} 位角色发言",
                }
            )
    records.sort(key=lambda row: str(row.get("ts") or ""))
    records_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"
    rounds_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rounds_lines) + "\n"
    return records_text, rounds_text


# --------------------------------------------------------------------------- #
# 组装与写出
# --------------------------------------------------------------------------- #
def build_demo_state(now: datetime | None = None) -> dict:
    """v1 场景 → 叠加 v2 内核字段与 v3 软目标/批次/知识字段的完整 state（不写盘）。"""
    mock = _load_mock_module()
    state = mock.build_state(now=now)
    state["schema_version"] = SCHEMA_VERSION
    state["_demo"] = True
    state["_demo_schema"] = "v3"

    run = state["run"]
    run_id = str(run.get("run_id") or "")
    started_at = str(run.get("started_at") or "")
    tasks = [task for task in state.get("tasks") or [] if isinstance(task, dict)]
    last_ts = _last_ts(tasks, started_at)

    lessons_by_task: dict[str, list[dict]] = {}
    for lesson in LESSONS:
        lessons_by_task.setdefault(lesson["task"], []).append(lesson)

    all_strategy_rows: list[dict] = []
    total_trace_tools = 0
    total_panels = 0
    total_memory_writes = 0
    for task in tasks:
        key = str(task.get("key") or "")
        controller = CONTROLLER_BY_TASK.get(key, "pipeline")
        task["controller"] = controller
        drift = task.get("drift") if isinstance(task.get("drift"), dict) else {}
        drift["type"] = DRIFT_TYPE.get(key, "unknown")
        drift["attribution"] = list(DRIFT_ATTRIBUTION.get(key, []))
        task["drift"] = drift
        task["goal_soft"] = _task_goal_soft(task)
        rounds = _task_rounds_agent(task, controller, started_at, last_ts)
        task["rounds"] = rounds
        total_trace_tools += sum(len(rnd["agent"]["trace"]) for rnd in rounds)
        panels = [p for p in task.get("panel") or [] if isinstance(p, dict)]
        total_panels += len(panels)
        strategy_rows = _strategy_stats(task)
        all_strategy_rows.extend(strategy_rows)
        task["strategy_stats"] = strategy_rows
        task["goal"] = _task_goal(task, run_id, started_at)
        task["memory"] = _task_memory(task, lessons_by_task)
        total_memory_writes += len(lessons_by_task.get(key, []))
        if controller == "kernel":
            failures = sum(1 for rnd in rounds if rnd["agent"].get("fallback"))
            task["agent"] = {
                "controller": "kernel",
                "kernel_rounds": len(rounds),
                "kernel_failures": failures,
                "delegates": len(panels) * 2,
                "panels": len(panels),
                "memory_writes": len(lessons_by_task.get(key, [])),
                "skill_writes": min(3, len(rounds)),
                "kernel_tokens": sum(rnd["agent"]["usage"]["input"] + rnd["agent"]["usage"]["output"] for rnd in rounds),
            }
        else:
            task["agent"] = {
                "controller": "pipeline",
                "pipeline_rounds": len(rounds),
                "panels": len(panels),
                "memory_writes": len(lessons_by_task.get(key, [])),
            }

    state["knowledge"] = _demo_knowledge()
    state["batches"] = _demo_batches(tasks)
    state["batch_index"] = len(state["batches"])

    memory_texts = _memory_texts(lessons_by_task, tasks)
    total_chars = sum(len(text) for text in memory_texts.values())
    usage_pct = round(total_chars * 100 / MEMORY_CAP_CHARS)
    entries = sum(
        1
        for text in memory_texts.values()
        for line in text.splitlines()
        if line.strip().startswith("- ")
    )
    run["memory"] = {
        "enabled": True,
        "dir": "",  # write_demo_bundle 回填绝对路径
        "db": "",
        "entries": entries,
        "usage": f"{usage_pct}% — {total_chars}/{MEMORY_CAP_CHARS} chars",
    }
    run["agent"] = {
        "controller": "kernel",
        "kernel_rounds": sum(1 for task in tasks if task.get("controller") == "kernel" for _ in task.get("rounds") or []),
        "kernel_failures": sum(1 for task in tasks for rnd in task.get("rounds") or [] if (rnd.get("agent") or {}).get("fallback")),
        "pipeline_rounds": sum(1 for task in tasks if task.get("controller") == "pipeline" for _ in task.get("rounds") or []),
        "delegates": total_panels * 2,
        "panels": total_panels,
        "memory_writes": total_memory_writes,
        "skill_writes": len(SKILL_USAGE),
        "kernel_tokens": sum(
            (rnd.get("agent") or {}).get("usage", {}).get("input", 0) + (rnd.get("agent") or {}).get("usage", {}).get("output", 0)
            for task in tasks
            for rnd in task.get("rounds") or []
        ),
    }
    state["_demo_memory_texts"] = memory_texts
    state["_demo_strategy_rows"] = all_strategy_rows
    state["_demo_lessons"] = LESSONS
    state["_demo_skill_usage"] = SKILL_USAGE
    return state


def write_demo_bundle(state_path: str | Path, *, include_db: bool = True) -> Path:
    """生成完整 v3 演示包（state.json + records/rounds + memory/*.md + rsi_state.db），返回 state 路径。"""
    state_path = Path(state_path).expanduser().resolve()
    root = state_path.parent
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "records.jsonl"
    rounds_path = root / "rounds.jsonl"
    memory_dir = root / "memory"
    db_path = root / "rsi_state.db"

    state = build_demo_state()
    memory_texts: dict[str, str] = state.pop("_demo_memory_texts")
    all_strategy_rows: list[dict] = state.pop("_demo_strategy_rows")
    lessons: list[dict] = state.pop("_demo_lessons")
    skill_usage: list[dict] = state.pop("_demo_skill_usage")
    state.pop("_demo_schema", None)

    memory_dir.mkdir(parents=True, exist_ok=True)
    for name, text in memory_texts.items():
        _atomic_write_text(memory_dir / name, text)
    for task in state.get("tasks") or []:
        key = str(task.get("key") or "")
        file_name = f"TASK_{key}.md"
        if task.get("memory") is not None:
            task["memory"]["snapshot_chars"] = len(memory_texts.get(file_name, ""))

    if include_db:
        _write_db(db_path, lessons, all_strategy_rows, str(state["run"].get("run_id") or ""), str(state["run"].get("started_at") or ""))
        state["run"]["memory"]["db"] = str(db_path)
    state["run"]["memory"]["dir"] = str(memory_dir)

    records_text, rounds_text = _records_lines(state)
    _atomic_write_text(records_path, records_text)
    _atomic_write_text(rounds_path, rounds_text)
    _atomic_write_text(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    return state_path


__all__ = ["SCHEMA_VERSION", "MEMORY_CAP_CHARS", "build_demo_state", "write_demo_bundle", "SKILL_USAGE"]
