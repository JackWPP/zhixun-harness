from __future__ import annotations

import json

from rsi.memory.db import StateDB


def _db(tmp_path) -> StateDB:
    return StateDB(tmp_path / "state.db")


def test_runs_rounds_experiments(tmp_path):
    db = _db(tmp_path)
    db.add_run("r1", "2026-01-01T00:00:00+00:00", controller="kernel",
               llm_model="qwen/qwen-3.8-27b", tasks="A,B,C", seed=42)
    assert db.run_count() == 1
    db.finish_run("r1", "done", "{}")
    db.add_round("r1", "A", 1, "promote", "retain", "recenter(batch_mean, 0.4)",
                 primary=0.72, delta=0.017, hypothesis="drift", payload={"x": 1})
    db.add_round("r1", "A", 2, "retain", "retain", "none", primary=0.72, delta=0.0)
    assert len(db.rounds("A")) == 2
    db.add_experiment("r1", "A", 1, "exp-0001", "retain", "recenter", True,
                      0.72, 0.017, 3.5, {"metrics": {}})
    db.add_llm_call("r1", "A", "diagnostician", True,
                    {"input": 100, "output": 50, "reasoning": 20}, 12.5)
    totals = db.llm_totals("r1")
    assert totals["calls"] == 1 and totals["input"] == 100
    db.close()


def test_lessons_dedupe_and_scope(tmp_path):
    db = _db(tmp_path)
    first = db.add_lesson("A", "recenter wins", scope="retain+recenter", wins=1)
    second = db.add_lesson("A", "recenter wins", scope="retain+recenter", wins=2)
    assert first == second
    db.add_lesson("A", "avoid continue_train", scope="continue_train")
    assert len(db.lessons("A")) == 2
    scoped = db.lessons("A", scope_contains="recenter")
    assert scoped and scoped[0]["wins"] == 2
    db.bump_lessons([first])
    assert db.lessons("A")[0]["uses"] == 1
    db.close()


def test_strategy_stats_accumulate(tmp_path):
    db = _db(tmp_path)
    db.record_strategy("A", "retain", "recenter", 0.02, True)
    db.record_strategy("A", "retain", "recenter", -0.01, False)
    stats = db.strategy_stats("A")
    assert len(stats) == 1
    item = stats[0]
    assert item["trials"] == 2 and item["promotions"] == 1
    assert abs(item["win_rate"] - 0.5) < 1e-9
    assert abs(item["mean_delta"] - 0.005) < 1e-9
    assert abs(item["best_delta"] - 0.02) < 1e-9
    db.close()


def test_full_text_search_and_goals(tmp_path):
    db = _db(tmp_path)
    db.index("round", "A:1", "A", "batch recentering fixed the sensor gain drift")
    db.index("lesson", "A", "A", "Toluene class has only five validation rows")
    hits = db.search("recenter")
    assert hits, "FTS/LIKE search must find the indexed document"
    assert any("recenter" in (hit["text"] or "") for hit in hits)
    db.save_goal("run-1", {"goal": "10 rounds", "turns_used": 3,
                           "status": "active", "subgoals": []})
    loaded = db.load_goal("run-1")
    assert loaded and loaded["turns_used"] == 3
    db.close()


def test_replay_from_jsonl(tmp_path):
    rounds_path = tmp_path / "rounds.jsonl"
    rows = [
        {"type": "round", "run_id": "r1", "task": "A", "round_index": 1,
         "decision": "promote", "strategy": "retain",
         "postprocess": "recenter(batch_mean, 0.4)", "delta": 0.01},
        {"type": "round", "run_id": "r1", "task": "A", "round_index": 2,
         "decision": "retain", "strategy": "retain", "postprocess": "none"},
    ]
    rounds_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                           encoding="utf-8")
    db = _db(tmp_path)
    count = db.replay_from_jsonl(rounds_path)
    assert count == 2
    assert len(db.rounds("A")) == 2
    db.close()


def test_skill_usage_aggregate(tmp_path):
    db = _db(tmp_path)
    db.record_skill_use("drift-diagnosis", "r1")
    db.record_skill_use("drift-diagnosis", "r1")
    db.record_skill_use("drift-diagnosis", "r2")
    assert db.skill_uses("drift-diagnosis") == 3
    usage = db.skill_usage()
    assert usage[0]["skill"] == "drift-diagnosis" and usage[0]["uses"] == 3
    db.close()
