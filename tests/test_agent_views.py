from __future__ import annotations

import asyncio
import json

from rsi.agent.views import agent_view, heldout_text


def test_agent_view_withholds_heldout_fields_without_mutating_audit_payload():
    payload = {
        "metrics": {
            "selection": {"primary": 0.73},
            "report": {"primary": 0.68},
            "probe_metrics": {"primary": 0.52},
        },
        "experiments": [{
            "metrics": {"selection": {"primary": 0.74}, "report": {"primary": 0.67}},
            "log_path": "/work/run/exp.log",
        }],
        "hypothesis": "test selection stability",
    }

    projected = agent_view(payload)

    assert projected["metrics"] == {"selection": {"primary": 0.73}}
    assert projected["experiments"][0]["metrics"] == {"selection": {"primary": 0.74}}
    assert "log_path" not in projected["experiments"][0]
    assert payload["metrics"]["report"]["primary"] == 0.68
    assert payload["experiments"][0]["log_path"].endswith("exp.log")


def test_agent_view_withholds_legacy_heldout_prose_as_a_whole():
    assert heldout_text("probe score improved; try this next round")
    assert agent_view("probe score improved; try this next round") == "[held-out evidence withheld]"
    assert agent_view("selection score improved; try this next round") == (
        "selection score improved; try this next round"
    )


def test_serialized_json_is_projected_before_it_can_be_clipped():
    raw = json.dumps({"report_metrics": {"primary": 0.1}, "selection": {"primary": 0.7}})
    assert agent_view(raw) == json.dumps({"selection": {"primary": 0.7}}, ensure_ascii=False)


def test_kernel_experiment_tool_hides_heldout_metrics_but_keeps_audit_result():
    from rsi.agent.actions import KernelEnv, build_kernel_tools
    from rsi.agent.guard import Guard
    from rsi.executor import ExperimentProposal
    from rsi.tasks import resolve_tasks

    class Runtime:
        class Budget:
            cap = 1000

            @staticmethod
            def exhausted():
                return False

        budget = Budget()

    class Host:
        @staticmethod
        def context():
            return {}

        @staticmethod
        def parent_primary():
            return 0.7

        @staticmethod
        def run_experiment(proposal):
            return {
                "experiment_id": "exp-1",
                "ok": True,
                "proposal": {"strategy": proposal.strategy},
                "metrics": {
                    "selection": {"primary": 0.72},
                    "report": {"primary": 0.11},
                    "probe": {"primary": 0.22},
                },
            }

    config = resolve_tasks(["A"])[0]
    env = KernelEnv(task=config, host=Host(), guard=Guard(config), runtime=Runtime())
    env.proposals["plan-01"] = ExperimentProposal("retain", {}, {})
    tool = next(item for item in build_kernel_tools(env) if item.name == "run_experiment")
    result = asyncio.run(tool.execute("call-1", tool.parameters(plan_id="plan-01")))
    visible = result.content[0]["text"]

    assert '"selection"' in visible and "0.72" in visible
    assert "report" not in visible and "probe" not in visible
    assert env.results["exp-1"]["metrics"]["report"]["primary"] == 0.11
    assert env.results["exp-1"]["metrics"]["probe"]["primary"] == 0.22


def test_record_search_projects_full_audit_payload_before_matching(tmp_path):
    from rsi.agent.views import search_selection_records
    from rsi.memory import Recall, StateDB

    db = StateDB(tmp_path / "records.db")
    db.add_round(
        "run-1", "A", 1, "retain", strategy="recenter", primary=0.72,
        hypothesis="check selection recenter", payload={
            "metrics": {
                "selection": {"primary": 0.72},
                "report": {"primary": 0.11},
                "probe": {"primary": 0.22},
            },
        })
    recall = Recall(db=db, memory=None, skills=None)

    found = search_selection_records(recall, "recenter", task="A")

    assert len(found) == 1
    text = json.dumps(found[0], ensure_ascii=False)
    assert "selection" in text and "0.72" in text
    assert "report" not in text and "probe" not in text
    assert "0.11" not in text and "0.22" not in text
    db.close()
