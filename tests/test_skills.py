from __future__ import annotations

import pytest

from rsi.memory.skills import (MAX_SKILL_BYTES, SkillStore, render_skill,
                               validate_skill_text)
from rsi.memory.store import MemoryError_


def _text(name: str, body: str = "## When\nDrift is detected.\n\n## Procedure\nDo X.") -> str:
    return render_skill(name, "short description", body, tags=["drift"])


def test_create_list_read_patch_delete(tmp_path):
    used = []
    store = SkillStore(tmp_path / "skills", on_use=used.append)
    created = store.create("my-skill", _text("my-skill"))
    assert created["name"] == "my-skill"
    assert store.names() == ["my-skill"]
    index = store.index_text()
    assert "my-skill" in index and "short description" in index
    text = store.read_text("my-skill")
    assert "## Procedure" in text
    assert used == ["my-skill"]
    patched = store.patch("my-skill", "Do X.", "Do X and Y.")
    assert patched["name"] == "my-skill"
    assert "Do X and Y." in store.read_text("my-skill")
    store.delete("my-skill")
    assert "my-skill" not in store.names()
    assert store.archive_root.exists()


def test_invalid_names_and_frontmatter_rejected(tmp_path):
    store = SkillStore(tmp_path / "skills")
    with pytest.raises(MemoryError_):
        store.create("BadName", _text("badname"))
    with pytest.raises(MemoryError_):
        store.create("no-frontmatter", "just some text without frontmatter")
    with pytest.raises(MemoryError_):
        store.create("wrong-name", _text("my-skill"))
    with pytest.raises(MemoryError_):
        validate_skill_text(render_skill("my-skill", "", "## When\nx"))
    with pytest.raises(MemoryError_):
        validate_skill_text(_text("my-skill", "## When\nsee https://example.com"))


def test_oversize_and_blocked_content_rejected(tmp_path):
    store = SkillStore(tmp_path / "skills")
    huge = "A" * (MAX_SKILL_BYTES + 10)
    with pytest.raises(MemoryError_):
        store.create("huge-skill", _text("huge-skill", f"## When\n{huge}"))
    with pytest.raises(MemoryError_):
        store.create("shell-skill", _text("shell-skill", "## Procedure\n```bash\nrm -rf /\n```"))
    with pytest.raises(MemoryError_):
        store.create("eval-skill", _text("eval-skill", "## Procedure\neval(payload)"))


def test_seed_skills_cannot_be_deleted_by_agent(tmp_path):
    store = SkillStore(tmp_path / "skills")
    store.create("drift-diagnosis", _text("drift-diagnosis"))
    with pytest.raises(MemoryError_):
        store.delete("drift-diagnosis")
    archived = store.archive("drift-diagnosis")
    assert archived["archived"] is True


def test_patch_requires_unique_old_string(tmp_path):
    store = SkillStore(tmp_path / "skills")
    store.create("my-skill", _text("my-skill", "## Procedure\nStep A. Step A."))
    with pytest.raises(MemoryError_):
        store.patch("my-skill", "Step A.", "Step B.")
    with pytest.raises(MemoryError_):
        store.patch("my-skill", "missing text", "x")


def test_runner_skill_store_is_run_local(tmp_path):
    from rsi.runner import HARNESS_ROOT, Runner

    runner = Runner.__new__(Runner)
    runner.memory_root = tmp_path / "run-state"
    runner.db = None
    store = runner._build_skill_store()
    source = HARNESS_ROOT / "skills" / "drift-diagnosis" / "SKILL.md"
    original = source.read_text(encoding="utf-8")
    before = store.load("drift-diagnosis").body

    store.patch("drift-diagnosis", "无标签阶段不得引用 label_shift 或性能 drop", "临时实验更新")

    assert source.read_text(encoding="utf-8") == original
    assert store.load("drift-diagnosis").body != before
    assert store.root == runner.memory_root / "skills"


def test_agent_skill_edits_are_staged_then_gated_and_rollbackable(tmp_path):
    from rsi.memory.skill_candidates import SkillCandidateStore, _file_hash

    active = SkillStore(tmp_path / "active")
    active.create("my-skill", _text("my-skill", "## Procedure\nParent."))
    before = (active.root / "my-skill" / "SKILL.md").read_text()
    candidates = SkillCandidateStore(active, tmp_path / "candidates")
    candidate = candidates.patch("my-skill", "Parent.", "Candidate.")

    assert candidate["status"] == "staged"
    assert (active.root / "my-skill" / "SKILL.md").read_text() == before
    report = tmp_path / "evaluation.json"
    report.write_text('{"independent":true}\n')
    evaluation = {
        "independent": True,
        "candidate_id": candidate["candidate_id"],
        "parent_skill_hashes": candidate["parent_skill_hashes"],
        "candidate_skill_hashes": candidate["candidate_skill_hashes"],
        "report_path": str(report),
        "report_sha256": _file_hash(report),
        "seeds": [42, 43],
        "seed_metrics": [
            {"seed": 42, "baseline": 0.7, "candidate": 0.72, "worst_group_delta": -0.01},
            {"seed": 43, "baseline": 0.71, "candidate": 0.73, "worst_group_delta": 0.0},
        ],
        "direction": "higher",
        "minimum_delta": 0.01,
        "max_group_regression": 0.02,
    }
    promoted = candidates.promote(candidate["candidate_id"], evaluation)
    assert promoted["status"] == "promoted"
    assert "Candidate." in active.read_text("my-skill")
    rolled = candidates.rollback(candidate["candidate_id"])
    assert rolled["status"] == "rolled_back"
    assert active.read_text("my-skill").find("Parent.") >= 0


def test_skill_candidate_requires_independent_multiseed_gate(tmp_path):
    from rsi.memory.skill_candidates import SkillCandidateStore

    active = SkillStore(tmp_path / "active")
    active.create("my-skill", _text("my-skill"))
    candidates = SkillCandidateStore(active, tmp_path / "candidates")
    candidate = candidates.patch("my-skill", "Do X.", "Do X safely.")
    with pytest.raises(ValueError, match="independent evaluation"):
        candidates.promote(candidate["candidate_id"], {})
