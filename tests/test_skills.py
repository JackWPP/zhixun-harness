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
