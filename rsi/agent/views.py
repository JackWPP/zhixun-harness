"""Selection-only agent views; raw audit data stays untouched.

Projection precedes truncation so a clipped JSON/FTS snippet cannot lose the key
identifying its held-out origin. Legacy prose explicitly naming a held-out
source is withheld whole. Old unlabelled prose has no reliable provenance and
must not be reused in a strict clean run.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping


_HELDOUT_KEY = re.compile(
    r"(?<![a-z])(?:report|probe|hidden|external|holdout|held[_ -]?out)"
    r"(?:[_ -](?:metrics?|scores?|primary|rmse|f1|macro_f1|results?|evidence|split))?"
    r"(?![a-z])|隐藏(?:集|测试|标签)|外部(?:探针|指标|分数)|报告集|留出集",
    re.I,
)
_HELDOUT_TEXT = re.compile(
    r"(?<![a-z])(?:report|probe|holdout|held[_ -]?out)(?![a-z])|"
    r"(?<![a-z])(?:hidden|external)[_ -]+(?:metrics?|scores?|primary|rmse|f1|results?|evidence)(?![a-z])|"
    r"外部(?:探针|指标|分数)|报告集|留出集|隐藏(?:集|测试)[^。\n]{0,12}(?:指标|分数)",
    re.I,
)
_RAW_FIELDS = {"payload_json", "summary_json", "log_tail", "traceback", "log_path"}
_WITHHELD = "[held-out evidence withheld]"


def heldout_text(text: str) -> bool:
    return bool(_HELDOUT_TEXT.search(text or ""))


def agent_view(value):
    """Return a detached projection, never mutate audit objects.

    Parse serialized JSON before projecting. Withhold arbitrary prose as a
    whole if it names a held-out source: its number/recommendation may be on a
    different line from the split name.
    """
    if isinstance(value, Mapping):
        return {str(key): agent_view(item) for key, item in value.items()
                if str(key).lower() not in _RAW_FIELDS
                and not _HELDOUT_KEY.search(str(key))}
    if isinstance(value, (list, tuple)):
        return [agent_view(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError):
                pass
            else:
                return json.dumps(agent_view(parsed), ensure_ascii=False)
        return _WITHHELD if heldout_text(value) else value
    return value


def search_selection_records(recall, query: str, *, task: str, limit: int = 8) -> list[dict]:
    """Search projected structured rounds, never raw FTS snippets.

    FTS indexes full audit experiments. Its snippets can contain a held-out
    score without its split key, so post-filtering snippets is insufficient.
    """
    db = getattr(recall, "db", None)
    if db is None or heldout_text(query):
        return []
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", query or "")[:10]
    if not tokens:
        return []
    tokens = [token.casefold() for token in tokens]
    matches = []
    for row in db.rounds(task=task, limit=1000):
        payload = row.get("payload_json")
        try:
            payload = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            payload = {}
        visible = agent_view({
            "run_id": row.get("run_id"), "task": row.get("task"),
            "round_index": row.get("round_index"), "decision": row.get("decision"),
            "strategy": row.get("strategy"), "postprocess": row.get("postprocess"),
            "selection_primary": row.get("primary_value"), "delta": row.get("delta"),
            "hypothesis": row.get("hypothesis"), "record": payload,
        })
        text = json.dumps(visible, ensure_ascii=False).casefold()
        if any(token in text for token in tokens):
            matches.append(visible)
            if len(matches) >= limit:
                break
    return matches
