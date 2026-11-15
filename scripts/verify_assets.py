#!/usr/bin/env python3
"""Pre-flight asset self-check for the packaged harness.

Runs inside the image (entrypoint calls it before `run`) or on the host:

    ./.venv/bin/python scripts/verify_assets.py --json-out .scratch/assets.json

Checks: three official packages (files + sha256), checkpoints load with the
contest loader, strict JSONL splits read, seed skills validate, predict entry is
importable, the state DB can be created.  Exit code 1 on any hard failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: list[dict] = []


def check(name: str, fn, *, required: bool = True) -> bool:
    try:
        detail = fn()
        RESULTS.append({"check": name, "ok": True, "required": required, "detail": detail})
        print(f"[ok]   {name}: {detail}")
        return True
    except Exception as exc:  # noqa: BLE001 - report every failure, never crash
        RESULTS.append({"check": name, "ok": False, "required": required,
                        "error": f"{exc.__class__.__name__}: {exc}"})
        print(f"[FAIL] {name}: {exc.__class__.__name__}: {exc}")
        return not required


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--skills-root", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    from rsi.data import load_jsonl_dataset
    from rsi.model import ExtendedModel, checkpoint_sha256
    from rsi.memory.skills import SkillStore
    from rsi.memory.db import StateDB
    from rsi.knowledge import default_knowledge
    from rsi.tasks import TASKS

    if args.data_root:
        import os
        os.environ["RSI_DATA_ROOT"] = args.data_root

    hard_ok = True

    def datasets() -> str:
        counts = {}
        for key in "ABC":
            config = TASKS[key]
            for name in ("train", "validation"):
                path = config.train_path if name == "train" else config.validation_path
                split = load_jsonl_dataset(path, config.task, config.classes)
                counts[f"{key}:{name}"] = len(split)
        return json.dumps(counts)

    def checkpoints() -> str:
        out = {}
        for key in "ABC":
            config = TASKS[key]
            model = ExtendedModel.from_checkpoint(config.checkpoint_path)
            out[key] = {"params": int(model.meta.get("parameter_count", 0)),
                        "sha256": checkpoint_sha256(config.checkpoint_path)}
        return json.dumps(out)

    def packages() -> str:
        out = {}
        for key in "ABC":
            config = TASKS[key]
            base = Path(config.package_dir)
            files = sorted(p.name for p in base.rglob("*") if p.is_file())
            out[key] = {"dir": str(base), "files": len(files)}
        return json.dumps(out)

    def knowledge() -> str:
        kb = default_knowledge()
        kb.ensure_index()
        stats = kb.stats()
        if int(stats.get("docs", 0)) < 10:
            raise ValueError(f"knowledge index too small: {stats}")
        return json.dumps(stats)

    def skills() -> str:
        root = Path(args.skills_root) if args.skills_root else ROOT / "skills"
        store = SkillStore(root)
        names = [skill.name if hasattr(skill, "name") else str(skill) for skill in store.list()]
        return json.dumps({"skills": names, "index_chars": len(store.index_text())})

    def predict_entry() -> str:
        from rsi.inference import format_outputs  # noqa: F401
        from rsi.predict import main as predict_main  # noqa: F401
        return "rsi.predict importable"

    def db() -> str:
        with tempfile.TemporaryDirectory(prefix="rsi-assets-") as tmp:
            database = StateDB(Path(tmp) / "check.db")
            database.close()
            return "StateDB open/close ok"

    hard_ok &= check("official datasets readable", datasets)
    hard_ok &= check("baseline checkpoints load", checkpoints)
    hard_ok &= check("three packages present", packages)
    hard_ok &= check("seed skills valid", skills, required=False)
    hard_ok &= check("offline knowledge index", knowledge, required=False)
    hard_ok &= check("predict entry importable", predict_entry)
    hard_ok &= check("state db writable", db, required=False)

    payload = {"ok": hard_ok, "root": str(ROOT), "checks": RESULTS}
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"\n[{ 'PASS' if hard_ok else 'FAIL' }] asset self-check")
    return 0 if hard_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
