#!/usr/bin/env python3
"""Build the deliverable submission bundle from a finished run.

    ./.venv/bin/python scripts/make_submission.py --run-dir .scratch/rsi-canonical/<run>

Creates ``dist/submission_<run_id>/`` with one directory per task containing
``checkpoint.npz``, ``postprocess.json``, a standalone predict entry and a
lineage manifest, then packs the whole thing into ``submission_<run_id>.tar.gz``
and asserts the platform limit (<= 64 MiB).

The bundle is evidence/protocol-shaped, not a platform API call: uploading is a
separate, human-approved step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIMIT_BYTES = 64 * 1024 * 1024
PREDICT_ENTRY = '''#!/usr/bin/env python3
"""Standalone predictor for this task bundle (official stdin/stdout contract).

    python predict.py --checkpoint checkpoint.npz --postprocess postprocess.json < input.jsonl

Reads one JSON object per line: {"task": "...", "inputs": {...}}
Writes one JSON object per line:
  classification {"outputs": [{"label": ..., "score": ...}, ...]}
  regression     {"outputs": {"value": float}}
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rsi.inference import apply_postprocess, format_outputs, resolve_postprocess  # noqa: E402
from rsi.model import ExtendedModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(HERE / "checkpoint.npz"))
    parser.add_argument("--postprocess", default=str(HERE / "postprocess.json"))
    args = parser.parse_args()
    model = ExtendedModel.from_checkpoint(args.checkpoint)
    postprocess = json.loads(Path(args.postprocess).read_text())
    rows = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        return 0
    import numpy as np
    from rsi.data_official import flatten

    matrix = [flatten(row["inputs"])[1] for row in rows]
    predicted = apply_postprocess(resolve_postprocess(postprocess, model), model, np.asarray(matrix))
    for index in range(len(rows)):
        output = format_outputs(predicted[index], model.meta["task"], model.meta["classes"])
        print(json.dumps(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def find_checkpoint(run_dir: Path, task: dict) -> Path:
    explicit = task.get("checkpoint_path")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    rounds = task.get("rounds") or []
    sha = (task.get("checkpoint_sha256")
           or (rounds[-1].get("checkpoint_sha256") if rounds else "") or "").replace("sha256:", "")
    if not sha:
        raise SystemExit(f"task {task.get('key')}: no checkpoint path or sha in state")
    for candidate in run_dir.glob(f"{task['key']}/experiments/*/candidate.npz"):
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == sha:
            return candidate
    raise SystemExit(f"task {task['key']}: cannot locate checkpoint with sha {sha}")


def build(run_dir: Path, output_root: Path) -> Path:
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary_by_key = {dataset["key"]: dataset for dataset in summary["datasets"]}
    bundle = output_root / f"submission_{state['run']['run_id']}"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    for task in state["tasks"]:
        key = task["key"]
        task_dir = bundle / key
        task_dir.mkdir()
        checkpoint = find_checkpoint(run_dir, task)
        shutil.copy2(checkpoint, task_dir / "checkpoint.npz")
        postprocess = task.get("postprocess") or {"type": "none"}
        (task_dir / "postprocess.json").write_text(json.dumps(postprocess, indent=2) + "\n",
                                                  encoding="utf-8")
        (task_dir / "predict.py").write_text(PREDICT_ENTRY, encoding="utf-8")
        (task_dir / "predict.py").chmod(0o755)
        rsi_dir = task_dir / "rsi"
        rsi_dir.mkdir()
        for name in ("__init__.py", "inference.py", "model.py", "data_official.py"):
            shutil.copy2(ROOT / "rsi" / name, rsi_dir / name)
        dataset_summary = summary_by_key.get(key, {})
        manifest = {
            "task": key,
            "task_type": task.get("task"),
            "primary_metric": task.get("primary_metric"),
            "checkpoint_sha256": (task.get("checkpoint_sha256")
                                  or (task["rounds"][-1].get("checkpoint_sha256") if task.get("rounds") else None)),
            "parent_checkpoint_sha256": ((task.get("rounds") or [{}])[-1].get("parent_checkpoint_sha256")
                                         if task.get("rounds") else None),
            "baseline_checkpoint_sha256": dataset_summary.get("baseline_checkpoint_sha256"),
            "postprocess": postprocess,
            "metrics": dataset_summary.get("metrics", {}),
            "controller": task.get("controller"),
            "degraded": task.get("degraded", False),
            "rounds": [{"index": round.get("index"), "decision": round.get("decision"),
                        "delta": round.get("delta"),
                        "hypothesis": round.get("hypothesis"),
                        "change_summary": round.get("change_summary"),
                        "validation_metrics": round.get("validation_metrics"),
                        "parent_checkpoint_sha256": round.get("parent_checkpoint_sha256"),
                        "checkpoint_sha256": round.get("checkpoint_sha256"),
                        "code_sha256": round.get("code_sha256")} for round in task.get("rounds", [])],
            "files": {path.name: sha256(path) for path in sorted(task_dir.rglob("*")) if path.is_file()},
            "note": "built by scripts/make_submission.py; prediction is independent of training",
        }
        (task_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")

    archive = output_root / f"{bundle.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
    size = archive.stat().st_size
    if size > LIMIT_BYTES:
        raise SystemExit(f"submission archive {size} bytes exceeds 64 MiB limit")
    print(f"bundle:  {bundle}")
    print(f"archive: {archive} ({size / 1024 / 1024:.2f} MiB)")
    print(f"sha256:  {sha256(archive)}")
    print(f"tasks:   {', '.join(task['key'] for task in state['tasks'])}; limit ok (<= 64 MiB)")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-root", default="dist")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser()
    if not (run_dir / "state.json").is_file():
        raise SystemExit(f"state.json not found under {run_dir}")
    build(run_dir, Path(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
