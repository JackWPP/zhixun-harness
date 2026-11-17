#!/usr/bin/env python3
"""L2 batch rehearsal through the real workspace-v2 entrypoint.

Reuses the local fake host from ``scripts/emulate_workspace_v2.py`` but feeds
the platform protocol one labelled batch at a time: each round gets the
cumulative revealed history as ``/data/train.jsonl`` and the history hold-out
as ``/data/validation.jsonl``, uploads a candidate artifact, then predicts the
current batch inputs and scores them outside the harness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from rsi.stream.sources import full_batches
from rsi.tasks import resolve_tasks

ROOT = Path(__file__).resolve().parents[2]


def _load_emulator():
    path = ROOT / "scripts" / "emulate_workspace_v2.py"
    spec = importlib.util.spec_from_file_location("emu_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _macro_f1(expected: list[str], predicted: list[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    scores = []
    for label in labels:
        tp = sum(1 for e, p in zip(expected, predicted) if e == label and p == label)
        fp = sum(1 for e, p in zip(expected, predicted) if e != label and p == label)
        fn = sum(1 for e, p in zip(expected, predicted) if e == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _score(task: str, classes: list[str] | None, answers: dict[str, str], outputs: list[dict],
           prediction_ids: list[str]) -> dict:
    if task == "classification":
        expected = [answers[i] for i in prediction_ids]
        predicted = []
        for output in outputs:
            body = output["outputs"]
            best = max(body, key=lambda item: item["score"])
            predicted.append(best["label"])
        return {"rows": len(expected), "accuracy": float(sum(e == p for e, p in zip(expected, predicted)) / max(1, len(expected))),
                "macro_f1": float(_macro_f1(expected, predicted))}
    values = np.asarray([output["outputs"]["value"] for output in outputs], dtype=np.float64)
    truth = np.asarray([float(answers[i]) for i in prediction_ids], dtype=np.float64)
    return {"rows": len(truth), "rmse": float(np.sqrt(np.mean((values - truth) ** 2))),
            "mae": float(np.mean(np.abs(values - truth)))}



def _predict_all(emu, root: Path, task_key: str, artifact_dir: Path, rows_path: Path,
                 work: Path, timeout: float) -> list[dict]:
    """Run the v2 predict entrypoint over *all* rows (not the 3-row smoke cap)."""

    checkpoint = artifact_dir / "checkpoint.npz"
    manifest = {path.name: path.read_bytes() for path in artifact_dir.iterdir() if path.is_file()}
    artifact = emu.declaration(artifact_dir.name, manifest)
    context = json.dumps({
        "protocol": "contest.predict-context.v1",
        "dataset_key": task_key,
        "task": emu.TASKS[task_key]["task"],
        "artifact": {
            "checkpoint_paths": ["checkpoint.npz"],
            "code_paths": ["submission_meta.json"],
            "predict_argv": ["python3", "/app/harness.py", "predict"],
        },
        "artifact_id": artifact["artifact"],
    }, sort_keys=True)
    env = emu.child_env(root, work, work)
    env["CONTEST_ARTIFACT_PATH"] = str(checkpoint)
    env["CONTEST_ARTIFACT_MANIFEST"] = str(artifact_dir / "manifest.json")
    env["CONTEST_PREDICTION_CONTEXT"] = context
    env["CONTEST_DATASET_KEY"] = task_key
    env["CONTEST_TASK"] = emu.TASKS[task_key]["task"]
    env["EMU_MODEL_DIR"] = str(artifact_dir)
    requests = []
    with rows_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                requests.append(json.dumps({"task": emu.TASKS[task_key]["task"],
                                            "inputs": row["inputs"]}))
    process = subprocess.run(
        [sys.executable, str(emu.APP / "harness.py"), "predict"],
        input="\n".join(requests) + "\n", capture_output=True, text=True,
        cwd=str(work), env=env, timeout=timeout,
    )
    if process.returncode != 0:
        raise RuntimeError("predict failed (%d)\n%s" % (process.returncode, process.stderr))
    outputs = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if len(outputs) != len(requests):
        raise RuntimeError("predict returned %d lines for %d rows" % (len(outputs), len(requests)))
    for output in outputs:
        body = output.get("outputs")
        if emu.TASKS[task_key]["task"] == "classification":
            if not isinstance(body, list) or not body:
                raise RuntimeError("classification output invalid")
            labels = {item["label"] for item in body}
            scores = [item["score"] for item in body]
            if len(labels) != len(body) or any(score < 0 or score > 1 for score in scores):
                raise RuntimeError("classification output invalid")
            if sum(score == max(scores) for score in scores) != 1:
                raise RuntimeError("classification output needs exactly one top score")
        else:
            if not isinstance(body, dict) or set(body) != {"value"}:
                raise RuntimeError("regression output invalid")
            value = body["value"]
            if value < 0 or value != value:
                raise RuntimeError("regression value invalid")
    return outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="A,B,C")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--hidden-tail", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=0,
                        help="cap rounds (0 = all batches)")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default=".scratch/sim/L2")
    args = parser.parse_args(argv)

    emu = _load_emulator()
    keys = [key.strip().upper() for key in args.tasks.split(",") if key.strip()]
    report: dict[str, dict] = {}
    for config in resolve_tasks(keys):
        batches = full_batches(config, n_batches=args.batches,
                               hidden_tail=args.hidden_tail, seed=42)
        if args.rounds:
            batches = batches[: args.rounds]
        root = Path(tempfile.mkdtemp(prefix=f"l2-{config.key}-"))
        work = root / "workspace"
        work.mkdir()
        previous = root / "previous"
        previous.mkdir()
        emu.materialise_sitecustomize(root)
        parent_checkpoint = Path(config.checkpoint_path)
        (previous / "checkpoint.npz").write_bytes(parent_checkpoint.read_bytes())
        parent = {
            "artifact_id": "official-baseline",
            "artifact_sha256": "sha256:" + emu.sha256_file(parent_checkpoint),
            "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                         "predict_argv": ["python3", "/app/harness.py", "predict"]},
            "files": [{"path": "checkpoint.npz",
                       "sha256": "sha256:" + emu.sha256_file(parent_checkpoint),
                       "size_bytes": parent_checkpoint.stat().st_size}],
        }
        history: list[dict] = []
        rounds: list[dict] = []
        try:
            for position, batch in enumerate(batches):
                package = root / f"package-{position:02d}"
                (package / "public").mkdir(parents=True)
                shutil.copy2(parent_checkpoint, package / "checkpoint.npz")
                _write_rows(package / "public" / "train.jsonl", batch.train)
                _write_rows(package / "public" / "validation.jsonl", batch.dev)
                prediction_path = root / f"prediction-{position:02d}.jsonl"
                _write_rows(prediction_path, batch.prediction)
                artifact_dir = root / f"artifact-{position + 1:02d}"
                state_dir = root / f"state-{position + 1:02d}"
                reply = json.dumps({
                    "choice": 1,
                    "hypothesis": "上一批已揭示标签支持校准/续训后复用",
                    "reason": "L2 rehearsal: prefer the validated candidate",
                }, ensure_ascii=False)
                served = emu.run_train(root, config.key, package, position + 1, previous, work,
                                       artifact_dir, state_dir, parent, history, reply,
                                       args.timeout)
                emu.validate_artifact(served)
                checkpoint = artifact_dir / "checkpoint.npz"
                if not checkpoint.is_file():
                    raise RuntimeError("candidate checkpoint missing")
                outputs = _predict_all(emu, root, config.key, artifact_dir, prediction_path,
                                       work, args.timeout)
                if len(outputs) < len(batch.prediction):
                    raise RuntimeError(f"predict returned {len(outputs)} rows for {len(batch.prediction)}")
                metrics = _score(config.task, config.classes, batch.answers, outputs,
                                 [str(row["id"]) for row in batch.prediction])
                record = served["record"]
                rounds.append({
                    "round_index": position + 1,
                    "batch_no": batch.meta.get("batch"),
                    "hidden": bool(batch.meta.get("hidden")),
                    "history_rows": len(batch.train) + len(batch.dev),
                    "prediction_rows": len(batch.prediction),
                    "selected_artifact_id": record["selected_artifact_id"],
                    "metrics": metrics,
                })
                history.append({
                    "id": f"round-{position + 1}",
                    "round_index": position + 1,
                    "selected_artifact_id": record["selected_artifact_id"],
                    "hypothesis": record["hypothesis"],
                    "change_summary": record["change_summary"],
                    "reason": record["reason"],
                })
                parent_checkpoint = checkpoint
                (previous / "checkpoint.npz").write_bytes(checkpoint.read_bytes())
                parent = {
                    "artifact_id": record["selected_artifact_id"],
                    "artifact_sha256": "sha256:" + emu.sha256_file(checkpoint),
                    "artifact": {"checkpoint_paths": ["checkpoint.npz"],
                                 "code_paths": ["submission_meta.json"],
                                 "predict_argv": ["python3", "/app/harness.py", "predict"]},
                    "files": [
                        {"path": "checkpoint.npz", "sha256": "sha256:" + emu.sha256_file(checkpoint),
                         "size_bytes": checkpoint.stat().st_size},
                        {"path": "submission_meta.json",
                         "sha256": "sha256:" + emu.sha256_file(artifact_dir / "submission_meta.json"),
                         "size_bytes": (artifact_dir / "submission_meta.json").stat().st_size},
                    ],
                }
                print(f"[L2:{config.key}] round {position + 1}/{len(batches)} "
                      f"pred={len(outputs)} metrics={metrics}", flush=True)
            report[config.key] = {"ok": True, "rounds": rounds, "root": str(root)}
        except Exception as exc:  # noqa: BLE001
            report[config.key] = {"ok": False, "rounds": rounds,
                                  "error": f"{type(exc).__name__}: {exc}", "root": str(root)}
        finally:
            print(f"[L2:{config.key}] {'PASS' if report[config.key]['ok'] else 'FAIL'} "
                  f"rounds={len(rounds)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[L2] report -> {out}")
    return 0 if all(item.get("ok") for item in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
