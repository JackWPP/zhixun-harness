#!/usr/bin/env python3
"""Local fake host for ``contest.harness.workspace.v2``.

The competition platform only exposes the v2 protocol through its own runtime,
so this script reproduces the host side locally: it serves a real ``start``
frame for the official A package, spawns ``/app/harness.py train`` exactly the
way the platform does, answers the JSONL RPC frames (history / model gateway /
candidate upload / state persistence), checks the finished round record, then
runs ``/app/harness.py predict`` against the uploaded artifact and validates the
line-oriented prediction contract.

Usage::

    ./.venv/bin/python scripts/emulate_workspace_v2.py --task A
    ./.venv/bin/python scripts/emulate_workspace_v2.py --task A --rounds 2 --json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = Path("/app") if Path("/app/harness.py").is_file() else REPO / "docker" / "app"
SDK = REPO / "vendor" / "contest_harness-1.0.4"
SDK_PARENT = REPO if (REPO / "contest_harness").is_dir() else SDK
PYTHON = sys.executable

sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
from upload_host import UploadStore  # noqa: E402

_UPLOAD_LIMITS = {
    "artifact_upload_protocol": "chunked-v1",
    "artifact_bytes": 512 * 1024 * 1024,
    "state_bytes": 512 * 1024 * 1024,
    "candidate_upload_bytes_per_round": 512 * 1024 * 1024,
    "state_upload_bytes_per_round": 512 * 1024 * 1024,
    "upload_chunk_bytes": 1024 * 1024,
}

TASKS = {
    "A": {
        "package": REPO / "datasets" / "cpu-a-guide-20261110-v1",
        "task": "classification",
        "metric": "macro_f1",
        "direction": "higher",
    },
    "B": {
        "package": REPO / "datasets" / "cpu-b-rul-20261110-v1",
        "task": "regression",
        "metric": "rmse",
        "direction": "lower",
    },
    "C": {
        "package": REPO / "datasets" / "cpu-c-guide-20261110-v1",
        "task": "regression",
        "metric": "rmse",
        "direction": "lower",
    },
}

PROMPT = (
    "你是工业模型自进化 harness。每一轮从官方基线/父检查点出发，只使用公开训练行拟合，"
    "公开验证行仅用于评估与选择；提交本轮候选产物并记录真实实验过程。"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declaration(name: str, files: dict[str, bytes]) -> dict:
    sizes = {key: len(value) for key, value in files.items()}
    manifest = {
        "files": [{"path": key, "size_bytes": sizes[key], "sha256": hashlib.sha256(value).hexdigest()}
                  for key, value in sorted(files.items())],
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "artifact": name,
        "artifact_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "files": manifest["files"],
    }


def write_artifact(directory: Path, files: dict[str, bytes]) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return declaration(directory.name, files)


def install_previous(previous: Path, artifact_dir: Path) -> None:
    """Mount exactly what the host would mount for the next round.

    The v2 contract gives each round a fresh container with ``/previous`` set
    to the selected parent artifact.  The fake host must therefore replace the
    previous directory with the candidate files as well; otherwise a
    ``continue_train`` candidate (different checkpoint bytes) fails the next
    round with ``PARENT_DIGEST_MISMATCH``.
    """
    previous.mkdir(parents=True, exist_ok=True)
    for stale in previous.iterdir():
        if stale.is_file() or stale.is_symlink():
            stale.unlink()
    for path in artifact_dir.iterdir():
        if path.is_file():
            shutil.copyfile(path, previous / path.name)


def materialise_sitecustomize(root: Path) -> None:
    """Point the SDK's fixed /previous and /model paths at temporary directories."""
    source = root / "sitecustomize.py"
    source.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import contest_harness.round as _round\n"
        "import contest_harness.predict as _predict\n"
        "_orig = _round.ParentView.__init__\n"
        "def _init(self, frame):\n"
        "    _orig(self, frame)\n"
        "    self.root = Path(os.environ['EMU_PREVIOUS'])\n"
        "_round.ParentView.__init__ = _init\n"
        "_predict.model_dir = lambda: Path(os.environ.get('EMU_MODEL_DIR', '/model'))\n",
        encoding="utf-8",
    )


def child_env(root: Path, work: Path, previous: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(root), str(REPO), str(SDK_PARENT), str(APP), env.get("PYTHONPATH", "")
    ])
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_WORK"] = str(work)
    env["EMU_PREVIOUS"] = str(previous)
    env["EMU_MODEL_DIR"] = str(work)
    env.pop("CONTEST_ARTIFACT_PATH", None)
    env.pop("CONTEST_PREDICTION_CONTEXT", None)
    env.pop("RSI_ALLOW_EXTENDED_DATA", None)
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def start_frame(task_key: str, package: Path, parent: dict, round_index: int, history: list,
                artifacts: list, state_path: str | None, state_sha: str) -> dict:
    digest = "sha256:" + sha256_file(package / "checkpoint.npz")
    return {
        "protocol": "contest.harness.workspace.v2",
        "run_id": "emu-run",
        "dataset_key": task_key,
        "round_index": round_index,
        "stage_id": "public-development",
        "prompt": PROMPT,
        "prompt_version": "industrial-workspace-flexible-v3",
        "rounds_per_dataset": 10,
        "parent_checkpoint_sha256": digest,
        "task_contract": {
            "dataset_key": task_key,
            "task": TASKS[task_key]["task"],
            "metric": TASKS[task_key]["metric"],
            "metric_direction": TASKS[task_key]["direction"],
            "public_projection": {
                "train_path": str(package / "public" / "train.jsonl"),
                "validation_path": str(package / "public" / "validation.jsonl"),
                "row_fields": ["id", "inputs", "answer"],
                "source_metadata": False,
            },
        },
        "model_name": "base:local-emulator",
        "parent_artifact": parent,
        "artifacts": artifacts,
        "history": [_history_record(item) for item in history],
        "state_path": state_path,
        "state_sha256": state_sha,
        "max_output_tokens": 256,
        "resource_limits": dict(_UPLOAD_LIMITS),
        "current_validation": {},
        "wall_seconds": 300,
    }


def _history_record(item: dict) -> dict:
    digest = "sha256:" + hashlib.sha256(
        json.dumps(item, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**item, "artifact_sha256": digest}


def serve(process: subprocess.Popen, frame: dict, artifact_dir: Path, state_dir: Path,
          model_reply: str, round_index: int) -> dict:
    process.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
    process.stdin.flush()
    sequence = 0
    call_ids: list[str] = []
    files: dict[str, bytes] = {}
    uploads = UploadStore(max_bytes=int(frame.get("resource_limits", {}).get("artifact_bytes", 512 * 1024 * 1024)))
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("participant closed stdout before round_complete")
        reply = json.loads(line)
        kind = reply.get("type")
        if kind == "round_failed":
            raise RuntimeError("participant failed: %s" % reply.get("error"))
        if kind == "round_complete":
            record = reply["record"]
            process.stdin.write(json.dumps({"type": "round_ack", "accepted": True}) + "\n")
            process.stdin.flush()
            process.stdin.close()
            return {"record": record, "files": files, "call_ids": call_ids}
        if reply.get("id") != sequence:
            raise RuntimeError("rpc sequence mismatch")
        sequence += 1
        upload_reply = uploads.handle(reply)
        if upload_reply is not None:
            process.stdin.write(json.dumps(upload_reply, ensure_ascii=False) + "\n")
            process.stdin.flush()
            continue
        if kind == "history_request":
            out = {"type": "history_response", "id": reply["id"], "records": frame.get("history", [])}
        elif kind == "model_request":
            call_id = "emu-call-%d" % len(call_ids)
            call_ids.append(call_id)
            message = {"role": "assistant", "content": model_reply}
            out = {
                "type": "model_response",
                "id": reply["id"],
                "call_id": call_id,
                "response": {"id": "chatcmpl-emu", "choices": [{"message": message}],
                             "usage": {"prompt_tokens": 100, "completion_tokens": 24}},
            }
        elif kind == "candidate_request":
            artifact = reply.get("artifact") or {}
            if not isinstance(artifact, dict) or not artifact.get("checkpoint_paths"):
                raise RuntimeError("candidate artifact invalid")
            upload_id = reply.get("upload_id")
            if not isinstance(upload_id, str) or not upload_id:
                raise RuntimeError("candidate request is missing upload_id")
            files = uploads.take(upload_id, "candidate")
            if len(files) > 64 or sum(len(value) for value in files.values()) > 512 * 1024 * 1024:
                raise RuntimeError("candidate too large")
            for name in list(artifact.get("checkpoint_paths", [])) + list(artifact.get("code_paths", [])):
                if name not in files:
                    raise RuntimeError("declared candidate file missing: %s" % name)
            write_artifact(artifact_dir, files)
            out = {"type": "candidate_response", "id": reply["id"],
                   "artifact_id": "emu-artifact-r%d" % round_index, "summary": {"metrics": {}}}
        elif kind == "state_request":
            upload_id = reply.get("upload_id")
            if not isinstance(upload_id, str) or not upload_id:
                raise RuntimeError("state request is missing upload_id")
            payload = uploads.take(upload_id, "state")
            state_dir.mkdir(parents=True, exist_ok=True)
            for name, data in payload.items():
                target = state_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            digest = hashlib.sha256(json.dumps(sorted(payload), default=str).encode()).hexdigest()
            out = {"type": "state_response", "id": reply["id"], "state_sha256": "sha256:" + digest}
        else:
            raise RuntimeError("unexpected frame: %s" % kind)
        process.stdin.write(json.dumps(out, ensure_ascii=False) + "\n")
        process.stdin.flush()


def run_train(root: Path, task_key: str, package: Path, round_index: int, previous: Path,
              work: Path, artifact_dir: Path, state_dir: Path, parent: dict, history: list,
              model_reply: str, timeout: float) -> dict:
    frame = start_frame(task_key, package, parent, round_index, history, [parent], None, "")
    process = subprocess.Popen(
        [PYTHON, str(APP / "harness.py"), "train"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(work), env=child_env(root, work, previous),
    )
    try:
        served = serve(process, frame, artifact_dir, state_dir, model_reply, round_index)
    except Exception:
        try:
            process.kill()
        except OSError:
            pass
        stderr = process.stderr.read() if process.stderr else ""
        raise RuntimeError("train failed\n--- child stderr ---\n%s" % stderr)
    stderr = process.stderr.read() if process.stderr else ""
    if stderr.strip():
        print("[emulator] child stderr:\n%s" % stderr.strip(), file=sys.stderr)
    return served


def validate_artifact(served: dict) -> None:
    record = served["record"]
    for key in ("selected_artifact_id", "hypothesis", "change_summary", "reason",
                "configuration", "history_ids", "model_call_ids", "seed"):
        if key not in record:
            raise RuntimeError("round record missing %s" % key)
    if not record["model_call_ids"]:
        raise RuntimeError("round did not cite a model call")
    for call_id in record["model_call_ids"]:
        if call_id not in served["call_ids"]:
            raise RuntimeError("unknown model_call_id %s" % call_id)


def run_predict(root: Path, task_key: str, artifact_dir: Path, rows_path: Path,
                work: Path, timeout: float) -> list[dict]:
    checkpoint = artifact_dir / "checkpoint.npz"
    artifact = declaration(artifact_dir.name, {path.name: path.read_bytes() for path in artifact_dir.iterdir() if path.is_file()})
    context = json.dumps({
        "protocol": "contest.predict-context.v1",
        "dataset_key": task_key,
        "task": TASKS[task_key]["task"],
        "artifact": {
            "checkpoint_paths": ["checkpoint.npz"],
            "code_paths": ["submission_meta.json"],
            "predict_argv": ["python3", "/app/harness.py", "predict"],
        },
        "artifact_id": artifact["artifact"],
    }, sort_keys=True)
    env = child_env(root, work, work)
    env["CONTEST_ARTIFACT_PATH"] = str(checkpoint)
    env["CONTEST_ARTIFACT_MANIFEST"] = str(artifact_dir / "manifest.json")
    env["CONTEST_PREDICTION_CONTEXT"] = context
    env["CONTEST_DATASET_KEY"] = task_key
    env["CONTEST_TASK"] = TASKS[task_key]["task"]
    env["EMU_MODEL_DIR"] = str(artifact_dir)

    requests = []
    with rows_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 3:
                break
            row = json.loads(line)
            requests.append(json.dumps({"task": TASKS[task_key]["task"], "inputs": row["inputs"]}))
    process = subprocess.run(
        [PYTHON, str(APP / "harness.py"), "predict"],
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
        if TASKS[task_key]["task"] == "classification":
            if not isinstance(body, list) or not body:
                raise RuntimeError("classification output invalid")
            labels = {item["label"] for item in body}
            if len(labels) != len(body):
                raise RuntimeError("duplicate labels in classification output")
            scores = [item["score"] for item in body]
            if any(score < 0 or score > 1 for score in scores):
                raise RuntimeError("score out of [0,1]")
            if sum(score == max(scores) for score in scores) != 1:
                raise RuntimeError("classification output needs exactly one top score")
        else:
            if not isinstance(body, dict) or set(body) != {"value"}:
                raise RuntimeError("regression output invalid")
            if body["value"] < 0 or body["value"] != body["value"]:
                raise RuntimeError("regression value invalid")
    return outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="local contest.harness.workspace.v2 host")
    parser.add_argument("--task", default="A", choices=sorted(TASKS))
    parser.add_argument("--rounds", type=int, default=1,
                        help="how many sequential rounds to emulate (parent lineage is enforced)")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    package = TASKS[args.task]["package"]
    if not (package / "checkpoint.npz").is_file():
        raise SystemExit("missing package: %s" % package)
    root = Path(tempfile.mkdtemp(prefix="emu-v2-"))
    work = root / "workspace"
    work.mkdir()
    previous = root / "previous"
    previous.mkdir()
    materialise_sitecustomize(root)

    official = {"checkpoint.npz": (package / "checkpoint.npz").read_bytes()}
    (previous / "checkpoint.npz").write_bytes(official["checkpoint.npz"])
    parent = {
        "artifact_id": "official-baseline",
        "artifact_sha256": "sha256:" + sha256_file(package / "checkpoint.npz"),
        "artifact": {
            "checkpoint_paths": ["checkpoint.npz"],
            "code_paths": [],
            "predict_argv": ["python3", "/app/harness.py", "predict"],
        },
        "files": [{"path": "checkpoint.npz",
                   "sha256": "sha256:" + sha256_file(package / "checkpoint.npz"),
                   "size_bytes": (package / "checkpoint.npz").stat().st_size}],
    }
    history: list[dict] = []
    state_path = ""
    results = []
    for round_index in range(1, max(1, args.rounds) + 1):
        artifact_dir = root / ("artifact-%02d" % round_index)
        state_dir = root / ("state-%02d" % round_index)
        reply_choice = 1 if TASKS[args.task]["task"] == "classification" else 1
        reply = json.dumps({
            "choice": reply_choice,
            "hypothesis": "公开验证集上的漂移校准候选应优于父模型",
            "reason": "emulator: 选择改进最大的候选（确定性回退也会得到同一结果）",
        }, ensure_ascii=False)
        served = run_train(root, args.task, package, round_index, previous, work,
                           artifact_dir, state_dir, parent, history, reply, args.timeout)
        validate_artifact(served)
        install_previous(previous, artifact_dir)
        checkpoint = artifact_dir / "checkpoint.npz"
        if not checkpoint.is_file():
            raise RuntimeError("candidate checkpoint missing")
        predictions = run_predict(root, args.task, artifact_dir,
                                  package / "public" / "validation.jsonl", work, args.timeout)
        record = served["record"]
        history.append({
            "id": "round-%d" % round_index,
            "round_index": round_index,
            "selected_artifact_id": record["selected_artifact_id"],
            "hypothesis": record["hypothesis"],
            "change_summary": record["change_summary"],
            "reason": record["reason"],
        })
        state_path = str(state_dir)
        parent = {
            "artifact_id": record["selected_artifact_id"],
            "artifact_sha256": "sha256:" + sha256_file(checkpoint),
            "artifact": {
                "checkpoint_paths": ["checkpoint.npz"],
                "code_paths": ["submission_meta.json"],
                "predict_argv": ["python3", "/app/harness.py", "predict"],
            },
            "files": [
                {"path": "checkpoint.npz", "sha256": "sha256:" + sha256_file(checkpoint),
                 "size_bytes": checkpoint.stat().st_size},
                {"path": "submission_meta.json",
                 "sha256": "sha256:" + sha256_file(artifact_dir / "submission_meta.json"),
                 "size_bytes": (artifact_dir / "submission_meta.json").stat().st_size},
            ],
        }
        results.append({
            "round_index": round_index,
            "selected_artifact_id": record["selected_artifact_id"],
            "model_call_ids": record["model_call_ids"],
            "predictions": len(predictions),
            "state_sha256": "",
        })
    report = {"ok": True, "protocol": "contest.harness.workspace.v2", "task": args.task,
              "rounds": results, "root": str(root)}
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else
          "workspace v2 emulation PASS  task=%s rounds=%d" % (args.task, len(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
