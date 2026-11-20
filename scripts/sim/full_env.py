#!/usr/bin/env python3
"""Full RSI simulation environment (workspace v2 + real gateway + SwanLab).

This is the local "contest-like" host used to rehearse the agent-driven RSI loop
end to end:

* the participant is the real ``/app/harness.py train|predict`` entry point
  (the same module that ships in the Docker image),
* rounds follow ``contest.harness.workspace.v2`` (start frame, history, model
  gateway, candidate upload, state persistence, finish record),
* the model gateway talks to a real OpenAI-compatible provider (default: the
  PyTRiO ``Qwen/Qwen3.8-27B`` endpoint from ``.scratch/pytrio.env``); call ids
  are the provider response ids, so every round cites genuine model calls,
* data arrives as cumulative drops (official packages grow across rounds; the
  external UCI-gas package adds one unseen batch per round),
* every round logs validation + hidden-proxy metrics and can stream them to
  SwanLab (online when ``SWANLAB_API_KEY`` is set, otherwise offline).

Usage::

    .venv/bin/python scripts/sim/full_env.py --scenario official --rounds 3
    .venv/bin/python scripts/sim/full_env.py --scenario external
    .venv/bin/python scripts/sim/full_env.py --scenario schema --controller kernel
    .venv/bin/python scripts/sim/full_env.py --scenario official --provider fake
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import numpy as np
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "sim"))

import datasets as _datasets  # noqa: E402
from upload_host import UploadStore  # noqa: E402

from rsi.metrics import from_predictions  # noqa: E402

PYTHON = sys.executable
APP = Path("/app") if Path("/app/harness.py").is_file() else REPO / "docker" / "app"
SDK = REPO / "vendor" / "contest_harness-1.0.4"
SDK_PARENT = REPO if (REPO / "contest_harness").is_dir() else SDK

_UPLOAD_LIMITS = {
    "artifact_upload_protocol": "chunked-v1",
    "artifact_bytes": 512 * 1024 * 1024,
    "state_bytes": 512 * 1024 * 1024,
    "candidate_upload_bytes_per_round": 512 * 1024 * 1024,
    "state_upload_bytes_per_round": 512 * 1024 * 1024,
    "upload_chunk_bytes": 1024 * 1024,
}


# ---------------------------------------------------------------- provider --
class ProviderError(RuntimeError):
    pass


def _load_env_file(path: Path) -> dict:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().removeprefix("export ").strip()] = value.strip().strip('"').strip("'")
    return values


class OpenAIProvider:
    """Minimal OpenAI-compatible chat-completions client with retries."""

    def __init__(self, base_url: str, api_key: str, model: str, thinking: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.thinking = bool(thinking)
        # Providers disagree on how to turn reasoning off.  tokenplus (the
        # SiliconFlow-compatible endpoint) rejects ``enable_thinking`` and wants
        # ``thinking={"type": "disabled"}``; PyTRiO accepts ``enable_thinking``.
        lowered = self.base_url.lower()
        self.thinking_style = "thinking" if ("tokenplus" in lowered or "siliconflow" in lowered) else "enable_thinking"
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def _thinking_fields(self):
        enabled = "enabled" if self.thinking else "disabled"
        if self.thinking_style == "thinking":
            return {"thinking": {"type": enabled}}
        return {"enable_thinking": bool(self.thinking)}

    def chat(self, messages, *, max_tokens=512, temperature=0.0, tools=None, tool_choice="auto"):
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens}
        payload.update(self._thinking_fields())
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        error = None
        for attempt in range(3):
            request = urllib.request.Request(
                self.base_url + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + self.api_key})
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read().decode("utf-8"))
                usage = data.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                self.calls += 1
                if not data.get("id"):
                    data["id"] = "sim-" + uuid.uuid4().hex
                message = ((data.get("choices") or [{}])[0].get("message") or {})
                if (not (message.get("content") or "").strip() and self.thinking
                        and attempt < 2 and int(payload.get("max_tokens") or 0) < 4096):
                    # Reasoning consumed the whole budget; retry once with room.
                    payload["max_tokens"] = min(4096, max(1024, int(payload["max_tokens"]) * 2))
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    continue
                return data
            except urllib.error.HTTPError as exc:  # noqa: PERF203 - include provider body
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:600]
                except Exception:
                    detail = ""
                error = f"HTTP {exc.code}: {detail}"
                if exc.code == 400 and "thinking" in detail and "enable_thinking" in detail:
                    self.thinking_style = "thinking" if self.thinking_style == "enable_thinking" else "enable_thinking"
                    payload.pop("enable_thinking", None)
                    payload.pop("thinking", None)
                    payload.update(self._thinking_fields())
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    continue
                if exc.code == 400 and "enable_thinking" not in detail and self.thinking_style == "thinking":
                    payload.pop("thinking", None)
                    payload["enable_thinking"] = False
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    continue
                time.sleep(0.8 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001 - network/provider errors are retried
                error = exc
                time.sleep(0.8 * (attempt + 1))
        raise ProviderError(f"gateway call failed: {error}")


class FakeProvider:
    """Deterministic scripted actions for offline smoke tests (kernel protocol)."""

    def __init__(self, model: str = "fake"):
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self._script = [
            {"tool": "get_run_state", "args": {}},
            {"tool": "propose_experiment", "args": {
                "strategy": "retain", "params_json": "{}",
                "postprocess_json": "{\"type\": \"recenter\", \"mode\": \"batch_mean\", \"weight\": 0.4}",
                "hypothesis": "recenter under drift"}},
            {"tool": "run_experiment", "args": {"plan_id": "plan-01"}},
            {"tool": "decide", "args": {"decision": "retain", "reason": "margin not met"}},
            {"tool": "finish_round", "args": {"summary": "offline smoke", "lesson": "keep margin"}},
        ]

    def chat(self, messages, *, max_tokens=512, temperature=0.0, tools=None, tool_choice="auto"):
        action = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        self.prompt_tokens += 120
        self.completion_tokens += 24
        return {
            "id": "fake-call-" + uuid.uuid4().hex[:8],
            "choices": [{"message": {"role": "assistant",
                                     "content": json.dumps(action, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 24},
        }


def resolve_provider(args):
    if args.provider == "fake":
        return FakeProvider()
    base_url = args.base_url or os.environ.get("SIM_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL")
    api_key = args.api_key or os.environ.get("SIM_LLM_API_KEY") or os.environ.get("LLM_API_KEY")
    model = args.model or os.environ.get("SIM_LLM_MODEL") or os.environ.get("LLM_MODEL")
    if not (base_url and api_key and model):
        for candidate in (REPO / ".scratch" / "llm.env", REPO / ".scratch" / "pytrio.env"):
            values = _load_env_file(candidate)
            base_url = base_url or values.get("LLM_BASE_URL")
            api_key = api_key or values.get("LLM_API_KEY")
            model = model or values.get("LLM_MODEL")
            if base_url and api_key and model:
                break
    if not (base_url and api_key and model):
        raise SystemExit("no provider configured: set SIM_LLM_BASE_URL/SIM_LLM_API_KEY/SIM_LLM_MODEL "
                         "or use --provider fake")
    thinking_mode = str(getattr(args, "thinking", "auto") or "auto").lower()
    # E6 A/B: thinking on gave identical decisions/scores but ~9x wall time,
    # so "auto" now means off; --thinking on remains for debugging.
    thinking = thinking_mode == "on"
    return OpenAIProvider(base_url, api_key, model, thinking=thinking)


# -------------------------------------------------------------------- host --
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialise_sitecustomize(root: Path) -> Path:
    """Point the SDK's fixed /previous and /model paths at per-round directories."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sitecustomize.py").write_text(
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
        encoding="utf-8")
    return root


def parent_payload(artifact_id: str, checkpoint: Path, code_paths: list[str] | None = None) -> dict:
    digest = "sha256:" + sha256_file(checkpoint)
    return {
        "artifact_id": artifact_id,
        "artifact_sha256": digest,
        "artifact": {
            "checkpoint_paths": ["checkpoint.npz"],
            "code_paths": list(code_paths or []),
            "predict_argv": ["python3", "/app/harness.py", "predict"],
        },
        "files": [{"path": "checkpoint.npz", "sha256": digest,
                   "size_bytes": checkpoint.stat().st_size}],
    }


def write_files(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def serve(process, frame, artifact_dir, state_dir, provider, round_index, model_options):
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
            messages = reply.get("messages") or []
            options = {key: reply[key] for key in
                       ("tools", "tool_choice", "max_tokens", "temperature", "top_p", "seed")
                       if key in reply}
            max_tokens = min(int(options.pop("max_tokens", 512) or 512), int(model_options["max_tokens"]))
            temperature = float(options.pop("temperature", 0.0) or 0.0)
            try:
                response = provider.chat(messages, max_tokens=max_tokens, temperature=temperature,
                                         tools=options.get("tools"),
                                         tool_choice=options.get("tool_choice", "auto"))
            except Exception:
                if os.environ.get("SIM_DEBUG"):
                    (state_dir / "model_request_failed.json").write_text(
                        json.dumps({"messages": messages, "options": options}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
                raise
            call_id = str(response.get("id"))
            call_ids.append(call_id)
            out = {"type": "model_response", "id": reply["id"], "call_id": call_id,
                   "response": response}
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
            write_files(artifact_dir, files)
            out = {"type": "candidate_response", "id": reply["id"],
                   "artifact_id": "sim-artifact-%s-r%d" % (frame.get("dataset_key"), round_index),
                   "summary": {"metrics": {}}}
        elif kind == "state_request":
            upload_id = reply.get("upload_id")
            if not isinstance(upload_id, str) or not upload_id:
                raise RuntimeError("state request is missing upload_id")
            payload = uploads.take(upload_id, "state")
            write_files(state_dir, payload)
            digest = hashlib.sha256(json.dumps(sorted(payload), default=str).encode()).hexdigest()
            out = {"type": "state_response", "id": reply["id"], "state_sha256": "sha256:" + digest}
        else:
            raise RuntimeError("unexpected frame: %s" % kind)
        process.stdin.write(json.dumps(out, ensure_ascii=False) + "\n")
        process.stdin.flush()


def build_frame(spec: dict, round_index: int, rounds: int, data_dir: Path, parent: dict,
                history: list, state_path: str | None, state_sha: str, parent_sha: str) -> dict:
    return {
        "protocol": "contest.harness.workspace.v2",
        "run_id": spec.get("run_id", "sim-run"),
        "dataset_key": spec["key"],
        "round_index": round_index,
        "stage_id": "public-development",
        "prompt": "你是工业模型自进化 harness。每一轮只使用当前阶段公开训练行拟合，"
                  "公开验证行仅用于评估与选择；提交本轮候选产物并记录真实实验过程。",
        "prompt_version": "industrial-workspace-flexible-v3",
        "rounds_per_dataset": rounds,
        "parent_checkpoint_sha256": parent_sha,
        "task_contract": {
            "dataset_key": spec["key"],
            "task": spec["task"],
            "metric": spec["metric"],
            "metric_direction": spec["direction"],
            "public_projection": {
                "train_path": str(data_dir / "public" / "train.jsonl"),
                "validation_path": str(data_dir / "public" / "validation.jsonl"),
                "row_fields": ["id", "inputs", "answer"],
                "source_metadata": False,
            },
        },
        "model_name": spec.get("model_name", "base:sim-host"),
        "parent_artifact": parent,
        "artifacts": [parent],
        "history": history,
        "state_path": state_path,
        "state_sha256": state_sha,
        "max_output_tokens": int(os.environ.get("RSI_V2_MAX_OUTPUT", "256") or 256),
        "resource_limits": dict(_UPLOAD_LIMITS),
        "current_validation": {},
        "wall_seconds": 900,
    }


def run_train(work: Path, previous: Path, frame: dict, artifact_dir: Path, state_dir: Path,
              provider, round_index: int, controller: str, timeout: float,
              site_root: Path | None = None) -> dict:
    env = os.environ.copy()
    paths = [str(REPO), str(SDK_PARENT), str(APP)]
    if site_root is not None:
        paths.insert(0, str(site_root))
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_WORK"] = str(work)
    env["EMU_PREVIOUS"] = str(previous)
    env["EMU_MODEL_DIR"] = str(work)
    env["RSI_V2_CONTROLLER"] = controller
    env.pop("CONTEST_ARTIFACT_PATH", None)
    env.pop("CONTEST_PREDICTION_CONTEXT", None)
    env.setdefault("OMP_NUM_THREADS", "1")
    process = subprocess.Popen([PYTHON, str(APP / "harness.py"), "train"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, cwd=str(work), env=env)
    try:
        served = serve(process, frame, artifact_dir, state_dir, provider, round_index,
                       {"max_tokens": int(os.environ.get("RSI_V2_MAX_OUTPUT", "768") or 768)})
    except Exception:
        try:
            process.kill()
        except OSError:
            pass
        stderr = process.stderr.read() if process.stderr else ""
        raise RuntimeError("train failed\n--- child stderr ---\n%s" % stderr)
    stderr = process.stderr.read() if process.stderr else ""
    if stderr.strip():
        print("[sim] child stderr:\n%s" % stderr.strip()[-4000:], file=sys.stderr)
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
    return served


def predict_rows(artifact_dir: Path, rows: list[dict], task: str, classes: list[str],
                 dataset_key: str, timeout: float, site_root: Path | None = None) -> list:
    model_dir = artifact_dir
    env = os.environ.copy()
    paths = [str(REPO), str(SDK_PARENT), str(APP)]
    if site_root is not None:
        paths.insert(0, str(site_root))
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    env["CONTEST_ARTIFACT_PATH"] = str(artifact_dir / "checkpoint.npz")
    env["CONTEST_PREDICTION_CONTEXT"] = json.dumps({
        "protocol": "contest.predict-context.v1",
        "dataset_key": dataset_key,
        "task": task,
        "artifact": {"checkpoint_paths": ["checkpoint.npz"],
                     "code_paths": ["submission_meta.json"],
                     "predict_argv": ["python3", "/app/harness.py", "predict"]},
        "artifact_id": artifact_dir.name,
    }, sort_keys=True)
    env["CONTEST_DATASET_KEY"] = dataset_key
    env["CONTEST_TASK"] = task
    env["EMU_MODEL_DIR"] = str(model_dir)
    env.setdefault("OMP_NUM_THREADS", "1")
    process = subprocess.Popen([PYTHON, str(APP / "harness.py"), "predict"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, cwd=str(artifact_dir), env=env)
    payload = "".join(json.dumps({"task": task, "inputs": row["inputs"]}, ensure_ascii=False) + "\n"
                      for row in rows)
    try:
        stdout, stderr = process.communicate(input=payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    if process.returncode not in (0, None):
        raise RuntimeError("predict failed rc=%s\n%s" % (process.returncode, stderr[-2000:]))
    predicted = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        payload_item = json.loads(line)
        outputs = payload_item.get("outputs")
        if task == "classification":
            predicted.append(max(outputs, key=lambda item: item["score"])["label"])
        else:
            predicted.append(float(outputs["value"]))
    if len(predicted) != len(rows):
        raise RuntimeError("predict produced %d outputs for %d rows\n%s"
                           % (len(predicted), len(rows), stderr[-1000:]))
    return predicted


# ------------------------------------------------------------------ specs --
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def official_specs(tasks: list[str], rounds: int, cap_train: int) -> list[dict]:
    specs = []
    for key in tasks:
        package = REPO / "datasets" / {
            "A": "cpu-a-guide-20261110-v1",
            "B": "cpu-b-rul-20261110-v1",
            "C": "cpu-c-guide-20261110-v1",
        }[key]
        train = read_jsonl(package / "public" / "train.jsonl")
        validation = read_jsonl(package / "public" / "validation.jsonl")
        validation_rows = validation[:600]
        hidden = validation[600:600 + (1500 if key == "C" else 800)]
        specs.append({
            "key": key, "package": package, "run_id": "sim-" + key, "task": "classification" if key == "A" else "regression",
            "metric": "macro_f1" if key == "A" else "rmse",
            "direction": "higher" if key == "A" else "lower",
            "classes": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"] if key == "A" else [],
            "rounds": rounds, "train_rows": train, "validation_rows": validation_rows,
            "hidden_rows": hidden, "cap_train": cap_train, "drops": None,
        })
    return specs


def official_seq_spec(rounds: int = 10, seed: int = 0) -> dict:
    """Faithful official-A protocol: b1 + revealed drops -> predict next drop.

    Round r trains on UCI batch1 plus drops[0..r-2] (labels revealed by the
    previous round) and predicts drops[r-1].  Validation stays the public
    projection (batch2) so selection never sees the prediction target.
    """
    built = _datasets.ensure_packages()["a_stream"]
    root = Path(built["root"])
    rng = np.random.default_rng(0xA5EED + int(seed))
    train = read_jsonl(root / "public" / "train.jsonl")
    validation = read_jsonl(root / "public" / "validation.jsonl")
    drops = [read_jsonl(path) for path in sorted((root / "stream").glob("b*.jsonl"))]
    train = [train[index] for index in rng.permutation(len(train))]
    validation = [validation[index] for index in rng.permutation(len(validation))]
    rounds = max(1, min(int(rounds), len(drops)))
    return {
        "run_id": "sim-A-SEQ", "key": "A", "package": Path("datasets") / "cpu-a-guide-20261110-v1",
        "task": "classification", "metric": "macro_f1", "direction": "higher",
        "classes": list(built["classes"]), "rounds": rounds,
        "train_rows": train, "validation_rows": validation, "hidden_rows": [],
        "cap_train": 0, "drops": drops, "sequence": True, "seed": int(seed),
    }


def external_spec(rounds: int = 5) -> dict:
    built = _datasets.ensure_packages()["gas"]
    root = Path(built["root"])
    initial = read_jsonl(root / "public" / "train.jsonl")
    drops = [read_jsonl(sorted((root / "drops").glob("*.jsonl"))[i]) for i in range(len(list((root / "drops").glob("*.jsonl"))))]
    drops = [read_jsonl(path) for path in sorted((root / "drops").glob("*.jsonl"))]
    hidden = []
    for path in sorted((root / "hidden").glob("*.jsonl")):
        hidden.extend(read_jsonl(path))
    return {
        "run_id": "sim-GAS", "key": "A", "package": root, "task": "classification", "metric": "macro_f1",
        "direction": "higher", "classes": list(built["classes"]), "rounds": min(rounds, len(drops) + 1),
        "train_rows": initial, "validation_rows": drops[0] if drops else [],
        "hidden_rows": hidden, "cap_train": 0, "drops": drops,
    }


def schema_spec(rounds: int = 2) -> dict:
    built = _datasets.ensure_packages()["schema"]
    root = Path(built["root"])
    return {
        "run_id": "sim-SCHEMA", "key": "A", "package": root, "task": "classification", "metric": "macro_f1",
        "direction": "higher", "classes": list(built["classes"]), "rounds": max(1, int(rounds)),
        "train_rows": read_jsonl(root / "public" / "train.jsonl"),
        "validation_rows": read_jsonl(root / "public" / "validation.jsonl"),
        "hidden_rows": read_jsonl(root / "hidden" / "private_schema.jsonl"),
        "cap_train": 0, "drops": None,
    }


def stage_data(spec: dict, round_index: int, rounds: int, data_dir: Path) -> None:
    train = list(spec["train_rows"])
    validation = list(spec["validation_rows"])
    if spec.get("sequence"):
        # SDK 1.0.4: the public validation split is for measurement/selection
        # only and must not be merged into gradient training.  Training grows
        # with the drops revealed by previous rounds, nothing else.
        train = list(spec["train_rows"])
        for index in range(min(round_index - 1, len(spec["drops"]))):
            train.extend(spec["drops"][index])
        validation = list(spec["validation_rows"])
    elif spec["drops"]:
        train = list(spec["train_rows"])
        for index in range(min(round_index - 1, len(spec["drops"]))):
            train.extend(spec["drops"][index])
        validation = spec["drops"][min(round_index - 1, len(spec["drops"]) - 1)]
    else:
        fraction = round_index / float(rounds)
        cut = max(1, int(len(train) * fraction))
        if spec.get("cap_train"):
            cut = min(cut, spec["cap_train"])
        train = train[:cut]
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        target = data_dir / "public" / (name + ".jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def score(spec: dict, rows: list[dict], predicted: list) -> float:
    expected = [row["answer"] for row in rows]
    if spec["metric"] == "macro_f1":
        labels = list(spec["classes"])
        predicted_idx = [labels.index(str(item)) if str(item) in labels else -1 for item in predicted]
        expected_idx = [labels.index(str(item)) if str(item) in labels else -1 for item in expected]
        f1 = []
        for index in range(len(labels)):
            tp = sum(1 for a, b in zip(predicted_idx, expected_idx) if a == index and b == index)
            fp = sum(1 for a, b in zip(predicted_idx, expected_idx) if a == index and b != index)
            fn = sum(1 for a, b in zip(predicted_idx, expected_idx) if a != index and b == index)
            f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return float(sum(f1) / len(f1))
    predicted_arr = np.asarray(predicted, dtype=float)
    expected_arr = np.asarray(expected, dtype=float)
    return float(from_predictions("regression", predicted_arr, expected_arr).get("rmse") or 0.0)


# ------------------------------------------------------------------- main --
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="full RSI simulation host (workspace v2)")
    parser.add_argument("--scenario", choices=["official", "official-seq", "external", "schema"], default="official")
    parser.add_argument("--tasks", default="A,B,C")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cap-train", type=int, default=24000)
    parser.add_argument("--controller", choices=["kernel", "pipeline", "auto"], default="kernel")
    parser.add_argument("--provider", choices=["real", "fake"], default="real")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default=str(REPO / ".scratch" / "sim" / "runs"))
    parser.add_argument("--swanlab", choices=["auto", "offline", "off"], default="auto")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--policy", choices=["on", "off"], default="on",
                        help="gen2 drift policy switch passed to the participant")
    parser.add_argument("--pair-delta", type=float, default=None,
                        help="override RSI_V2_PAIR_DELTA for the historical pair arm")
    parser.add_argument("--hidden-every-round", action="store_true",
                        help="score the hidden proxy on every round (learning curve)")
    parser.add_argument("--tag", default="", help="experiment arm tag used in run dir/SwanLab name")
    parser.add_argument("--seed", type=int, default=0, help="world seed (row order + participant init)")
    parser.add_argument("--adapt-arm", choices=["auto", "pair", "retrain", "retrain_pair", "off"],
                        default="auto", help="restrict cumulative-adaptation candidate arms")
    parser.add_argument("--capacity-arm", choices=["off", "h64", "ls10", "both"],
                        default="off", help="opt-in capacity arm (official v1 checkpoint)")
    parser.add_argument("--gate", choices=["on", "off"], default="on",
                        help="paired block-bootstrap selection gate (default on)")
    parser.add_argument("--kernel-tools", choices=["full", "core"], default="full",
                        help="kernel tool surface: full 19 tools or a compact core set")
    parser.add_argument("--native-tools", choices=["on", "off"], default="on",
                        help="native OpenAI tool calling for the host gateway")
    parser.add_argument("--kernel-context", choices=["full", "minimal"], default="full",
                        help="kernel brief context: full skills/knowledge or minimal")
    parser.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                        help="provider reasoning mode: on for kernel/auto, off for pipeline")
    parser.add_argument("--max-turns", type=int, default=6, help="kernel tool turns per round")
    parser.add_argument("--token-cap", type=int, default=0, help="kernel token budget (0 = unlimited)")
    parser.add_argument("--max-output", type=int, default=256, help="max output tokens per gateway call")
    args = parser.parse_args(argv)

    provider = resolve_provider(args)
    os.environ["RSI_V2_DRIFT_POLICY"] = "1" if args.policy == "on" else "0"
    os.environ["RSI_V2_GATE"] = args.gate
    os.environ["RSI_V2_SEED"] = str(int(args.seed))
    os.environ["RSI_V2_ADAPT_ARM"] = args.adapt_arm
    os.environ["RSI_V2_CAPACITY_ARM"] = args.capacity_arm
    os.environ["RSI_V2_MAX_TURNS"] = str(int(args.max_turns))
    os.environ["RSI_V2_TOKEN_CAP"] = str(int(args.token_cap))
    os.environ["RSI_V2_MAX_OUTPUT"] = str(int(args.max_output))
    os.environ["RSI_V2_THINKING"] = args.thinking
    os.environ["RSI_V2_KERNEL_TOOLS"] = args.kernel_tools
    os.environ["RSI_V2_NATIVE_TOOLS"] = "1" if args.native_tools == "on" else "0"
    os.environ["RSI_V2_KERNEL_CONTEXT"] = args.kernel_context
    if args.pair_delta is not None:
        os.environ["RSI_V2_PAIR_DELTA"] = str(float(args.pair_delta))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = re.sub(r"[^0-9A-Za-z_.-]+", "-", args.tag).strip("-")
    run_name = f"rsi-sim-{args.scenario}-{tag + '-' if tag else ''}{stamp}"
    run_root = Path(args.out) / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    site_root = materialise_sitecustomize(run_root / "site")
    print(f"[sim] run_root={run_root} provider={type(provider).__name__} controller={args.controller}")

    if args.scenario == "official":
        specs = official_specs([item for item in args.tasks.split(",") if item], args.rounds, args.cap_train)
    elif args.scenario == "official-seq":
        specs = [official_seq_spec(args.rounds, args.seed)]
    elif args.scenario == "external":
        specs = [external_spec(args.rounds)]
    else:
        specs = [schema_spec(args.rounds)]

    swan_run = None
    if args.swanlab != "off":
        try:
            import swanlab

            mode = "online" if os.environ.get("SWANLAB_API_KEY") else "offline"
            if args.swanlab == "offline":
                mode = "offline"
            swan_name = f"{args.scenario}-{tag}-{stamp}" if tag else f"{args.scenario}-{stamp}"
            swan_run = swanlab.init(project=os.environ.get("SWANLAB_PROJECT", "zhixun-rsi"),
                                    mode=mode, name=swan_name,
                                    log_dir=str(run_root / "swanlab"))
            print(f"[sim] swanlab mode={mode} log_dir={run_root / 'swanlab'}")
        except Exception as exc:  # noqa: BLE001 - optional dependency
            print(f"[sim] swanlab disabled: {exc}", file=sys.stderr)
            swan_run = None

    summary = {"run_root": str(run_root), "scenario": args.scenario, "controller": args.controller,
               "provider": type(provider).__name__, "rounds": args.rounds,
               "policy": args.policy, "pair_delta": args.pair_delta, "tag": tag,
               "seed": int(args.seed),
               "hidden_every_round": bool(args.hidden_every_round), "tasks": {}}
    for spec in specs:
        task_root = run_root / "tasks" / spec["key"]
        task_root.mkdir(parents=True, exist_ok=True)
        previous = task_root / "parent"
        parent_files = {"checkpoint.npz": (spec["package"] / "checkpoint.npz").read_bytes()}
        write_files(previous, parent_files)
        parent = parent_payload("official-baseline", previous / "checkpoint.npz")
        history: list[dict] = []
        state_dir = task_root / "state"
        rounds = spec["rounds"]
        task_summary = {"key": spec["key"], "task": spec["task"], "metric": spec["metric"],
                        "direction": spec["direction"], "rounds": [], "hidden_proxy": None}
        for round_index in range(1, rounds + 1):
            data_dir = task_root / "data" / f"r{round_index}"
            stage_data(spec, round_index, rounds, data_dir)
            work = task_root / "work" / f"r{round_index}"
            work.mkdir(parents=True, exist_ok=True)
            artifact_dir = task_root / "artifacts" / f"r{round_index}"
            round_state = task_root / "state" / f"r{round_index}"
            round_state.mkdir(parents=True, exist_ok=True)
            state_path = str(state_dir) if state_dir.is_dir() and any(state_dir.iterdir()) else None
            state_sha = ""
            frame = build_frame(spec, round_index, rounds, data_dir, parent, history,
                                state_path, state_sha, parent["artifact_sha256"])
            started = time.time()
            served = run_train(work, previous, frame, artifact_dir, round_state, provider,
                               round_index, args.controller, args.timeout, site_root)
            wall = time.time() - started
            record = served["record"]
            chosen = ""
            try:
                chosen = (record.get("configuration") or {}).get("chosen", "")
            except AttributeError:
                chosen = ""
            predicted = predict_rows(artifact_dir, spec["validation_rows"], spec["task"],
                                     spec["classes"], spec["key"], args.timeout, site_root)
            val_metric = score(spec, spec["validation_rows"], predicted)
            hidden_metric = None
            target_rows = None
            if spec.get("sequence") and round_index <= len(spec["drops"]):
                target_rows = spec["drops"][round_index - 1]
            elif spec["hidden_rows"] and (args.hidden_every_round or round_index == rounds):
                target_rows = spec["hidden_rows"]
            if target_rows:
                hidden_pred = predict_rows(artifact_dir, target_rows, spec["task"],
                                           spec["classes"], spec["key"], args.timeout, site_root)
                hidden_metric = score(spec, target_rows, hidden_pred)
                if spec.get("sequence"):
                    task_summary["hidden_proxy_per_round"] = task_summary.get("hidden_proxy_per_round") or []
                    task_summary["hidden_proxy_per_round"].append(round(0.0 if hidden_metric is None else float(hidden_metric), 4))
                if round_index == rounds:
                    task_summary["hidden_proxy"] = hidden_metric
            entry = {"round": round_index, "wall_s": round(wall, 1), "calls": len(served["call_ids"]),
                     "decision": record.get("hypothesis", "")[:200], "chosen": chosen,
                     "validation": val_metric, "hidden_proxy": hidden_metric,
                     "parent_sha": parent["artifact_sha256"][:23],
                     "candidate_sha": (record.get("configuration") or {}).get("candidate_sha256", "")[:23]}
            task_summary["rounds"].append(entry)
            print(f"[sim] {spec['key']} r{round_index}/{rounds} calls={entry['calls']} "
                  f"val={val_metric:.4f} hidden={hidden_metric} wall={wall:.1f}s")
            if swan_run is not None:
                try:
                    key = spec["key"]
                    swan_run.log({f"{key}/validation_primary": val_metric,
                                  f"{key}/hidden_proxy_primary": hidden_metric if hidden_metric is not None else float("nan"),
                                  f"{key}/model_calls": len(served["call_ids"]),
                                  f"{key}/prompt_tokens": getattr(provider, "prompt_tokens", 0),
                                  f"{key}/completion_tokens": getattr(provider, "completion_tokens", 0),
                                  f"{key}/wall_s": round(wall, 1)}, step=round_index)
                except Exception:
                    pass
            history.append({"id": f"sim-{spec['key']}-r{round_index}", "round_index": round_index,
                            "artifact_id": f"sim-artifact-{spec['key']}-r{round_index}",
                            "checkpoint_sha256": "sha256:" + sha256_file(artifact_dir / "checkpoint.npz")})
            previous = task_root / "previous" / f"r{round_index}"
            write_files(previous, {"checkpoint.npz": (artifact_dir / "checkpoint.npz").read_bytes()})
            parent = parent_payload(f"sim-artifact-{spec['key']}-r{round_index}",
                                    previous / "checkpoint.npz", ["submission_meta.json"])
            state_dir = round_state
        summary["tasks"][spec["key"]] = task_summary
        (task_root / "task_summary.json").write_text(json.dumps(task_summary, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    summary["provider_usage"] = {"calls": getattr(provider, "calls", 0),
                                 "prompt_tokens": getattr(provider, "prompt_tokens", 0),
                                 "completion_tokens": getattr(provider, "completion_tokens", 0)}
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[sim] summary written:", run_root / "summary.json")
    print(json.dumps({key: {"hidden_proxy": value.get("hidden_proxy"),
                            "last_val": (value["rounds"][-1]["validation"] if value["rounds"] else None)}
                      for key, value in summary["tasks"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
