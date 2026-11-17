"""Workspace v2 participant entry (``contest.harness.workspace.v2``).

The platform runs ``/app/harness.py train`` for every round and later
``/app/harness.py predict`` for the selected artifact.  This module is the
participant implementation used by both commands:

``train(rnd)``
    Reads the public training/validation projection, evaluates the parent
    checkpoint, runs one model call through the host gateway, builds at most
    two candidate post-processing/continuation variants on the *official
    train split only*, gates them on validation, uploads the winner as the
    round candidate, persists state and finishes the round with a real
    ``model_call_id`` and ``history_ids``.

``predict(sample, model_dir)``
    Loads the selected checkpoint plus its ``submission_meta.json`` and
    answers one JSONL line at a time.  Post-processing is applied per sample
    (causal streams), so the module stays compatible with the batch-scored
    hidden stream while never reading an answer column.

Design notes
------------
* The checkpoint stays a byte-compatible continuation of the official
  baseline: continuation candidates only call ``Model.fit`` with the official
  ``public/train.jsonl`` rows; inference-time calibration lives in the
  artifact metadata (``submission_meta.json``).
* Everything is bounded: one gateway call per round, at most two candidates,
  no writes outside ``/workspace``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from rsi.inference import (
    POSTPROCESS_KINDS,
    DamageShrinkStream,
    RecenterStream,
    apply_postprocess,
)
from rsi.metrics import from_predictions, primary as primary_metric
from rsi.model import ExtendedModel

TASKS = ("classification", "regression")

TASK_DEFAULTS = {
    "A": {"task": "classification", "metric": "macro_f1", "direction": "higher", "margin": 0.004},
    "B": {"task": "regression", "metric": "rmse", "direction": "lower", "margin": 5.0},
    "C": {"task": "regression", "metric": "rmse", "direction": "lower", "margin": 0.3},
}
TASK_KEYS = "ABC"
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
META_NAME = "submission_meta.json"
CHECKPOINT_NAME = "checkpoint.npz"


def _work_dir() -> Path:
    configured = os.environ.get("CONTEST_WORK")
    if configured:
        path = Path(configured)
    elif Path("/workspace").is_dir():
        path = Path("/workspace")
    else:
        path = Path(tempfile.mkdtemp(prefix="contest-workspace-"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "id" not in row or "inputs" not in row:
                raise ValueError(f"invalid public projection row in {path}")
            rows.append(row)
    return rows


def _load_specs(model: ExtendedModel) -> list[list[list[str]]]:
    raw = model.meta.get("paths") or []
    specs = []
    for item in raw:
        spec = json.loads(item) if isinstance(item, str) else item
        specs.append([list(entry) for entry in spec])
    return specs


def _flatten_row(specs: list[list[list[str]]], inputs: dict) -> np.ndarray:
    values = []
    for spec in specs:
        current = inputs
        for kind, argument in spec:
            if kind == "key":
                current = current[argument]
            elif kind == "index":
                current = current[argument]
            else:  # pragma: no cover - malformed checkpoint metadata
                raise ValueError(f"unsupported path spec: {spec}")
        values.append(float(current))
    return np.asarray(values, dtype=np.float64)


def _xy_rows(rows: list[dict], model: ExtendedModel, classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    specs = _load_specs(model)
    features = np.empty((len(rows), len(specs)), dtype=np.float64)
    task = model.meta.get("task")
    labels = np.empty(len(rows), dtype=np.float64)
    index = {name: position for position, name in enumerate(classes)}
    for row, position in zip(rows, range(len(rows))):
        features[position] = _flatten_row(specs, row["inputs"])
        answer = row.get("answer")
        if task == "classification":
            if answer not in index:
                raise ValueError(f"unknown class label: {answer!r}")
            labels[position] = index[answer]
        else:
            labels[position] = float(answer)
    return features, labels, np.arange(len(rows))


def _evaluate(model: ExtendedModel, features: np.ndarray, labels: np.ndarray, task: str,
              classes: list[str], postprocess: dict | None) -> tuple[dict, float]:
    predicted = apply_postprocess(postprocess or {"type": "none"}, model, features)
    values = from_predictions(task, predicted, labels, classes)
    primary = values.get("primary")
    if primary is None:
        primary = primary_metric(task, values)
    return values, float(primary)


def _choose(rows: list[dict], direction: str, margin: float) -> int:
    """Deterministic fallback: best candidate that beats the parent margin."""
    parent = next((row for row in rows if row["kind"] == "retain"), rows[0])
    best_position = 0
    best_value = float(parent["primary"])
    for position, row in enumerate(rows):
        if position == 0:
            continue
        value = float(row["primary"])
        better = (value < best_value) if direction == "lower" else (value > best_value)
        if better:
            best_position, best_value = position, value
    if best_position == 0:
        return 0
    parent_value = float(parent["primary"])
    if direction == "lower":
        return best_position if best_value < parent_value - margin else 0
    return best_position if best_value > parent_value + margin else 0


def _parse_choice(text: str, upper: int) -> int | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"\{[^{}]*\}", text, re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            choice = payload.get("choice")
            if isinstance(choice, int) and 0 <= choice < upper:
                return choice
            if isinstance(choice, str):
                digits = re.search(r"\d+", choice)
                if digits and 0 <= int(digits.group(0)) < upper:
                    return int(digits.group(0))
        except json.JSONDecodeError:
            pass
    digits = re.search(r"\b([0-9])\b", text)
    if digits and 0 <= int(digits.group(1)) < upper:
        return int(digits.group(1))
    return None


def _candidate_plans(task_key: str, model: ExtendedModel, train, validation) -> list[dict]:
    """Build candidate plan descriptors sorted by expected value (cheap first)."""
    task = model.meta.get("task")
    classes = list(model.meta.get("classes") or [])
    target_mean = float(model.meta.get("target_mean") or 0.0)
    x_train, y_train, _ = _xy_rows(train, model, classes)
    x_val, y_val, _ = _xy_rows(validation, model, classes)
    parent_values, parent_primary = _evaluate(model, x_val, y_val, task, classes, None)
    plans: list[dict] = [{
        "kind": "retain",
        "label": "保留父模型（不做任何改动）",
        "postprocess": {"type": "none"},
        "values": parent_values,
        "primary": parent_primary,
        "checkpoint": None,
    }]
    candidates = []
    if task == "classification":
        for weight, mode in ((0.4, "batch_mean"), (0.2, "batch_std")):
            candidates.append({
                "kind": "recenter",
                "label": f"推理期 recenter（{mode}, w={weight}），不改权重",
                "postprocess": {"type": "recenter", "mode": mode, "weight": weight},
                "checkpoint": None,
            })
    else:
        candidates.append({
            "kind": "shrink",
            "label": "train 目标均值收缩校准（α=0.3），不改权重",
            "postprocess": {"type": "shrink", "alpha": 0.3, "center": target_mean},
            "checkpoint": None,
        })
        if task_key == "B":
            candidates.append({
                "kind": "damage_shrink",
                "label": "damage_shrink 因果分段校准（train 标签拟合）",
                "postprocess": "FIT_DAMAGE",
                "checkpoint": None,
            })
        if task_key == "C":
            candidates.append({
                "kind": "continue_train",
                "label": "官方 train 续训 40 epoch（lr=1e-3，checkpoint 兼容）",
                "postprocess": {"type": "none"},
                "checkpoint": "CONTINUE_TRAIN_40",
            })
    for plan in candidates:
        postprocess = plan["postprocess"]
        checkpoint_kind = plan.get("checkpoint")
        if checkpoint_kind == "CONTINUE_TRAIN_40":
            candidate_model = ExtendedModel.from_checkpoint(Path(model.meta.get("_source_path", "")))
            candidate_model.set_learning_rate(1e-3)
            candidate_model.fit(x_train, y_train, epochs=40)
        else:
            candidate_model = model
        if postprocess == "FIT_DAMAGE":
            from rsi.inference import fit_damage_shrink
            dataset = _as_dataset(x_train, y_train, classes, task)
            postprocess = fit_damage_shrink(dataset)
        values, primary = _evaluate(candidate_model, x_val, y_val, task, classes, postprocess)
        plans.append({
            **plan,
            "postprocess": postprocess,
            "values_values": None,
            "values": values,
            "primary": primary,
            "model": candidate_model,
            "y_train": y_train,
        })
    return plans


def _as_dataset(x, y, classes, task):
    from rsi.data import Dataset

    return Dataset(name="official-train", ids=[str(i) for i in range(len(x))], x=x, y=y,
                   paths=[], sha256="", groups=["all"] * len(x))


def _format_plan_table(plans: list[dict], metric: str) -> str:
    lines = []
    for position, plan in enumerate(plans):
        lines.append("%d. [%s] %s -> %s=%.6f" % (
            position, plan["kind"], plan["label"], metric, float(plan["primary"])))
    return "\n".join(lines)


def _state_payload(rnd, plans, chosen, note: str) -> dict:
    return {
        "round_index": int(rnd.round_index),
        "dataset_key": rnd.dataset_key,
        "chosen": chosen["kind"],
        "postprocess": chosen["postprocess"],
        "candidates": [{"kind": plan["kind"], "primary": float(plan["primary"])} for plan in plans],
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "note": note,
    }


def train(rnd):
    """One workspace v2 round."""
    task_key = str(rnd.dataset_key).upper()
    if task_key not in TASK_DEFAULTS:
        raise ValueError(f"unsupported dataset key: {task_key}")
    contract = rnd.task or {}
    task = contract.get("task") or TASK_DEFAULTS[task_key]["task"]
    metric = contract.get("metric") or TASK_DEFAULTS[task_key]["metric"]

    parent_path = rnd.parent.path(CHECKPOINT_NAME)
    model = ExtendedModel.from_checkpoint(parent_path)
    model.meta["_source_path"] = str(parent_path)
    if str(model.meta.get("task")) != task:
        raise ValueError(f"checkpoint task mismatch: {model.meta.get('task')} != {task}")

    train_rows = list(rnd.data.training_rows())
    validation_rows = list(rnd.data.validation_rows())
    if not train_rows or not validation_rows:
        raise ValueError("empty public projection")

    plans = _candidate_plans(task_key, model, train_rows, validation_rows)
    history = []
    try:
        history = rnd.history()
    except Exception:  # host may legitimately return no history
        history = []
    table = _format_plan_table(plans, metric)
    prompt = (
        "本轮是 %s 数据集第 %d 轮，指标 %s（方向 %s）。父模型在公开验证集上的 %s 已给出，"
        "候选如下（数字为本机公开验证评估，平台会独立复测）：\n%s\n"
        "请只用一行 JSON 回复：{\"choice\": <候选编号>, \"hypothesis\": \"...\", \"reason\": \"...\"}。"
        "只能选改进父模型的候选；不确定时选 0。"
    ) % (task_key, int(rnd.round_index), metric,
         TASK_DEFAULTS[task_key]["direction"], table, table)
    receipt = rnd.model(
        [{"role": "user", "content": prompt[:6000]}],
        max_tokens=min(256, int(rnd.max_output_tokens)),
        temperature=0,
    )
    llm_choice = _parse_choice(receipt.text, len(plans))
    fallback = _choose(plans, TASK_DEFAULTS[task_key]["direction"], TASK_DEFAULTS[task_key]["margin"])
    chosen = plans[fallback]
    if llm_choice is not None and llm_choice != fallback:
        candidate = plans[llm_choice]
        parent_primary = float(plans[0]["primary"])
        is_better = (candidate["primary"] > parent_primary + TASK_DEFAULTS[task_key]["margin"]
                     if TASK_DEFAULTS[task_key]["direction"] == "higher"
                     else candidate["primary"] < parent_primary - TASK_DEFAULTS[task_key]["margin"])
        if is_better:
            chosen = candidate

    work = _work_dir()
    candidate_dir = Path(tempfile.mkdtemp(prefix="candidate-", dir=work))
    state_dir = Path(tempfile.mkdtemp(prefix="state-", dir=work))
    checkpoint_model = chosen.get("model") or model
    if chosen["kind"] in ("continue_train",):
        checkpoint_model.save(candidate_dir / CHECKPOINT_NAME)
    else:
        shutil.copyfile(parent_path, candidate_dir / CHECKPOINT_NAME)
    postprocess = chosen["postprocess"]
    if postprocess == "FIT_DAMAGE":
        postprocess = {"type": "none"}
    meta = {
        "harness": "zhixun-rsi-harness",
        "version": "0.9.0",
        "task_key": task_key,
        "task": task,
        "metric": metric,
        "classes": list(model.meta.get("classes") or []),
        "postprocess": postprocess,
        "round_index": int(rnd.round_index),
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "chosen": chosen["kind"],
    }
    (candidate_dir / META_NAME).write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    state = _state_payload(rnd, plans, chosen, receipt.text[:1200])
    (state_dir / "state.json").write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis="本轮候选：%s" % chosen["label"],
        change_summary=("仅推理期后处理，未改 checkpoint" if chosen["kind"] != "continue_train"
                        else "基于官方 train 的短续训，父谱系与结构兼容"),
        configuration={"candidates": [{"kind": plan["kind"], "primary": float(plan["primary"])}
                                      for plan in plans],
                       "chosen": chosen["kind"],
                       "llm_choice": llm_choice,
                       "fallback_choice": fallback},
        seed=20261117,
        checkpoint_paths=[CHECKPOINT_NAME],
        code_paths=[META_NAME],
        predict_argv=PREDICT_ARGV,
    )
    rnd.save_state(state_dir)
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis="本轮候选：%s" % chosen["label"],
        change_summary=("仅推理期后处理，未改 checkpoint" if chosen["kind"] != "continue_train"
                        else "基于官方 train 的短续训，父谱系与结构兼容"),
        reason="依公开验证指标与模型网关选择 %s（%s=%.6f）；模型调用 %s。" % (
            chosen["kind"], metric, float(chosen["primary"]), receipt.call_id),
        configuration={"chosen": chosen["kind"], "postprocess": postprocess},
        seed=20261117,
        history_ids=[item["id"] for item in history],
    )


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _artifact_path(model_dir) -> Path:
    raw = os.environ.get("CONTEST_ARTIFACT_PATH")
    if raw:
        return Path(raw)
    base = Path(model_dir or "/model")
    for name in (CHECKPOINT_NAME, "model.npz"):
        if (base / name).is_file():
            return base / name
    found = sorted(base.glob("*.npz"))
    if not found:
        raise FileNotFoundError(f"no checkpoint found under {base}")
    return found[0]


def _load_artifact(model_dir) -> dict:
    path = _artifact_path(model_dir)
    key = str(path)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    model = ExtendedModel.from_checkpoint(path)
    meta = {}
    meta_path = path.parent / META_NAME
    if not meta_path.is_file() and os.environ.get("CONTEST_ARTIFACT_MANIFEST"):
        try:
            manifest = json.loads(Path(os.environ["CONTEST_ARTIFACT_MANIFEST"]).read_text(encoding="utf-8"))
            if isinstance(manifest, dict) and isinstance(manifest.get("submission_meta"), dict):
                meta = manifest["submission_meta"]
        except (OSError, ValueError):
            meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    postprocess = meta.get("postprocess") or {"type": "none"}
    task = meta.get("task") or model.meta.get("task")
    classes = list(meta.get("classes") or model.meta.get("classes") or [])
    if task not in TASKS:
        raise ValueError(f"unknown task in artifact: {task}")
    payload = {
        "model": model,
        "task": task,
        "classes": classes,
        "postprocess": postprocess,
        "stream": None,
    }
    kind = str(postprocess.get("type", "none"))
    if kind == "recenter":
        payload["stream"] = RecenterStream(
            model.mean, model.scale, mode=postprocess.get("mode", "batch_mean"), window=256, min_count=8)
    elif kind == "damage_shrink":
        payload["stream"] = DamageShrinkStream(postprocess)
    _CACHE.clear()
    _CACHE[key] = payload
    return payload


def _uniform_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        values = np.full(len(values), 1.0 / max(len(values), 1))
    else:
        values = values / total
    top = int(np.argmax(values))
    if int((values == values[top]).sum()) != 1:
        values = values.copy()
        values[top] = min(1.0, float(values[top]) + 1e-9)
    return values


def predict(sample, model_dir):
    """One validated prediction line; keep it allocation-light and streaming."""
    artifact = _load_artifact(model_dir)
    model: ExtendedModel = artifact["model"]
    task = sample.task or artifact["task"]
    if task != artifact["task"]:
        raise ValueError(f"task mismatch: request={task} artifact={artifact['task']}")
    features = _flatten_row(_load_specs(model), sample.inputs or {})
    row = features.reshape(1, -1)
    postprocess = artifact.get("postprocess") or {"type": "none"}
    kind = str(postprocess.get("type", "none"))
    raw = model.predict(row)
    if artifact["task"] == "classification":
        scores = np.asarray(raw, dtype=np.float64).reshape(-1)
        if kind == "recenter" and artifact.get("stream") is not None:
            weight = float(postprocess.get("weight", 0.4))
            calibrated = model.predict(artifact["stream"].calibrate(features).reshape(1, -1))
            scores = (1.0 - weight) * scores + weight * np.asarray(calibrated).reshape(-1)
        elif kind == "pair_threshold":
            left, right = (int(value) for value in postprocess["pair"])
            logits = model.predict_logits(row)[0]
            score = float(logits[right] - logits[left])
            predicted = int(np.argmax(scores))
            if predicted == left and score < float(postprocess["delta"]):
                scores = scores.copy()
                scores[left] = 0.0
                scores[right] = 1.0
            elif predicted == right and score >= float(postprocess["delta"]):
                scores = scores.copy()
                scores[right] = 0.0
                scores[left] = 1.0
        values = _uniform_scores(scores)
        return [
            {"label": label, "score": float(values[position])}
            for position, label in enumerate(artifact["classes"])
        ]
    value = float(np.asarray(raw).reshape(-1)[0])
    if kind == "shrink":
        alpha = float(postprocess.get("alpha", 0.3))
        center = float(postprocess.get("center", model.meta.get("target_mean", 0.0)))
        value = alpha * value + (1.0 - alpha) * center
    elif kind == "damage_shrink" and artifact.get("stream") is not None:
        value = artifact["stream"].transform(features, value)
    if not np.isfinite(value) or value < 0:
        value = max(0.0, float(model.meta.get("target_mean", 0.0)) if not np.isfinite(value) else 0.0)
    return {"value": float(value)}
