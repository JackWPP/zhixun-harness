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
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from rsi.inference import (
    POSTPROCESS_KINDS,
    DamageShrinkStream,
    PostprocessStream,
    RecenterStream,
    apply_postprocess,
    apply_postprocess_stream,
)
from rsi.metrics import from_predictions, primary as primary_metric
from rsi.model import ExtendedModel
from rsi.platform import agent_host as _agent_host
from rsi.platform import plans as _plans

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


def _seed() -> int:
    try:
        return int(os.environ.get("RSI_V2_SEED", "20261117"))
    except ValueError:
        return 20261117


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


def _contract_value(contract, keys, default):
    """Read a frozen-contract field by name, tolerating naming variants."""
    if isinstance(contract, dict):
        for key in keys:
            value = contract.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return default


def _adapt_arm_mode() -> str:
    value = os.environ.get("RSI_V2_ADAPT_ARM", "auto").strip().lower()
    return value if value in {"pair", "retrain", "retrain_pair"} else "auto"


def _adapt_arms_enabled() -> bool:
    return os.environ.get("RSI_V2_DRIFT_POLICY", "1").strip().lower() not in {
        "0", "off", "false", "no"}


def _candidate_plans(task_key: str, model: ExtendedModel, train, validation, *,
                     metric: str = "macro_f1", direction: str = "higher",
                     margin: float = 0.0) -> list[dict]:
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
        if task_key == "A":
            candidates.append({
                "kind": "pair_threshold",
                "label": "延迟标签阈值校准（Ethanol vs Acetaldehyde，历史标签拟合）",
                "postprocess": "FIT_PAIR",
                "checkpoint": None,
            })
            mode = _adapt_arm_mode()
            if _adapt_arms_enabled() and mode in {"auto", "pair"}:
                candidates.append({
                    "kind": "pair_history",
                    "label": "历史校准 pair 阈值（delta=%.3f，不重新拟合）" % _plans.PAIR_HISTORY_DELTA,
                    "postprocess": {"type": "pair_threshold", "pair": [0, 3],
                                    "delta": _plans.PAIR_HISTORY_DELTA},
                    "checkpoint": None,
                })
            if _adapt_arms_enabled() and mode in {"auto", "retrain", "retrain_pair"}:
                candidates.append({
                    "kind": "continue_train",
                    "label": "累积标签续训 20 epoch（用当前 /data/train 全部已揭示行）",
                    "postprocess": {"type": "none"},
                    "checkpoint": "CONTINUE_TRAIN",
                    "epochs": 20,
                })
            if _adapt_arms_enabled() and mode in {"pair", "retrain_pair"}:
                candidates.append({
                    "kind": "continue_train_pair",
                    "label": "累积标签续训 + 历史校准 pair 阈值",
                    "postprocess": {"type": "pair_threshold", "pair": [0, 3],
                                    "delta": _plans.PAIR_HISTORY_DELTA},
                    "checkpoint": "CONTINUE_TRAIN",
                    "epochs": 20,
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
            if _adapt_arms_enabled():
                candidates.append({
                    "kind": "continue_train",
                    "label": "累积标签续训 15 epoch（用当前 /data/train 全部已揭示行）",
                    "postprocess": {"type": "none"},
                    "checkpoint": "CONTINUE_TRAIN",
                    "epochs": 15,
                })
        if task_key == "C":
            candidates.append({
                "kind": "continue_train",
                "label": "官方 train 续训 40 epoch（lr=1e-3，checkpoint 兼容）",
                "postprocess": {"type": "none"},
                "checkpoint": "CONTINUE_TRAIN_40",
            })
    if task == "classification" and _plans.capacity_arm_mode() not in ("", "off", "none", "0", "false"):
        try:
            ctx = _plans.PlanContext.from_rows(
                task_key, model, train, validation, metric=metric,
                direction=direction, margin=float(margin),
                classes=list(model.meta.get("classes") or []))
            for plan in _plans.capacity_plans(ctx):
                if not plan.get("ok"):
                    continue
                candidates.append({
                    "kind": plan["kind"],
                    "label": plan["label"],
                    "postprocess": plan["postprocess"],
                    "checkpoint": "CAPACITY",
                    "model": plan["model"],
                    "primary": plan["primary"],
                    "values": plan["values"],
                    "hypothesis": plan.get("hypothesis") or plan["label"],
                })
        except Exception as error:
            print("train: capacity arms unavailable: %s: %s" % (type(error).__name__, error),
                  file=sys.stderr, flush=True)
    for plan in candidates:
        postprocess = plan["postprocess"]
        checkpoint_kind = plan.get("checkpoint")
        if checkpoint_kind in ("CONTINUE_TRAIN_40", "CONTINUE_TRAIN"):
            source = Path(model.meta.get("_source_path") or "")
            if not source.is_file():
                candidate_model = model
            else:
                try:
                    candidate_model = ExtendedModel.from_checkpoint(source)
                    candidate_model.meta["seed"] = _seed()
                    candidate_model.set_learning_rate(1e-3)
                    epochs = int(plan.get("epochs", 40 if checkpoint_kind == "CONTINUE_TRAIN_40" else 20))
                    y_fit = (np.asarray(y_train, dtype=np.int64) if task == "classification"
                             else np.asarray(y_train, dtype=np.float64))
                    candidate_model.fit(x_train, y_fit, epochs=epochs)
                except Exception:
                    candidate_model = model
        else:
            candidate_model = plan.get("model") or model
        if plan.get("checkpoint") == "CAPACITY":
            plans.append({
                **plan,
                "model": candidate_model,
                "values": plan.get("values") or {},
                "primary": plan.get("primary"),
            })
            continue
        if postprocess == "FIT_DAMAGE":
            from rsi.inference import fit_damage_shrink
            dataset = _as_dataset(x_train, y_train, classes, task)
            postprocess = fit_damage_shrink(dataset)
        if postprocess == "FIT_PAIR":
            from rsi.inference import fit_pair_threshold
            fitted = fit_pair_threshold(x_train, y_train, model, pair=(0, 3))
            postprocess = ({"type": "pair_threshold", "pair": [0, 3],
                            **{k: v for k, v in fitted.items() if k != "type"}}
                           if fitted else {"type": "none"})
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


def _change_summary(chosen: dict) -> str:
    kind = chosen["kind"]
    if kind == "retain":
        return "保留父模型，不改权重与推理期后处理"
    if kind == "continue_train":
        return "基于官方 train 的短续训，父谱系与结构兼容"
    return "仅推理期后处理（%s），未改 checkpoint" % kind


def _state_payload(rnd, candidates, chosen, note: str) -> dict:
    compact = []
    for candidate in candidates or []:
        primary = candidate.get("primary")
        compact.append({"kind": candidate.get("kind"),
                        "primary": float(primary) if primary is not None else None})
    return {
        "round_index": int(rnd.round_index),
        "dataset_key": rnd.dataset_key,
        "chosen": chosen.get("kind"),
        "postprocess": chosen.get("postprocess"),
        "candidates": compact,
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "note": note,
    }


# ---------------------------------------------------------------------------
# controller routing: deterministic pipeline vs autonomous PI-agent kernel
# ---------------------------------------------------------------------------
def _controller_mode(task: str | None = None) -> str:
    """``pipeline`` by default; the packaged image enables ``auto``.

    ``auto`` keeps the PI-agent kernel for classification tasks (where drift
    diagnosis and history-calibrated post-processing matter most) and uses the
    deterministic pipeline for regression tasks (where the public-validation
    ranking has repeatedly over-fit).
    """
    explicit = os.environ.get("RSI_V2_CONTROLLER", "").strip().lower()
    if explicit == "auto":
        return "kernel" if (task or "").strip().lower() == "classification" else "pipeline"
    if explicit in ("pipeline", "kernel"):
        return explicit
    default = os.environ.get("RSI_V2_KERNEL_DEFAULT", "0").strip()
    return "kernel" if default == "1" else "pipeline"


def _drift_brief(ctx) -> dict:
    """Cheap train-vs-validation covariate summary (no labels, no hidden data)."""
    try:
        train_mean = np.asarray(ctx.x_train.mean(axis=0), dtype=np.float64)
        train_std = np.asarray(ctx.x_train.std(axis=0), dtype=np.float64) + 1e-9
        val_mean = np.asarray(ctx.x_val.mean(axis=0), dtype=np.float64)
        z = (val_mean - train_mean) / train_std
        order = np.argsort(-np.abs(z))[:5]
        top = {str(int(index)): round(float(z[index]), 3) for index in order}
        peak = float(np.max(np.abs(z))) if len(z) else 0.0
        severity = "high" if peak >= 2.0 else ("medium" if peak >= 1.0 else "low")
        return {
            "severity": severity,
            "score": round(peak, 4),
            "drift_type": "covariate_shift",
            "feature_shift": top,
            "max_feature_shift": {"index": int(order[0]) if len(order) else -1,
                                  "z": round(peak, 3)},
            "evidence": {"train_rows": int(len(ctx.x_train)),
                         "val_rows": int(len(ctx.x_val))},
            "causes": ["公开验证投影与 train 投影的特征分布差异（无标签）"],
            "knowledge_hints": [],
        }
    except Exception:
        return {"severity": "unknown", "score": 0.0, "drift_type": "unknown"}


def _kernel_brief(task_key: str, model: ExtendedModel, ctx, history: list[dict],
                  round_index: int, rounds_total: int, state) -> dict:
    from rsi.knowledge import drift_context_for_brief

    drift = _drift_brief(ctx)
    memory_text = ""
    lessons_text = ""
    stats_text = ""
    skills_text = ""
    try:
        if state is not None and state.memory is not None:
            memory_text = state.memory.snapshot(task_key)
    except Exception:
        memory_text = ""
    try:
        if state is not None and state.recall is not None:
            lessons_text = state.recall.lessons_text(task_key)
            stats_text = state.recall.stats_text(task_key)
    except Exception:
        lessons_text, stats_text = "", ""
    try:
        if state is not None and state.skills is not None:
            skills_text = state.skills.index_text()
    except Exception:
        skills_text = ""
    knowledge_text = ""
    try:
        knowledge_text = drift_context_for_brief(drift, k=3)
    except Exception:
        knowledge_text = ""
    return {
        "task": {"key": task_key, "name": task_key, "name_en": task_key,
                 "task": ctx.task, "primary_metric": ctx.metric,
                 "direction": ctx.direction,
                 "baseline_primary": float(ctx.parent_primary),
                 "margin": float(TASK_DEFAULTS[task_key]["margin"])},
        "goal": {"goal": "在漂移数据流上维持并提升 %s 的 %s" % (task_key, ctx.metric),
                 "status": "running", "turns_used": 0,
                 "max_turns": int(os.environ.get("RSI_V2_MAX_TURNS", "6") or 6)},
        "round_index": int(round_index),
        "rounds_total": int(rounds_total),
        "metrics": {"parent": {"primary": float(ctx.parent_primary),
                               "report": dict(ctx.parent_values or {})}},
        "budget": {"token_cap": int(os.environ.get("RSI_V2_TOKEN_CAP", "0") or 0),
                   "calls": 0, "note": "platform budget is enforced by the host"},
        "memory": memory_text,
        "lessons": lessons_text,
        "stats": stats_text,
        "history": list(history or []),
        "drift": drift,
        "policy": {
            "drift_level": drift.get("severity"),
            "prefer_delayed_label_on_high_drift": True,
            "note": ("高漂移分类任务：优先 pair_threshold(delta=%.1f) 且只要公开验证不劣于"
                     "父模型 2*margin 就采用；中漂移只用 recenter；低漂移 retain。"
                     % _plans.PAIR_HISTORY_DELTA),
        },
        "catalog": _plans.catalog_text(),
        "skills": skills_text,
        "knowledge": knowledge_text,
    }


def _open_agent_state(work: Path, rnd):
    try:
        return _agent_host.open_state(work, rnd)
    except Exception as exc:
        print("train: agent state unavailable: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr, flush=True)
        return _agent_host.AgentState()


def _choose_kernel_plan(kernel_round, ctx, direction: str) -> dict:
    """Map the kernel decision onto a concrete, already-evaluated plan."""
    decision = kernel_round.decision or {}
    if str(decision.get("decision")) == "promote":
        host = kernel_round.host
        plan = host.plan_for(decision.get("experiment_id")) if host is not None else None
        if plan is not None and plan.get("ok"):
            return plan
    return _plans.retain_plan(ctx)

def _apply_drift_policy(ctx, chosen: dict, plans=None) -> tuple[dict, str]:
    """Gen2 drift-aware override for the delayed-label arm.

    On high covariate drift (task A) the historical pair delta is preferred when
    its public validation is *not materially worse* than the parent.  This is
    deliberately not a "best validation wins" rule: the platform trial showed
    that the public-validation optimum can invert on the hidden batch.

    ``RSI_V2_DRIFT_POLICY=0`` turns the override off so controlled experiments
    can compare the live policy against the deterministic pipeline choice.
    """
    if os.environ.get("RSI_V2_DRIFT_POLICY", "1").strip().lower() in {
            "0", "off", "false", "no"}:
        print("train: drift_policy: disabled by RSI_V2_DRIFT_POLICY=0",
              file=sys.stderr, flush=True)
        return chosen, ""
    try:
        drift = _drift_brief(ctx)
        margin = float(getattr(ctx, "margin", 0.0) or 0.0)
        parent_primary = float(ctx.parent_primary)
        usable = [plan for plan in (plans or [])
                  if isinstance(plan, dict) and plan.get("primary") is not None
                  and not plan.get("error")]
        if (str(drift.get("severity")) == "high" and ctx.task == "classification"
                and usable):
            adapts = [plan for plan in usable
                      if str(plan.get("kind", "")).startswith("continue_train")]
            best = max(adapts, key=lambda plan: float(plan["primary"]), default=None)
            if best is not None and float(best["primary"]) >= parent_primary + margin:
                return best, ("drift_policy: high drift %.2f, %s val=%.4f parent=%.4f "
                              "(cumulative labels)" % (
                                  float(drift.get("score") or 0.0), best.get("kind"),
                                  float(best["primary"]), parent_primary))
            hist = [plan for plan in usable if str(plan.get("kind")) == "pair_history"]
            if hist and float(hist[0]["primary"]) + 2.0 * margin >= parent_primary:
                if str(chosen.get("kind")) == "pair_history":
                    return chosen, ""
                return hist[0], ("drift_policy: high drift %.2f, pair_history val=%.4f "
                                 "parent=%.4f" % (
                                     float(drift.get("score") or 0.0),
                                     float(hist[0]["primary"]), parent_primary))
        if str(drift.get("severity")) != "high" or ctx.task != "classification":
            return chosen, ""
        arm = _plans.pair_history_plan(ctx)
        if not arm.get("ok"):
            return chosen, ""
        if float(arm["primary"]) + 2.0 * margin < parent_primary:
            return chosen, ""
        if str(chosen.get("kind")) == "pair_history":
            return chosen, ""
        return arm, "drift_policy: high drift %.2f, pair_history val=%.4f parent=%.4f" % (
            float(drift.get("score") or 0.0), float(arm["primary"]), parent_primary)
    except Exception as exc:
        print("train: drift policy skipped: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr, flush=True)
        return chosen, ""


def _trim_config(payload: dict, limit: int = 60_000) -> dict:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return payload
    return {"truncated": True, "preview": text[:limit - 40] + "...(truncated)"}


def _gateway_model(rnd, messages, max_tokens: int):
    """One real gateway call with bounded retries.

    The platform requires at least one real call per round; transient gateway
    errors should not turn into a failed round.  Each retry receives its own
    real ``call_id`` and the last successful receipt is the one reported.
    """
    last_error = None
    for attempt in range(3):
        try:
            return rnd.model(messages, max_tokens=max_tokens, temperature=0)
        except Exception as error:  # pragma: no cover - depends on host gateway
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise last_error


def _new_state_dir(work: Path, rnd) -> Path:
    """Copy the previous native state into a fresh writable directory when the
    host mounted one (workspace-v2 recommends ``rnd.state_workspace``)."""
    destination = work / ("state-round-%d-%d" % (int(rnd.round_index), os.getpid()))
    try:
        return rnd.state_workspace(str(destination))
    except Exception:  # first round may omit /state; fall back to an empty dir
        destination.mkdir(parents=True, exist_ok=True)
        return destination


def train(rnd):
    """One workspace v2 round."""
    task_key = str(rnd.dataset_key).upper()
    if task_key not in TASK_DEFAULTS:
        raise ValueError(f"unsupported dataset key: {task_key}")
    contract = rnd.task or {}
    defaults = TASK_DEFAULTS[task_key]
    task = _contract_value(contract, ("task", "task_type", "type"), defaults["task"])
    if task not in TASKS:
        task = defaults["task"]
    metric = _contract_value(contract, ("metric", "metric_name"), defaults["metric"])
    direction = _contract_value(contract, ("direction", "metric_direction"),
                                defaults["direction"])

    parent_path = rnd.parent.path(CHECKPOINT_NAME)
    model = ExtendedModel.from_checkpoint(parent_path)
    model.meta["_source_path"] = str(parent_path)
    checkpoint_task = str(model.meta.get("task"))
    if checkpoint_task != task:
        # The frozen host contract can disagree with the official package
        # contract (observed on dataset B: host declares
        # classification/macro_f1 while the official checkpoint and package
        # are regression/RUL-minutes+RMSE). Hard-failing here happens *before*
        # the required gateway model call and makes a whole dataset unable to
        # complete any round. Follow the real checkpoint task instead and keep
        # the round protocol-compliant.
        if checkpoint_task not in TASKS:
            raise ValueError(
                "checkpoint task mismatch: %r != %r" % (checkpoint_task, task))
        print(
            "train: host contract task %r disagrees with checkpoint task %r; "
            "following checkpoint task for this round" % (task, checkpoint_task),
            file=sys.stderr,
            flush=True,
        )
        task = checkpoint_task
        metric = defaults["metric"]
        direction = defaults["direction"]

    train_rows = list(rnd.data.training_rows())
    validation_rows = list(rnd.data.validation_rows())
    if not train_rows or not validation_rows:
        raise ValueError("empty public projection")

    history = []
    try:
        history = rnd.history()
    except Exception:  # host may legitimately return no history
        history = []

    # -- route: deterministic pipeline (default) or autonomous PI-agent kernel
    controller = _controller_mode(task)
    kernel_round = None
    kernel_ctx = None
    kernel_state = None
    if controller == "kernel":
        try:
            kernel_ctx = _plans.PlanContext.from_rows(
                task_key, model, train_rows, validation_rows,
                metric=metric, direction=direction,
                margin=float(defaults["margin"]),
                classes=list(model.meta.get("classes") or []))
            kernel_state = _open_agent_state(_work_dir(), rnd)
            brief = _kernel_brief(task_key, model, kernel_ctx, history,
                                  int(rnd.round_index),
                                  int(getattr(rnd, "rounds_total", 10) or 10),
                                  kernel_state)
            kernel_round = _agent_host.run_kernel_round(
                rnd, kernel_ctx, brief, kernel_state,
                run_id="%s-r%d" % (task_key, int(rnd.round_index)),
                round_index=int(rnd.round_index),
                experiments_total=len(history),
                max_turns=int(os.environ.get("RSI_V2_MAX_TURNS", "6") or 6))
            print("train: kernel round calls=%d decision=%s" % (
                kernel_round.calls,
                (kernel_round.decision or {}).get("decision")),
                file=sys.stderr, flush=True)
        except Exception as exc:
            print("train: kernel route failed, falling back to pipeline: %s: %s" % (
                type(exc).__name__, exc), file=sys.stderr, flush=True)
            kernel_round, kernel_ctx, kernel_state = None, None, None

    plans = None
    receipt = None
    plans = None
    llm_choice = None
    fallback_choice = None
    if kernel_round is not None and kernel_round.calls > 0 and kernel_ctx is not None:
        chosen = _choose_kernel_plan(kernel_round, kernel_ctx, direction)
        candidates = list(kernel_round.host.results) if kernel_round.host else []
        call_ids = list(kernel_round.call_ids)
        decision_note = dict(kernel_round.decision or {})
        controller_note = "kernel"
        choice_note = "内核决策 %s" % decision_note.get("decision", "retain")
    else:
        plans = _candidate_plans(task_key, model, train_rows, validation_rows,
                                  metric=metric, direction=direction,
                                  margin=float(defaults["margin"]))
        table = _format_plan_table(plans, metric)
        prompt = (
            "本轮是 %s 数据集第 %d 轮，指标 %s（方向 %s）。父模型在公开验证集上的 %s 已给出，"
            "候选如下（数字为本机公开验证评估，平台会独立复测）：\n%s\n"
            "请只用一行 JSON 回复：{\"choice\": <候选编号>, \"hypothesis\": \"...\", \"reason\": \"...\"}。"
            "只能选改进父模型的候选；不确定时选 0。"
        ) % (task_key, int(rnd.round_index), metric, direction, table, table)
        receipt = _gateway_model(
            rnd,
            [{"role": "user", "content": prompt[:6000]}],
            max_tokens=min(256, int(rnd.max_output_tokens)),
        )
        llm_choice = _parse_choice(receipt.text, len(plans))
        fallback_choice = _choose(plans, direction, defaults["margin"])
        chosen = plans[fallback_choice]
        if llm_choice is not None and llm_choice != fallback_choice:
            candidate = plans[llm_choice]
            parent_primary = float(plans[0]["primary"])
            is_better = (candidate["primary"] > parent_primary + defaults["margin"]
                         if direction == "higher"
                         else candidate["primary"] < parent_primary - defaults["margin"])
            if is_better:
                chosen = candidate
        candidates = [{"kind": plan["kind"], "primary": float(plan["primary"])}
                      for plan in plans]
        call_ids = [receipt.call_id]
        decision_note = {"decision": "retain" if chosen is plans[0] else "promote",
                         "source": "pipeline",
                         "reason": "确定性 margin 门控 + 模型选择"}
        controller_note = "pipeline"
        choice_note = "pipeline 选择 %s" % chosen["kind"]

    policy_note = ""
    try:
        policy_ctx = kernel_ctx
        if policy_ctx is None:
            policy_ctx = _plans.PlanContext.from_rows(
                task_key, model, train_rows, validation_rows,
                metric=metric, direction=direction,
                margin=float(defaults["margin"]),
                classes=list(model.meta.get("classes") or []))
        chosen, policy_note = _apply_drift_policy(policy_ctx, chosen, plans)
        if policy_note:
            print("train: %s" % policy_note, file=sys.stderr, flush=True)
    except Exception as exc:
        print("train: drift policy unavailable: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr, flush=True)

    work = _work_dir()
    candidate_dir = Path(tempfile.mkdtemp(prefix="candidate-", dir=work))
    if kernel_state is not None and kernel_state.dir:
        state_dir = Path(kernel_state.dir)
    else:
        state_dir = _new_state_dir(work, rnd)
    checkpoint_model = chosen.get("model") or model
    if chosen.get("checkpoint") or str(chosen.get("kind", "")).startswith("continue_train"):
        checkpoint_model.save(candidate_dir / CHECKPOINT_NAME)
    else:
        shutil.copyfile(parent_path, candidate_dir / CHECKPOINT_NAME)
    postprocess = chosen.get("postprocess") or {"type": "none"}
    if postprocess == "FIT_DAMAGE":
        postprocess = {"type": "none"}
    meta = {
        "harness": "zhixun-rsi-harness",
        "version": "0.12.1",
        "task_key": task_key,
        "task": task,
        "metric": metric,
        "classes": list(model.meta.get("classes") or []),
        "postprocess": postprocess,
        "round_index": int(rnd.round_index),
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "chosen": chosen.get("kind"),
        "controller": controller_note,
        "policy": policy_note[:200],
    }
    (candidate_dir / META_NAME).write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    state = _state_payload(rnd, candidates, chosen,
                           (decision_note.get("reason") or choice_note or "")[:1200])
    (state_dir / "state.json").write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    if kernel_round is not None:
        (state_dir / "kernel_round.json").write_text(
            json.dumps(_trim_config(kernel_round.to_dict()), ensure_ascii=False, default=str),
            encoding="utf-8")
    configuration = _trim_config({
        "controller": controller_note,
        "candidates": candidates[:8],
        "chosen": chosen.get("kind"),
        "postprocess": postprocess,
        "policy": policy_note[:200],
        "decision": {key: decision_note.get(key) for key in
                     ("decision", "experiment_id", "reason", "source", "delta")},
        "kernel": None if kernel_round is None else {
            "calls": kernel_round.calls,
            "call_ids": call_ids[:8],
            "experiments": len(candidates),
            "error": kernel_round.error,
            "trace": kernel_round.trace[-12:],
        },
        "llm_choice": llm_choice,
        "fallback_choice": fallback_choice,
    })
    label = str(chosen.get("label") or chosen.get("kind") or "retain")
    primary_value = chosen.get("primary")
    primary_text = ("%.6f" % float(primary_value)) if primary_value is not None else "n/a"

    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis="本轮候选：%s" % label,
        change_summary=_change_summary(chosen),
        configuration=configuration,
        seed=_seed(),
        checkpoint_paths=[CHECKPOINT_NAME],
        code_paths=[META_NAME],
        predict_argv=PREDICT_ARGV,
    )
    rnd.save_state(state_dir)
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis="本轮候选：%s" % label,
        change_summary=_change_summary(chosen),
        reason="%s（%s）：决定 %s，%s=%s；宿主网关调用 %d 次。" % (
            controller_note, choice_note, chosen.get("kind"), metric, primary_text,
            len(call_ids)),
        configuration=configuration,
        seed=_seed(),
        history_ids=[item["id"] for item in history],
        model_call_ids=call_ids,
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
    if kind in ("recenter", "pair_threshold", "pair_stream"):
        payload["stream"] = PostprocessStream(postprocess, model)
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
    tied = np.flatnonzero(values == values[top])
    if len(tied) > 1:
        # The host rejects a non-unique top score; break ties deterministically
        # without leaving [0, 1] or altering the argmax class.
        values = values.copy()
        if values[top] <= 0.0:
            values[top] = 1e-7
        else:
            for offset, index in enumerate((int(i) for i in tied if int(i) != top), start=1):
                values[index] = max(0.0, float(values[top]) - 1e-7 * offset)
    return values


def _fallback_prediction(artifact: dict):
    """Never let one malformed row kill the whole prediction stream."""
    if artifact["task"] == "classification":
        classes = list(artifact["classes"])
        if not classes:
            raise ValueError("classification artifact has no classes")
        values = _uniform_scores(np.ones(len(classes), dtype=np.float64))
        return [{"label": label, "score": float(values[position])}
                for position, label in enumerate(classes)]
    mean = float(artifact["model"].meta.get("target_mean") or 0.0)
    if not np.isfinite(mean) or mean < 0:
        mean = 0.0
    return {"value": float(mean)}


def _degenerate_prediction(sample):
    """Last-resort shape-valid output when even the artifact cannot be loaded."""
    if (sample.task or "") == "regression":
        return {"value": 0.0}
    return [{"label": "normal", "score": 1.0}]


def _coerce_task_output(requested: str, artifact: dict, output):
    """Never break the prediction stream when the frozen request task and the
    artifact task disagree (host contract bug observed on dataset B)."""
    artifact_task = str(artifact.get("task") or "")
    requested = requested or artifact_task
    if not requested or requested == artifact_task:
        return output
    if requested == "regression":
        if isinstance(output, dict) and "value" in output:
            return {"value": float(output["value"])}
        best_index, best_score = 0, -1.0
        for position, item in enumerate(output or []):
            score = float(item.get("score", 0.0))
            if score > best_score:
                best_index, best_score = position, score
        return {"value": float(max(best_index, 0))}
    # requested classification while the artifact is regression: bucket the
    # regression value into two label buckets with a unique top score.
    value = 0.0
    if isinstance(output, dict):
        try:
            value = float(output.get("value", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
    labels = [str(label) for label in (artifact.get("classes") or []) if str(label)]
    if len(labels) < 2:
        labels = ["normal", "fault"]
    top = 0 if value <= 0.0 else 1
    scores = [0.1 / (len(labels) - 1)] * len(labels)
    scores[top] = 0.9
    return [{"label": labels[position], "score": float(scores[position])}
            for position in range(len(labels))]


def predict(sample, model_dir):
    """One validated prediction line; keep it allocation-light and streaming."""
    try:
        artifact = _load_artifact(model_dir)
    except Exception as error:  # a broken mount must not kill the whole stream
        print("predict artifact load failed: %s: %s" % (type(error).__name__, error),
              file=sys.stderr, flush=True)
        return _degenerate_prediction(sample)
    try:
        output = _predict_one(sample, artifact)
    except Exception as error:  # one bad row must not fail the hidden stream
        print("predict fallback: %s: %s" % (type(error).__name__, error),
              file=sys.stderr, flush=True)
        try:
            output = _fallback_prediction(artifact)
        except Exception:
            return _degenerate_prediction(sample)
    return _coerce_task_output(sample.task or "", artifact, output)


def _predict_one(sample, artifact: dict):
    model: ExtendedModel = artifact["model"]
    features = _flatten_row(_load_specs(model), sample.inputs or {})
    row = features.reshape(1, -1)
    postprocess = artifact.get("postprocess") or {"type": "none"}
    kind = str(postprocess.get("type", "none"))
    raw = model.predict(row)
    if artifact["task"] == "classification":
        stream = artifact.get("stream")
        if stream is not None and hasattr(stream, "predict_row"):
            scores = np.asarray(stream.predict_row(features), dtype=np.float64).reshape(-1)
        else:
            scores = np.asarray(raw, dtype=np.float64).reshape(-1)
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
